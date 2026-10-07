"""Med-NCA (Kalkhof et al., IPMI 2023) as a pathseg semantic segmenter.

Port of the backbone NCA and the multi-level forward of
https://github.com/MECLabTUDA/Med-NCA (commit a844a72, MIT License,
Copyright (c) 2020 Ming). Only the model is ported; the upstream training
loop, datasets and losses are not used. Deviations from upstream are logged in
``experiments/mednca/README.md``. The fast NCA step and the multi-level
encoder machinery are shared with OctreeNCA in ``pathseg.models.nca``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pathseg.models.nca import FoldedNCA, MultiLevelNCAEncoder
from pathseg.models.semantic_segmenter import (
    FeatureMaps,
    SemanticLogits,
    SemanticSegmenter,
)


class BackboneNCA(FoldedNCA):
    """One NCA level: upstream ``BackboneNCA`` (perception) + ``BasicNCA`` (update).

    The state is channels-last ``B x H x W x channel_n`` with layout
    ``[image | outputs | hidden]``. The first ``input_channels`` channels are
    re-injected after every step and are therefore never updated.
    """

    def __init__(
        self,
        channel_n: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        input_channels: int = 3,
    ) -> None:
        super().__init__(channel_n, fire_rate=fire_rate, input_channels=input_channels)

        # Same parameter names and creation order as upstream, so state dicts
        # are interchangeable and seeded initialization matches.
        self.fc0 = nn.Linear(self.channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.p0 = nn.Conv2d(
            self.channel_n,
            self.channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.p1 = nn.Conv2d(
            self.channel_n,
            self.channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def folded_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The update rule as ``(kernel, bias, fc1_weight)``, built from the upstream parameters.

        Upstream computes ``fc0(cat(x, p0(x), p1(x)))`` on ``x.transpose(1, 3)``,
        i.e. on the H/W-transposed grid. That is linear in ``x``, so it equals a
        single 3x3 conv with reflect padding, with kernel
        ``fc0_p0 @ p0 + fc0_p1 @ p1 (+ fc0_x at the centre tap)`` and bias
        ``fc0.bias + fc0_p0 @ p0.bias + fc0_p1 @ p1.bias``. A conv on the
        transposed grid equals a conv on the original grid with spatially
        transposed kernels (reflect padding commutes with the transpose), so
        the state stays channels-last and cuDNN gets a free NHWC view.

        ``fc1``'s rows for the image channels are zeroed: their update is
        discarded by upstream's re-injection, so ``dx`` is exactly 0 there.

        Built in fp32 (outside autocast) once per ``forward``; gradients flow to
        the original parameters, so the state_dict is unchanged.
        """
        with torch.autocast(self.fc0.weight.device.type, enabled=False):
            w_x, w_p0, w_p1 = self.fc0.weight.split(self.channel_n, dim=1)
            kernel = torch.einsum("hd,dcij->hcji", w_p0, self.p0.weight)
            kernel = kernel + torch.einsum("hd,dcij->hcji", w_p1, self.p1.weight)
            kernel = kernel + F.pad(w_x[..., None, None], (1, 1, 1, 1))
            bias = self.fc0.bias + w_p0 @ self.p0.bias + w_p1 @ self.p1.bias
            fc1_weight = F.pad(
                self.fc1.weight[self.input_channels :], (0, 0, self.input_channels, 0)
            )
        return kernel, bias, fc1_weight

    def fire_mask(self, x: torch.Tensor, fire_rate: float) -> torch.Tensor:
        # Deviation: upstream draws the mask on the CPU generator and copies it
        # to the device. Same distribution; identical to upstream on CPU.
        return (
            torch.rand([x.size(0), x.size(1), x.size(2), 1], device=x.device)
            > fire_rate
        )


class MedNCAEncoder(MultiLevelNCAEncoder):
    """Two-level Med-NCA (upstream ``Agent_Med_NCA.get_outputs``, inference path).

    The coarse level runs on the image downscaled by ``scale_factor``. Its state
    is upscaled (nearest) to the input size, the full-resolution image is
    re-injected, and the fine level runs. The stages are public so training
    code can recombine them; stage states are channels-last (upstream layout).
    Eval-time averaging (``n_eval_runs``), gradient checkpointing,
    ``max_batch_size`` and ``compile_step``: see ``MultiLevelNCAEncoder``.
    """

    level_names = ("coarse", "fine")

    def __init__(
        self,
        output_channels: int,
        *,
        channel_n: int = 64,
        hidden_size: int = 128,
        steps: int = 64,
        fire_rate: float = 0.5,
        scale_factor: int = 4,
        n_eval_runs: int = 1,
        grad_checkpointing_every: int | None = None,
        max_batch_size: int | None = None,
        compile_step: bool = False,
        input_channels: int = 3,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
        super().__init__(
            output_channels,
            channel_n=channel_n,
            n_eval_runs=n_eval_runs,
            grad_checkpointing_every=grad_checkpointing_every,
            max_batch_size=max_batch_size,
            input_channels=input_channels,
        )
        self.steps = int(steps)
        self.scale_factor = int(scale_factor)

        self.levels = nn.ModuleDict(
            {
                name: BackboneNCA(
                    self.channel_n,
                    fire_rate=fire_rate,
                    hidden_size=hidden_size,
                    input_channels=self.input_channels,
                )
                for name in self.level_names
            }
        )
        self.set_compile_step(compile_step)

    def level_modules(self):
        return self.levels.values()

    def level_module(self, level: str) -> BackboneNCA:
        if level not in self.levels:
            raise KeyError(f"Unknown level {level!r}. Available: {list(self.levels)}.")
        return self.levels[level]

    def level_steps(self, level: str) -> int:
        return self.steps

    def downscale(self, imgs: torch.Tensor) -> torch.Tensor:
        """``B x 3 x H x W`` -> ``B x 3 x H/s x W/s`` (linear, no antialiasing).

        Samples the same positions as upstream's ``torchio.Resize``.
        """
        size = (
            imgs.shape[-2] // self.scale_factor,
            imgs.shape[-1] // self.scale_factor,
        )
        return F.interpolate(
            imgs,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

    def forward_state(self, imgs: torch.Tensor) -> torch.Tensor:
        """One stochastic pass; returns the channels-last fine-level state."""
        small_imgs = self.downscale(imgs)
        state = self.run_level("coarse", self.init_state(small_imgs), small_imgs)
        state = self.upscale_state(state, imgs.shape[-2:])
        return self.run_level("fine", state, imgs)


class StateSliceDecoder(nn.Module):
    """Parameter-free decoder reading each task's logits from the NCA state.

    Output channels follow the image channels, one contiguous block per task in
    ``num_classes_by_task`` order. Task selection mirrors ``MultiTaskDecoder``.
    """

    def __init__(
        self,
        num_classes_by_task: dict[str, int],
        *,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if not num_classes_by_task:
            raise ValueError("num_classes_by_task must contain at least one task.")

        self._num_classes_by_task = {
            task: int(num_classes) for task, num_classes in num_classes_by_task.items()
        }
        self.input_channels = int(input_channels)

        self.slices: dict[str, slice] = {}
        start = self.input_channels
        for task, num_classes in self._num_classes_by_task.items():
            self.slices[task] = slice(start, start + num_classes)
            start += num_classes

    @property
    def num_classes_by_task(self) -> dict[str, int]:
        return dict(self._num_classes_by_task)

    @property
    def output_channels(self) -> int:
        return sum(self._num_classes_by_task.values())

    def forward(
        self,
        feature_maps: FeatureMaps,
        task: str | None = None,
    ) -> SemanticLogits:
        state = feature_maps[-1]

        if task is not None:
            if task not in self.slices:
                raise KeyError(
                    f"Unknown task {task!r}. Available tasks: {list(self.slices)}."
                )

            return {
                task: state[:, self.slices[task]],
            }

        return {name: state[:, task_slice] for name, task_slice in self.slices.items()}


class MedNCASegmenter(SemanticSegmenter):
    """Med-NCA whose output state channels are the semantic logits."""

    def __init__(
        self,
        num_classes_by_task: dict[str, int],
        *,
        channel_n: int = 64,
        hidden_size: int = 128,
        steps: int = 64,
        fire_rate: float = 0.5,
        scale_factor: int = 4,
        n_eval_runs: int = 1,
        grad_checkpointing_every: int | None = None,
        max_batch_size: int | None = None,
        compile_step: bool = False,
    ) -> None:
        decoder = StateSliceDecoder(num_classes_by_task, input_channels=3)

        encoder = MedNCAEncoder(
            output_channels=decoder.output_channels,
            channel_n=channel_n,
            hidden_size=hidden_size,
            steps=steps,
            fire_rate=fire_rate,
            scale_factor=scale_factor,
            n_eval_runs=n_eval_runs,
            grad_checkpointing_every=grad_checkpointing_every,
            max_batch_size=max_batch_size,
            compile_step=compile_step,
            input_channels=decoder.input_channels,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            upsample_logits=False,
        )
