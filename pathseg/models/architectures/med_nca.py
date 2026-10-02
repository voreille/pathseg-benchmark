"""Med-NCA (Kalkhof et al., IPMI 2023) as a pathseg semantic segmenter.

Port of the backbone NCA and the multi-level forward of
https://github.com/MECLabTUDA/Med-NCA (commit a844a72, MIT License,
Copyright (c) 2020 Ming). Only the model is ported; the upstream training
loop, datasets and losses are not used. Deviations from upstream are logged in
``experiments/mednca/README.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from pathseg.models.semantic_segmenter import (
    FeatureMaps,
    SemanticLogits,
    SemanticSegmenter,
)


class BackboneNCA(nn.Module):
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
        super().__init__()
        self.channel_n = int(channel_n)
        self.input_channels = int(input_channels)
        self.fire_rate = float(fire_rate)

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

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.p0(x), self.p1(x)), 1)

    def update(self, x: torch.Tensor, fire_rate: float | None = None) -> torch.Tensor:
        if fire_rate is None:
            fire_rate = self.fire_rate

        # Upstream transposes dims 1 and 3 (B x H x W x C -> B x C x W x H),
        # so the perception convs run on the H/W-transposed grid. Kept as is
        # for weight compatibility.
        dx = self.perceive(x.transpose(1, 3)).transpose(1, 3)
        dx = self.fc1(F.relu(self.fc0(dx)))

        # Deviation: upstream draws the mask on the CPU generator and copies it
        # to the device. Same distribution; identical to upstream on CPU.
        stochastic = (
            torch.rand([dx.size(0), dx.size(1), dx.size(2), 1], device=dx.device)
            > fire_rate
        )
        dx = dx * stochastic.float()

        return x + dx

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 64,
        fire_rate: float | None = None,
    ) -> torch.Tensor:
        for _ in range(steps):
            x_next = self.update(x, fire_rate)
            x = torch.cat(
                (x[..., : self.input_channels], x_next[..., self.input_channels :]),
                3,
            )
        return x


class MedNCAEncoder(nn.Module):
    """Two-level Med-NCA (upstream ``Agent_Med_NCA.get_outputs``, inference path).

    The coarse level runs on the image downscaled by ``scale_factor``. Its state
    is upscaled (nearest) to the input size, the full-resolution image is
    re-injected, and the fine level runs. The stages are public so training
    code can recombine them; stage states are channels-last (upstream layout).

    Stochastic firing is kept at inference with the training fire rate. In eval
    mode, ``forward_feature_maps`` returns the mean state of ``n_eval_runs``
    passes (``n_eval_runs=1`` is upstream's single pass).

    ``set_grad_checkpointing(every)`` recomputes chunks of ``every`` NCA steps
    in the backward pass. It is only active in training mode with grad enabled
    and does not change the computed function (fire masks are replayed through
    the preserved RNG state).
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
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if channel_n < input_channels + output_channels:
            raise ValueError(
                f"channel_n={channel_n} cannot hold {input_channels} image and "
                f"{output_channels} output channels."
            )
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
        if n_eval_runs < 1:
            raise ValueError(f"n_eval_runs must be positive, got {n_eval_runs}.")

        self.channel_n = int(channel_n)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.steps = int(steps)
        self.scale_factor = int(scale_factor)
        self.n_eval_runs = int(n_eval_runs)

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

        self.grad_checkpointing_every: int | None = None
        self.set_grad_checkpointing(grad_checkpointing_every)

    def set_grad_checkpointing(self, every: int | None = None) -> None:
        """Checkpoint every ``every`` NCA steps in training; ``None`` disables."""
        if every is not None and every < 1:
            raise ValueError(f"Checkpointing interval must be positive, got {every}.")
        self.grad_checkpointing_every = None if every is None else int(every)

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

    def init_state(self, imgs: torch.Tensor) -> torch.Tensor:
        """Seed: zeros with the image in the first channels, ``B x H x W x C``."""
        batch, _, height, width = imgs.shape
        state = imgs.new_zeros((batch, height, width, self.channel_n))
        state[..., : self.input_channels] = imgs.permute(0, 2, 3, 1)
        return state

    def upscale_state(
        self,
        state: torch.Tensor,
        size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Nearest-neighbour upscaling of a channels-last state to ``size``."""
        upscaled = F.interpolate(
            state.permute(0, 3, 1, 2),
            size=tuple(size),
            mode="nearest",
        )
        return upscaled.permute(0, 2, 3, 1)

    def run_level(
        self,
        level: str,
        state: torch.Tensor,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """Re-inject ``imgs`` and run ``level`` for ``steps`` steps."""
        if level not in self.levels:
            raise KeyError(f"Unknown level {level!r}. Available: {list(self.levels)}.")
        if tuple(imgs.shape[-2:]) != tuple(state.shape[1:3]):
            raise ValueError(
                f"Image size {tuple(imgs.shape[-2:])} does not match state size "
                f"{tuple(state.shape[1:3])}."
            )

        state = torch.cat(
            (imgs.permute(0, 2, 3, 1), state[..., self.input_channels :]),
            3,
        )
        nca = self.levels[level]

        every = self.grad_checkpointing_every
        if every is None or not (self.training and torch.is_grad_enabled()):
            return nca(state, steps=self.steps)

        remaining = self.steps
        while remaining > 0:
            chunk = min(every, remaining)
            state = checkpoint(nca, state, chunk, use_reentrant=False)
            remaining -= chunk
        return state

    def forward_state(self, imgs: torch.Tensor) -> torch.Tensor:
        """One stochastic pass; returns the channels-last fine-level state."""
        small_imgs = self.downscale(imgs)
        state = self.run_level("coarse", self.init_state(small_imgs), small_imgs)
        state = self.upscale_state(state, imgs.shape[-2:])
        return self.run_level("fine", state, imgs)

    def forward_feature_maps(self, imgs: torch.Tensor) -> tuple[torch.Tensor]:
        state = self.forward_state(imgs)
        if not self.training and self.n_eval_runs > 1:
            for _ in range(self.n_eval_runs - 1):
                state = state + self.forward_state(imgs)
            state = state / self.n_eval_runs
        return (state.permute(0, 3, 1, 2),)


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
            input_channels=decoder.input_channels,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            upsample_logits=False,
        )
