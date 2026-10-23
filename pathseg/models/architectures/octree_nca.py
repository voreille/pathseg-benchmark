"""OctreeNCA (Lemke et al., BMVC 2025) as a pathseg semantic segmenter.

Port of the backbone NCA (``BasicNCA2DFast``) and the multi-level forward
(``OctreeNCA2DPatch2``) of https://github.com/MECLabTUDA/OctreeNCA (commit
70f7742). Only the model is ported; the upstream training loop, datasets and
losses are not used, and neither is the CUDA inference kernel (same outputs,
speed/VRAM only). Deviations from upstream are logged in
``experiments/octreenca/README.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pathseg.models.architectures.med_nca import StateSliceDecoder
from pathseg.models.nca import FoldedNCA, MultiLevelNCAEncoder
from pathseg.models.semantic_segmenter import SemanticSegmenter

# Upstream PESO run (train_peso.py): 10 steps per level, alpha * 20 at the
# 20 px coarsest level, finest level first.
UPSTREAM_STEPS = (10, 10, 10, 10, 20)


class OctreeBackboneNCA(FoldedNCA):
    """One OctreeNCA level: upstream ``BasicNCA2DFast`` with ``normalization="none"``.

    Perception is ``cat(x, conv(x))`` with a depthwise 3x3 conv (reflect
    padding), then ``fc0`` (1x1) -> ReLU -> ``fc1`` (1x1, no bias, zero init),
    which writes only the non-image channels. The state is channels-last
    ``B x H x W x channel_n`` with layout ``[image | outputs | hidden]``; the
    image channels are never updated.

    Only ``normalization="none"`` (the PESO run) is ported: with a
    normalization layer the update no longer folds into one conv.
    """

    def __init__(
        self,
        channel_n: int,
        fire_rate: float = 0.5,
        hidden_size: int = 64,
        input_channels: int = 3,
    ) -> None:
        super().__init__(channel_n, fire_rate=fire_rate, input_channels=input_channels)

        # Same parameter names, shapes and creation order as upstream, so state
        # dicts are interchangeable and seeded initialization matches.
        self.fc0 = nn.Conv2d(self.channel_n * 2, hidden_size, kernel_size=1)
        self.fc1 = nn.Conv2d(
            hidden_size, self.channel_n - self.input_channels, kernel_size=1, bias=False
        )
        self.conv = nn.Conv2d(
            self.channel_n,
            self.channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            groups=self.channel_n,
        )
        with torch.no_grad():
            self.fc1.weight.zero_()

    def folded_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The update rule as ``(kernel, bias, fc1_weight)``, built from the upstream parameters.

        ``fc0(cat(x, conv(x)))`` is linear in ``x``, so it equals one 3x3 conv
        with reflect padding: kernel ``fc0_conv[h, c] * conv[c] (+ fc0_x at the
        centre tap)`` and bias ``fc0.bias + fc0_conv @ conv.bias``. Upstream
        runs on the untransposed grid, so the kernel is not transposed.

        ``fc1`` gets zero rows for the image channels, so ``dx`` is exactly 0
        there (upstream concatenates the image back after every step).

        Built in fp32 (outside autocast) once per ``forward``; gradients flow to
        the original parameters, so the state_dict is unchanged.
        """
        with torch.autocast(self.fc0.weight.device.type, enabled=False):
            w_x, w_conv = self.fc0.weight[:, :, 0, 0].split(self.channel_n, dim=1)
            kernel = w_conv[:, :, None, None] * self.conv.weight[:, 0]
            kernel = kernel + F.pad(w_x[..., None, None], (1, 1, 1, 1))
            bias = self.fc0.bias + w_conv @ self.conv.bias
            fc1_weight = F.pad(self.fc1.weight[:, :, 0, 0], (0, 0, self.input_channels, 0))
        return kernel, bias, fc1_weight

    def fire_mask(self, x: torch.Tensor, fire_rate: float) -> torch.Tensor:
        """A cell updates with probability ``fire_rate`` (upstream's draw).

        Same call as upstream (``bernoulli_`` on a ``B x 1 x H x W`` tensor on
        the state's device), so the masks match upstream under the same seed.
        """
        batch, height, width, _ = x.shape
        mask = torch.zeros([batch, 1, height, width], device=x.device)
        return mask.bernoulli_(p=fire_rate).permute(0, 2, 3, 1)


class OctreeNCAEncoder(MultiLevelNCAEncoder):
    """Multi-level OctreeNCA (upstream ``OctreeNCA2DPatch2``, inference path).

    Level 0 is the input resolution; level ``l`` has size ``ceil(size_{l-1} / 2)``
    (upstream's rule for inputs of any size). ``steps[l]`` is level ``l``'s step
    count, finest first as in upstream's ``res_and_steps``. Each level's image
    is the input downscaled with nearest interpolation (upstream's default
    ``F.interpolate`` mode). The coarsest level starts from the seed; the state
    is then upscaled (nearest) level by level, with the image re-injected before
    each level runs. The stages are public so training code can recombine
    them; stage states are channels-last. Eval-time averaging (``n_eval_runs``),
    gradient checkpointing, ``max_batch_size`` and ``compile_step``: see
    ``MultiLevelNCAEncoder``.
    """

    def __init__(
        self,
        output_channels: int,
        *,
        channel_n: int = 16,
        hidden_size: int = 64,
        steps: Sequence[int] = UPSTREAM_STEPS,
        fire_rate: float = 0.5,
        n_eval_runs: int = 1,
        grad_checkpointing_every: int | None = None,
        max_batch_size: int | None = None,
        compile_step: bool = False,
        input_channels: int = 3,
    ) -> None:
        steps = tuple(int(s) for s in steps)
        if not steps:
            raise ValueError("steps must list at least one level.")
        if any(s < 1 for s in steps):
            raise ValueError(f"steps must be positive, got {list(steps)}.")
        super().__init__(
            output_channels,
            channel_n=channel_n,
            n_eval_runs=n_eval_runs,
            grad_checkpointing_every=grad_checkpointing_every,
            max_batch_size=max_batch_size,
            input_channels=input_channels,
        )
        self.steps = steps

        self.backbone_ncas = nn.ModuleList(
            OctreeBackboneNCA(
                self.channel_n,
                fire_rate=fire_rate,
                hidden_size=hidden_size,
                input_channels=self.input_channels,
            )
            for _ in steps
        )
        self.set_compile_step(compile_step)

    @property
    def num_levels(self) -> int:
        return len(self.steps)

    def level_modules(self):
        return iter(self.backbone_ncas)

    def level_module(self, level: int) -> OctreeBackboneNCA:
        if not 0 <= level < self.num_levels:
            raise KeyError(
                f"Unknown level {level!r}. Available: {list(range(self.num_levels))}."
            )
        return self.backbone_ncas[level]

    def level_steps(self, level: int) -> int:
        return self.steps[level]

    def level_sizes(self, height: int, width: int) -> list[tuple[int, int]]:
        """``(H, W)`` per level, finest first: ``ceil`` halving per level."""
        sizes = [(int(height), int(width))]
        for _ in range(self.num_levels - 1):
            prev_h, prev_w = sizes[-1]
            sizes.append((-(-prev_h // 2), -(-prev_w // 2)))
        return sizes

    def pyramid(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        """The image at every level (finest first), nearest-downscaled from ``imgs``."""
        sizes = self.level_sizes(*imgs.shape[-2:])
        return [imgs] + [F.interpolate(imgs, size=size, mode="nearest") for size in sizes[1:]]

    def forward_state(self, imgs: torch.Tensor) -> torch.Tensor:
        """One stochastic pass; returns the channels-last level-0 state."""
        images = self.pyramid(imgs)
        coarsest = self.num_levels - 1
        state = self.init_state(images[coarsest])
        for level in range(coarsest, -1, -1):
            if level < coarsest:
                state = self.upscale_state(state, images[level].shape[-2:])
            state = self.run_level(level, state, images[level])
        return state


class OctreeNCASegmenter(SemanticSegmenter):
    """OctreeNCA whose output state channels are the semantic logits."""

    def __init__(
        self,
        num_classes_by_task: dict[str, int],
        *,
        channel_n: int = 16,
        hidden_size: int = 64,
        steps: Sequence[int] = UPSTREAM_STEPS,
        fire_rate: float = 0.5,
        n_eval_runs: int = 1,
        grad_checkpointing_every: int | None = None,
        max_batch_size: int | None = None,
        compile_step: bool = False,
    ) -> None:
        decoder = StateSliceDecoder(num_classes_by_task, input_channels=3)

        encoder = OctreeNCAEncoder(
            output_channels=decoder.output_channels,
            channel_n=channel_n,
            hidden_size=hidden_size,
            steps=steps,
            fire_rate=fire_rate,
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
