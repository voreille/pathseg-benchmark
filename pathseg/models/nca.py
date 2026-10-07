"""Shared building blocks for multi-level NCA segmenters (Med-NCA, OctreeNCA).

An NCA level whose update is ``fc1(relu(perceive_linear(x)))`` with a linear
3x3 perception folds into one 3x3 conv (kernel + bias) followed by ``fc1``.
``FoldedNCA`` runs that folded update on a channels-last ``B x H x W x C``
state with a hand-fused eager step (default) or a ``torch.compile``d plain-ops
step (``compile_step``, CUDA training only). Subclasses define how their
upstream parameters fold (``folded_weights``) and how the fire mask is drawn
(``fire_mask``).

``MultiLevelNCAEncoder`` holds what every multi-level NCA encoder shares: the
seed, nearest upscaling of the state, running one level with image
re-injection and optional gradient checkpointing, eval-time averaging over
``n_eval_runs`` stochastic passes, and the ``max_batch_size`` memory cap.
Subclasses define the levels and the inference composition (``forward_state``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype autocast would run convs/linears in, else ``x.dtype``."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


class _ReflectPadCastHW(torch.autograd.Function):
    """Reflect-pad H and W of a channels-last ``B x H x W x C`` by 1 and cast.

    Same values as ``F.pad(..., mode="reflect")`` on the NCHW view followed by
    ``.to(dtype)``, but the output stays NHWC (``F.pad`` returns NCHW memory)
    and the pad and cast are one full-tensor copy forward and one backward,
    plus thin border strips.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ctx.input_dtype = x.dtype
        batch, height, width, channels = x.shape
        out = x.new_empty((batch, height + 2, width + 2, channels), dtype=dtype)
        out[:, 1:-1, 1:-1] = x
        out[:, 0, 1:-1] = out[:, 2, 1:-1]
        out[:, -1, 1:-1] = out[:, -3, 1:-1]
        out[:, :, 0] = out[:, :, 2]
        out[:, :, -1] = out[:, :, -3]
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Fold the border onto the cells it was copied from: columns first
        # (corners included), then rows.
        grad_x = grad[:, 1:-1, 1:-1].to(ctx.input_dtype, copy=True)
        grad_x[:, :, 1] += grad[:, 1:-1, 0]
        grad_x[:, :, -2] += grad[:, 1:-1, -1]
        rows = grad[:, (0, -1)].to(ctx.input_dtype)
        rows[:, :, 2] += rows[:, :, 0]
        rows[:, :, -3] += rows[:, :, -1]
        grad_x[:, 1] += rows[:, 0, 1:-1]
        grad_x[:, -2] += rows[:, 1, 1:-1]
        return grad_x, None


def _reflect_pad_cast_hw(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _ReflectPadCastHW.apply(x, dtype)


class _CudnnConvBiasReLU(torch.autograd.Function):
    """``relu(conv2d(x, weight, bias))`` (3x3, no padding) as one cuDNN kernel.

    Eager PyTorch adds the conv bias and applies the ReLU as two extra passes
    over the hidden map. ``torch.cudnn_convolution_relu`` fuses them but has
    no autograd, so backward is spelled out with the ops autograd would use.
    The only difference from eager is one rounding instead of two (the bias is
    added before the output is cast).
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        out = torch.cudnn_convolution_relu(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
        ctx.save_for_backward(x, weight, out)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, weight, out = ctx.saved_tensors
        grad = torch.ops.aten.threshold_backward(grad, out, 0)
        return torch.ops.aten.convolution_backward(
            grad,
            x,
            weight,
            [weight.shape[0]],
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            (0, 0),
            1,
            ctx.needs_input_grad,
        )


def _conv_bias_relu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if x.is_cuda and torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled:
        return _CudnnConvBiasReLU.apply(x, weight.to(x.dtype), bias.to(x.dtype))
    return F.relu(F.conv2d(x, weight, bias))


def _nca_step(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    stochastic: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic part of one NCA step on a channels-last state."""
    # The conv casts to the autocast dtype anyway; casting while padding
    # gives the same values with less traffic.
    padded = _reflect_pad_cast_hw(x, compute_dtype)
    hidden = _conv_bias_relu(padded.permute(0, 3, 1, 2), kernel, bias)
    dx = F.linear(hidden.permute(0, 2, 3, 1), fc1_weight)
    # Masking in dx's dtype is exact (the mask is 0/1) and halves the traffic.
    return x + dx * stochastic.to(dx.dtype)


def _nca_step_plain(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    stochastic: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """``_nca_step`` in plain ops, written for ``torch.compile``.

    Same function as ``_nca_step``. The hand-fused autograd Functions there
    are opaque to Inductor; in plain ops it fuses pad + cast, bias + ReLU and
    mask + residual (forward and backward) itself, which is faster.
    """
    x_cast = x.to(compute_dtype)
    x_cast = torch.cat((x_cast[:, 1:2], x_cast, x_cast[:, -2:-1]), 1)
    padded = torch.cat((x_cast[:, :, 1:2], x_cast, x_cast[:, :, -2:-1]), 2)
    hidden = F.relu(F.conv2d(padded.permute(0, 3, 1, 2), kernel, bias))
    dx = F.linear(hidden.permute(0, 2, 3, 1), fc1_weight)
    return x + dx * stochastic.to(dx.dtype)


_compiled_nca_step = None


def _get_compiled_nca_step():
    """``_nca_step_plain`` compiled once per process, on first use.

    Same options as ``mode="max-autotune-no-cudagraphs"``, minus the autotune
    reports.

    ``dynamic=False``: with dynamic shapes, the checkpoint recomputation
    recompiled the step and failed the checkpoint metadata check. Training
    has one fixed shape per level, so this costs one graph per level.
    """
    global _compiled_nca_step
    if _compiled_nca_step is None:
        options = torch._inductor.list_mode_options("max-autotune-no-cudagraphs")
        # max-autotune prints a benchmark table per tuned conv/GEMM to stderr.
        options.update(
            autotune_num_choices_displayed=0,
            max_autotune_report_choices_stats=False,
        )
        _compiled_nca_step = torch.compile(
            _nca_step_plain, dynamic=False, options=options
        )
    return _compiled_nca_step


class FoldedNCA(nn.Module):
    """One NCA level whose update folds into ``fc1(relu(conv3x3(x)))``.

    The state is channels-last ``B x H x W x channel_n`` with layout
    ``[image | outputs | hidden]``. ``folded_weights`` must zero ``fc1``'s rows
    for the first ``input_channels`` channels, so ``dx`` is exactly 0 there:
    the image channels are never updated (upstream's re-injection).

    Subclasses implement:
        folded_weights() -> (kernel, bias, fc1_weight): the 3x3 conv
            ``hidden x channel_n x 3 x 3`` (applied on the NHWC view, reflect
            padding), its bias, and ``fc1`` as ``channel_n x hidden``.
        fire_mask(x, fire_rate) -> ``B x H x W x 1`` 0/1 mask of updating cells.
    """

    def __init__(
        self,
        channel_n: int,
        fire_rate: float = 0.5,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.channel_n = int(channel_n)
        self.input_channels = int(input_channels)
        self.fire_rate = float(fire_rate)
        # Not a parameter or buffer: the state_dict is unchanged.
        self.compile_step = False

    def folded_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fire_mask(self, x: torch.Tensor, fire_rate: float) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        x: torch.Tensor,
        fire_rate: float | None = None,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One step on a channels-last state. The image channels are left as is."""
        if fire_rate is None:
            fire_rate = self.fire_rate
        if weights is None:
            weights = self.folded_weights()

        # Drawn eagerly, outside any compiled region, so masks and checkpoint
        # replay are the same with and without compile_step.
        stochastic = self.fire_mask(x, fire_rate)
        step = _nca_step
        if self.compile_step and self.training and torch.is_grad_enabled() and x.is_cuda:
            step = _get_compiled_nca_step()
        return step(x, *weights, stochastic, _compute_dtype(x))

    def forward(
        self,
        x: torch.Tensor,
        steps: int,
        fire_rate: float | None = None,
    ) -> torch.Tensor:
        # update() leaves the image channels untouched (dx is exactly 0 there),
        # which is upstream's re-injection of x[..., :input_channels].
        weights = self.folded_weights()
        for _ in range(steps):
            x = self.update(x, fire_rate, weights)
        return x


class MultiLevelNCAEncoder(nn.Module):
    """Base encoder for multi-level NCAs; stage states are channels-last.

    Subclasses build their ``FoldedNCA`` levels in ``__init__`` (then call
    ``set_compile_step``) and implement ``level_modules``, ``level_module``,
    ``level_steps`` and ``forward_state`` (one stochastic pass, returning the
    finest level's channels-last state).

    Stochastic firing is kept at inference with the training fire rate. In
    eval mode, ``forward_feature_maps`` returns the mean state of
    ``n_eval_runs`` passes (``n_eval_runs=1`` is upstream's single pass).

    ``set_grad_checkpointing(every)`` recomputes chunks of ``every`` NCA steps
    in the backward pass. It is only active in training mode with grad enabled
    and does not change the computed function (fire masks are replayed through
    the preserved RNG state).

    ``max_batch_size`` caps how many images one forward pass processes; larger
    inputs are split into chunks and concatenated. The NCA has no cross-sample
    ops, so this does not change the computed function (only the order in which
    fire masks are drawn). It bounds memory when a caller passes many tiles at
    once (e.g. all tiles of a large ROI at validation). With grad enabled it
    saves little, since every chunk's activations are kept for backward.

    ``set_compile_step(True)`` runs each NCA step through a ``torch.compile``d
    plain-ops version (``max-autotune-no-cudagraphs``, static shapes; about 1 min
    of compilation on first use). It is only active in training mode with grad
    enabled on CUDA, where shapes are fixed. Eval and inference keep the eager
    step, since validation's varying chunk sizes would each recompile. The fire
    mask is drawn eagerly outside the compiled region, so masks and checkpoint
    replay are unchanged. Same function to float round-off; default off.
    """

    def __init__(
        self,
        output_channels: int,
        *,
        channel_n: int,
        n_eval_runs: int = 1,
        grad_checkpointing_every: int | None = None,
        max_batch_size: int | None = None,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if channel_n < input_channels + output_channels:
            raise ValueError(
                f"channel_n={channel_n} cannot hold {input_channels} image and "
                f"{output_channels} output channels."
            )
        if n_eval_runs < 1:
            raise ValueError(f"n_eval_runs must be positive, got {n_eval_runs}.")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be positive, got {max_batch_size}."
            )

        self.channel_n = int(channel_n)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.n_eval_runs = int(n_eval_runs)
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)

        self.grad_checkpointing_every: int | None = None
        self.set_grad_checkpointing(grad_checkpointing_every)

    def level_modules(self) -> Iterable[FoldedNCA]:
        raise NotImplementedError

    def level_module(self, level) -> FoldedNCA:
        raise NotImplementedError

    def level_steps(self, level) -> int:
        raise NotImplementedError

    def forward_state(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_grad_checkpointing(self, every: int | None = None) -> None:
        """Checkpoint every ``every`` NCA steps in training; ``None`` disables."""
        if every is not None and every < 1:
            raise ValueError(f"Checkpointing interval must be positive, got {every}.")
        self.grad_checkpointing_every = None if every is None else int(every)

    def set_compile_step(self, enabled: bool = True) -> None:
        """Use the compiled NCA step in CUDA training (see class docstring)."""
        for level in self.level_modules():
            level.compile_step = bool(enabled)

    @property
    def compile_step(self) -> bool:
        return all(level.compile_step for level in self.level_modules())

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
        level,
        state: torch.Tensor,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """Re-inject ``imgs`` and run ``level`` for its configured steps."""
        nca = self.level_module(level)
        if tuple(imgs.shape[-2:]) != tuple(state.shape[1:3]):
            raise ValueError(
                f"Image size {tuple(imgs.shape[-2:])} does not match state size "
                f"{tuple(state.shape[1:3])}."
            )

        state = torch.cat(
            (imgs.permute(0, 2, 3, 1), state[..., self.input_channels :]),
            3,
        )
        steps = self.level_steps(level)

        every = self.grad_checkpointing_every
        if every is None or not (self.training and torch.is_grad_enabled()):
            return nca(state, steps=steps)

        remaining = steps
        while remaining > 0:
            chunk = min(every, remaining)
            state = checkpoint(nca, state, chunk, use_reentrant=False)
            remaining -= chunk
        return state

    def forward_feature_maps(self, imgs: torch.Tensor) -> tuple[torch.Tensor]:
        if self.max_batch_size is None or imgs.shape[0] <= self.max_batch_size:
            return (self._mean_state(imgs).permute(0, 3, 1, 2),)
        states = [
            self._mean_state(chunk) for chunk in imgs.split(self.max_batch_size)
        ]
        return (torch.cat(states).permute(0, 3, 1, 2),)

    def _mean_state(self, imgs: torch.Tensor) -> torch.Tensor:
        """Channels-last state; the mean of ``n_eval_runs`` passes in eval mode."""
        state = self.forward_state(imgs)
        if not self.training and self.n_eval_runs > 1:
            for _ in range(self.n_eval_runs - 1):
                state = state + self.forward_state(imgs)
            state = state / self.n_eval_runs
        return state
