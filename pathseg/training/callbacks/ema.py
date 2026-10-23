"""EMA of the weights with the OctreeNCA authors' update rule.

Upstream (https://github.com/MECLabTUDA/OctreeNCA, commit 70f7742,
``src/utils/EMA.py`` + ``Agent.train``) keeps a shadow copy of the trainable
parameters, initialised from the initial weights, and after every epoch sets
``shadow = decay * shadow + (1 - decay) * param``. Evaluation and the saved
model use the shadow weights.

``UpstreamEMA`` gets the swap for validation, the EMA weights in the
checkpoint's ``state_dict`` (so ``validate``/``test --ckpt_path`` evaluate the
EMA model) and the copy into the model at the end of training from Lightning's
``WeightAveraging``. It adds upstream's timing and initialisation:

- updates after every ``every_n_steps`` optimizer steps (upstream: once per
  epoch of 200 steps), not after the first step;
- the first update blends with the initial weights, where Lightning's
  ``AveragedModel`` would copy the current weights;
- parameters only (``use_buffers=False``), as upstream.
"""

from __future__ import annotations

from typing import Any

from lightning.pytorch.callbacks.weight_averaging import WeightAveraging
from torch.optim.swa_utils import get_ema_multi_avg_fn


class UpstreamEMA(WeightAveraging):
    def __init__(self, decay: float = 0.99, every_n_steps: int = 200, **kwargs: Any):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}.")
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be positive, got {every_n_steps}.")
        kwargs.setdefault("use_buffers", False)
        super().__init__(multi_avg_fn=get_ema_multi_avg_fn(decay), **kwargs)
        self.decay = float(decay)
        self.every_n_steps = int(every_n_steps)

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        # step_idx is zero-based: update after steps every_n_steps, 2 * every_n_steps, ...
        return step_idx is not None and (step_idx + 1) % self.every_n_steps == 0

    def setup(self, trainer, pl_module, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self._average_model is not None:
            # AveragedModel copies the weights on its first update (n_averaged == 0).
            # Upstream's shadow already holds the initial weights, so the first
            # update is an EMA step from them. A resumed run restores n_averaged
            # from the checkpoint.
            self._average_model.n_averaged.fill_(1)
