"""OctreeNCA trained with the authors' recipe.

Training follows upstream's PESO run (https://github.com/MECLabTUDA/OctreeNCA,
commit 70f7742, ``train_peso.py`` + ``Agent.py``):

- every level runs on the whole tile (``model.train.patch_sizes: [None] * 5``),
  so the training step is the inherited full-tile ``SemanticTraining`` step.
  The loss is the task's criterion from ``tasks.<task>.loss_name``; upstream's
  per-class sigmoid Dice + BCE over present classes is ``upstream_dice_bce``;
- Adam with betas (0.9, 0.99) and no weight decay, ``ExponentialLR`` with
  gamma = 0.9999 ** 8 stepped once per upstream epoch (200 steps)
  (``upstream_optimizer=True``);
- an EMA of the weights (decay 0.99, updated once per epoch, initialised from
  the initial weights), used for validation, test and the saved checkpoint
  (``ema_decay``; ``pathseg.training.callbacks.ema.UpstreamEMA``).

Validation, test and prediction are inherited unchanged: full tiles through
``network.forward`` with the tiler, exactly as for every other model.
Deviations from upstream are logged in ``experiments/octreenca/README.md``.
"""

from __future__ import annotations

from typing import Any

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathseg.models.architectures.octree_nca import OctreeNCASegmenter
from pathseg.training.callbacks.ema import UpstreamEMA
from pathseg.training.semantic import SemanticTraining


class OctreeNCATraining(SemanticTraining):
    """OctreeNCA with the authors' training recipe (see module docstring).

    New init args:
        upstream_optimizer: Adam(``lr``, ``betas``) + ExponentialLR(``lr_gamma``)
            stepped every ``lr_every_n_steps`` steps, as upstream. ``False``
            keeps the benchmark's AdamW + poly decay. ``weight_decay``,
            ``poly_lr_decay_power`` and ``lr_multiplier_encoder`` are unused
            when it is ``True``.
        betas, lr_gamma, lr_every_n_steps: upstream (0.9, 0.99), 0.9999 ** 8
            and one upstream epoch (200 steps).
        ema_decay: decay of the weight EMA used for evaluation; ``None``
            disables it. Upstream: 0.99.
        ema_every_n_steps: steps between EMA updates. Upstream: one epoch (200).
    """

    def __init__(
        self,
        network: OctreeNCASegmenter,
        tasks: dict[str, Any],
        ignore_idx: int,
        img_size: tuple[int, int],
        tiler: None,
        lr: float = 16e-4,
        weight_decay: float = 0.0,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 1.0,
        freeze_encoder: bool = False,
        upstream_optimizer: bool = True,
        betas: tuple[float, float] = (0.9, 0.99),
        lr_gamma: float = 0.9999**8,
        lr_every_n_steps: int = 200,
        ema_decay: float | None = 0.99,
        ema_every_n_steps: int = 200,
    ) -> None:
        super().__init__(
            network=network,
            tasks=tasks,
            ignore_idx=ignore_idx,
            img_size=img_size,
            tiler=tiler,
            lr=lr,
            weight_decay=weight_decay,
            poly_lr_decay_power=poly_lr_decay_power,
            lr_multiplier_encoder=lr_multiplier_encoder,
            freeze_encoder=freeze_encoder,
        )
        if not isinstance(network, OctreeNCASegmenter):
            raise TypeError(
                f"OctreeNCATraining needs an OctreeNCASegmenter, got {type(network).__name__}."
            )
        if lr_every_n_steps < 1:
            raise ValueError(f"lr_every_n_steps must be positive, got {lr_every_n_steps}.")
        if ema_decay is not None and not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}.")
        if ema_every_n_steps < 1:
            raise ValueError(f"ema_every_n_steps must be positive, got {ema_every_n_steps}.")

        self.upstream_optimizer = bool(upstream_optimizer)
        self.betas = (float(betas[0]), float(betas[1]))
        self.lr_gamma = float(lr_gamma)
        self.lr_every_n_steps = int(lr_every_n_steps)
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.ema_every_n_steps = int(ema_every_n_steps)
        self.save_hyperparameters()

    def configure_callbacks(self):
        if self.ema_decay is None:
            return []
        return [UpstreamEMA(decay=self.ema_decay, every_n_steps=self.ema_every_n_steps)]

    def configure_optimizers(self):
        if not self.upstream_optimizer:
            return super().configure_optimizers()
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("The model has no trainable parameters.")
        optimizer = Adam(parameters, lr=self.lr, betas=self.betas)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.lr_every_n_steps,
            },
        }
