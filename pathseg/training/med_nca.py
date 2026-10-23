"""Variant B: Med-NCA trained with the authors' recipe.

Training follows upstream ``Agent_Med_NCA`` / ``Agent_Multi_NCA.batch_step``
(https://github.com/MECLabTUDA/Med-NCA, commit a844a72, MIT License):

- the coarse level runs on the whole downscaled tile, its state is upscaled,
  and the fine level runs on one random crop per sample (by default the size
  of the coarse level, upstream's ``input_size[0]``);
- the loss is computed on the fine-level crop only, with the task's
  criterion from ``tasks.<task>.loss_name``. Upstream's per-class sigmoid
  Dice + BCE over present classes is ``loss_name: upstream_dice_bce``
  (``pathseg.training.histo_loss.UpstreamDiceBCELoss``). A task whose crops
  have no labelled pixel is skipped, and so is the step if no task is left,
  as upstream skips the update;
- Adam with betas (0.5, 0.5), no weight decay, and ``ExponentialLR`` stepped
  every batch (``upstream_optimizer=True``).

Validation, test and prediction are inherited unchanged: full tiles through
``network.forward`` with the tiler, exactly as for every other model. The
network is the same ``MedNCASegmenter`` as in Variant A; only training differs.
Deviations from upstream are logged in ``experiments/mednca/README.md``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathseg.models.architectures.med_nca import MedNCASegmenter
from pathseg.training.semantic import SemanticTraining


class MedNCATraining(SemanticTraining):
    """Med-NCA with the authors' training recipe (see module docstring).

    New init args:
        crop_size: fine-level training crop (pixels, square). ``None`` uses
            the coarse level's size, ``img_size // scale_factor``, as upstream.
        upstream_optimizer: Adam(``lr``, ``betas``) + ExponentialLR(``lr_gamma``)
            stepped every batch, as upstream. ``False`` keeps the benchmark's
            AdamW + poly decay. ``weight_decay``, ``poly_lr_decay_power`` and
            ``lr_multiplier_encoder`` are unused when it is ``True``.
        betas, lr_gamma: upstream defaults (0.5, 0.5) and 0.9999.
    """

    def __init__(
        self,
        network: MedNCASegmenter,
        tasks: dict[str, Any],
        ignore_idx: int,
        img_size: tuple[int, int],
        tiler: None,
        lr: float = 16e-4,
        weight_decay: float = 0.0,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 1.0,
        freeze_encoder: bool = False,
        crop_size: int | None = None,
        upstream_optimizer: bool = True,
        betas: tuple[float, float] = (0.5, 0.5),
        lr_gamma: float = 0.9999,
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
        if not isinstance(network, MedNCASegmenter):
            raise TypeError(
                f"MedNCATraining needs a MedNCASegmenter, got {type(network).__name__}."
            )
        if crop_size is not None and crop_size < 1:
            raise ValueError(f"crop_size must be positive, got {crop_size}.")

        self.crop_size = None if crop_size is None else int(crop_size)
        self.upstream_optimizer = bool(upstream_optimizer)
        self.betas = (float(betas[0]), float(betas[1]))
        self.lr_gamma = float(lr_gamma)
        self.save_hyperparameters()

    def fine_crop_size(self, height: int, width: int) -> tuple[int, int]:
        if self.crop_size is not None:
            size = (self.crop_size, self.crop_size)
        else:
            scale = self.network.encoder.scale_factor
            size = (height // scale, width // scale)
        if size[0] > height or size[1] > width:
            raise ValueError(f"Crop {size} is larger than the tile {(height, width)}.")
        return size

    @staticmethod
    def random_crop(
        state: torch.Tensor,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One random crop per sample, at the same position in all three.

        ``state``: ``B x H x W x C`` (channels-last); ``imgs``: ``B x 3 x H x W``;
        ``targets``: ``B x H x W``. Upstream draws the corner with Python's
        ``random.randint``; here it comes from the torch CPU generator.
        """
        batch, height, width = targets.shape
        rows = torch.randint(0, height - size[0] + 1, (batch,)).tolist()
        cols = torch.randint(0, width - size[1] + 1, (batch,)).tolist()
        windows = [
            (slice(row, row + size[0]), slice(col, col + size[1]))
            for row, col in zip(rows, cols)
        ]
        return (
            torch.stack([state[b, r, c] for b, (r, c) in enumerate(windows)]),
            torch.stack([imgs[b, :, r, c] for b, (r, c) in enumerate(windows)]),
            torch.stack([targets[b, r, c] for b, (r, c) in enumerate(windows)]),
        )

    def training_step(self, batch, batch_idx):
        imgs, targets, task_names, _image_ids = self.unpack_batch(batch)
        if not torch.is_tensor(imgs) or imgs.ndim != 4:
            raise ValueError(
                "Training images must be a BxCxHxW tensor, got "
                f"{type(imgs).__name__} with shape={getattr(imgs, 'shape', None)}."
            )

        batch_size = int(imgs.shape[0])
        routes = self.task_routes(task_names, batch_size=batch_size, device=imgs.device)
        target_maps = torch.stack(
            self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        )
        target_maps = target_maps.long().to(imgs.device)

        # Upstream Agent_Med_NCA.get_outputs (training path), built from the
        # encoder stages that inference composes in forward_feature_maps.
        encoder = self.network.encoder
        x = imgs / 255.0
        small = encoder.downscale(x)
        state = encoder.run_level("coarse", encoder.init_state(small), small)
        state = encoder.upscale_state(state, x.shape[-2:])
        size = self.fine_crop_size(*x.shape[-2:])
        state, x_crop, target_crop = self.random_crop(state, x, target_maps, size)
        state = encoder.run_level("fine", state, x_crop)
        feature_maps = (state.permute(0, 3, 1, 2),)

        losses: list[torch.Tensor] = []
        for task, route in routes.items():
            task_maps = tuple(f.index_select(0, route.device_indices) for f in feature_maps)
            task_targets = target_crop.index_select(0, route.device_indices)
            if not bool((task_targets != self.ignore_idx).any()):
                # Upstream skips the update when no pixel is labelled.
                continue
            logits = self.network.decode(task_maps, task=task)[task]
            loss = self.criteria[task](logits.float(), task_targets)
            losses.append(self.task_specs[task].loss_weight * loss)
            self.log(f"train_{task}_loss", loss, sync_dist=True, batch_size=len(route))
            self.log(
                f"train_{task}_fraction",
                len(route) / batch_size,
                sync_dist=True,
                batch_size=batch_size,
            )

        if not losses:
            return None
        loss_total = torch.stack(losses).sum()
        self.log(
            "train_loss_total",
            loss_total,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss_total

    def configure_optimizers(self):
        if not self.upstream_optimizer:
            return super().configure_optimizers()
        # Upstream has one Adam + ExponentialLR per level; Adam is per-parameter
        # and there is no weight decay, so one optimizer over both is identical.
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("The model has no trainable parameters.")
        optimizer = Adam(parameters, lr=self.lr, betas=self.betas)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
