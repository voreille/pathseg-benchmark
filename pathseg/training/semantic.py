from __future__ import annotations

from typing import Any

import torch

from pathseg.models.semantic_segmenter import SemanticSegmenter
from pathseg.training.semantic_common import SemanticLightningModule


class SemanticTraining(SemanticLightningModule):
    """Train a single- or multi-task semantic segmenter.

    Mixed-task training batches follow this format::

        (imgs, targets, task_names, image_ids)

    Every task is trained only on samples carrying its task name. A single-task
    batch may omit task names.
    """

    def __init__(
        self,
        network: SemanticSegmenter,
        tasks: dict[str, Any],
        ignore_idx: int,
        img_size: tuple[int, int],
        tiler: None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(
            network=network,
            tasks=tasks,
            ignore_idx=ignore_idx,
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            poly_lr_decay_power=poly_lr_decay_power,
            tiler=tiler,
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        imgs, targets, task_names, _image_ids = self.unpack_batch(batch)
        if not torch.is_tensor(imgs) or imgs.ndim != 4:
            raise ValueError(
                "Training images must be a BxCxHxW tensor, got "
                f"{type(imgs).__name__} with shape="
                f"{getattr(imgs, 'shape', None)}."
            )

        batch_size = int(imgs.shape[0])
        routes = self.task_routes(
            task_names,
            batch_size=batch_size,
            device=imgs.device,
        )
        logits_by_task = self.routed_forward(imgs, routes)

        weighted_losses: list[torch.Tensor] = []

        for task, route in routes.items():
            loss = self.task_loss(
                task,
                logits_by_task[task],
                targets,
                route,
            )
            weighted_losses.append(self.task_specs[task].loss_weight * loss)

            self.log(
                f"train_{task}_loss",
                loss,
                sync_dist=True,
                prog_bar=False,
                batch_size=len(route),
            )
            self.log(
                f"train_{task}_fraction",
                len(route) / batch_size,
                sync_dist=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        loss_total = torch.stack(weighted_losses).sum()
        self.log(
            "train_loss_total",
            loss_total,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss_total
