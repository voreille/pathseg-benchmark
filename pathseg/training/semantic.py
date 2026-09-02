from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pathseg.models.builders import build_tiler
from pathseg.models.builders_semantic import build_semantic_segmenter
from pathseg.training.semantic_common import (
    SemanticLightningModule,
    parse_task_specs,
)


class SemanticTraining(SemanticLightningModule):
    """Train a single- or multi-task semantic segmenter.

    Mixed-task training batches follow this format::

        (imgs, targets, task_names, image_ids)

    Every task is trained only on samples carrying its task name. A single-task
    batch may omit task names.
    """

    def __init__(
        self,
        encoder_class_path: str,
        decoder_name: str,
        tasks: list[dict[str, Any]],
        ignore_idx: int,
        img_size: tuple[int, int],
        encoder_init_args: dict[str, Any] | None = None,
        decoder_init_args: dict[str, Any] | None = None,
        tiler_name: str | None = None,
        tiler_init_args: dict[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        upsample_logits: bool = False,
        interpolation_mode: str = "bilinear",
    ) -> None:
        task_specs = parse_task_specs(tasks)
        network = build_semantic_segmenter(
            encoder_class_path=encoder_class_path,
            encoder_init_args=encoder_init_args,
            decoder_name=decoder_name,
            decoder_init_args=decoder_init_args,
            num_classes_by_task={
                name: spec.num_classes for name, spec in task_specs.items()
            },
            upsample_logits=upsample_logits,
            interpolation_mode=interpolation_mode,
        )
        tiler = build_tiler(
            tiler_name=tiler_name,
            tiler_init_args=tiler_init_args,
        )

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
