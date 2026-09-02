from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from pathseg.training.histo_loss import CrossEntropyDiceLoss
from pathseg.training.lightning_module import LightningModule


@dataclass(frozen=True)
class SemanticTaskSpec:
    name: str
    num_classes: int
    source_id: int | None
    loss_weight: float
    loss_name: str
    class_weights: tuple[float, ...] | None


def parse_task_specs(
    tasks: dict[str, dict[str, Any]],
) -> dict[str, SemanticTaskSpec]:
    if not tasks:
        raise ValueError("At least one semantic task must be configured.")

    parsed: dict[str, SemanticTaskSpec] = {}
    source_ids: dict[int, str] = {}

    for name, config in tasks.items():
        source_id = config.get("source_id")
        source_id = None if source_id is None else int(source_id)

        if source_id is not None and source_id in source_ids:
            raise ValueError(
                f"Tasks {source_ids[source_id]!r} and {name!r} both use "
                f"source_id={source_id}."
            )
        if source_id is not None:
            source_ids[source_id] = name

        class_weights = config.get("class_weights")
        if class_weights is not None:
            class_weights = tuple(float(weight) for weight in class_weights)

        parsed[name] = SemanticTaskSpec(
            name=name,
            num_classes=int(config["num_classes"]),
            source_id=source_id,
            loss_weight=float(config.get("loss_weight", 1.0)),
            loss_name=str(config.get("loss_name", "cross_entropy")),
            class_weights=class_weights,
        )

    if len(parsed) > 1 and any(spec.source_id is None for spec in parsed.values()):
        raise ValueError("Every task needs a source_id when multiple tasks are used.")

    return parsed


def build_criterion(
    spec: SemanticTaskSpec,
    ignore_idx: int,
) -> nn.Module:
    weight = (
        torch.tensor(spec.class_weights, dtype=torch.float32)
        if spec.class_weights is not None
        else None
    )

    if spec.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_idx, weight=weight)
    if spec.loss_name == "cross_entropy_dice":
        return CrossEntropyDiceLoss(ignore_index=ignore_idx, weight=weight)

    raise ValueError(
        f"Unknown loss {spec.loss_name!r} for task {spec.name!r}."
    )


class SemanticLightningModule(LightningModule):
    """Shared multi-task semantic evaluation and optimization infrastructure.

    `eval_task_names[dataloader_idx]` identifies the head evaluated by each
    validation/test dataloader. Training batches may mix tasks and use source_ids
    to select the supervised subset for every head.
    """

    def __init__(
        self,
        *,
        network: nn.Module,
        tasks: dict[str, dict[str, Any]],
        eval_task_names: Sequence[str],
        ignore_idx: int,
        img_size: tuple[int, int],
        freeze_encoder: bool,
        weight_decay: float,
        lr: float,
        lr_multiplier_encoder: float,
        poly_lr_decay_power: float,
        tiler=None,
    ) -> None:
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            tiler=tiler,
        )

        self.task_specs = parse_task_specs(tasks)
        self.eval_task_names = tuple(eval_task_names)
        if not self.eval_task_names:
            raise ValueError("eval_task_names cannot be empty.")
        unknown_eval_tasks = set(self.eval_task_names) - set(self.task_specs)
        if unknown_eval_tasks:
            raise ValueError(f"Unknown evaluation tasks: {unknown_eval_tasks}.")

        self.ignore_idx = int(ignore_idx)
        self.poly_lr_decay_power = float(poly_lr_decay_power)

        self.criteria = nn.ModuleDict(
            {
                name: build_criterion(spec, self.ignore_idx)
                for name, spec in self.task_specs.items()
            }
        )

        self.iou_metrics, self.f1_metrics = self._make_metric_streams()

    @property
    def num_classes_by_task(self) -> dict[str, int]:
        return {
            name: spec.num_classes
            for name, spec in self.task_specs.items()
        }

    def forward(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # Images arrive as uint8-like 0..255 tensors in the existing pipeline;
        # the encoder wrapper performs its own mean/std normalization.
        return self.network(imgs / 255.0, task=task)

    def _make_metric_streams(self) -> tuple[nn.ModuleList, nn.ModuleList]:
        iou = nn.ModuleList()
        f1 = nn.ModuleList()

        for task in self.eval_task_names:
            num_classes = self.task_specs[task].num_classes
            iou.append(
                MulticlassJaccardIndex(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=self.ignore_idx,
                    average=None,
                )
            )
            f1.append(
                MulticlassF1Score(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=self.ignore_idx,
                    average=None,
                )
            )

        return iou, f1

    @staticmethod
    def unpack_batch(batch):
        if len(batch) == 2:
            imgs, targets = batch
            return imgs, targets, None, None
        if len(batch) == 3:
            imgs, targets, source_ids = batch
            return imgs, targets, source_ids, None
        if len(batch) == 4:
            return batch
        raise ValueError(
            "Expected (imgs, targets), (imgs, targets, source_ids), or "
            "(imgs, targets, source_ids, image_ids)."
        )

    @staticmethod
    def _select_batch(value, mask: torch.Tensor):
        if torch.is_tensor(value):
            return value[mask]
        return [item for item, keep in zip(value, mask.tolist()) if keep]

    def _targets_to_per_pixel(self, targets) -> list[torch.Tensor]:
        if torch.is_tensor(targets):
            if targets.ndim != 3:
                raise ValueError(
                    f"Per-pixel targets must be BxHxW, got {tuple(targets.shape)}."
                )
            return [target.long() for target in targets]

        return self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

    def task_mask(
        self,
        task: str,
        source_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        source_id = self.task_specs[task].source_id
        if source_id is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if source_ids is None:
            raise ValueError(
                f"Task {task!r} has source_id={source_id}, but the batch has "
                "no source_ids."
            )
        return source_ids.to(device) == source_id

    def task_loss(
        self,
        task: str,
        logits: torch.Tensor,
        targets,
        subset_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the absent head connected to the graph for DDP without changing
        # the loss value.
        if not subset_mask.any():
            return logits.sum() * 0.0

        logits_subset = logits[subset_mask]
        targets_subset = self._select_batch(targets, subset_mask)
        target_maps = self._targets_to_per_pixel(targets_subset)
        target_tensor = torch.stack(target_maps).long().to(logits.device)

        return self.criteria[task](logits_subset, target_tensor)

    def stitch_crop_logits(
        self,
        crop_logits: torch.Tensor,
        origins,
        img_sizes,
    ) -> list[torch.Tensor]:
        return self.revert_window_logits_semantic(
            crop_logits,
            origins,
            img_sizes,
        )

    def stitch_logits_by_task(
        self,
        crop_logits_by_task: Mapping[str, torch.Tensor],
        origins,
        img_sizes,
    ) -> dict[str, list[torch.Tensor]]:
        if not crop_logits_by_task:
            raise ValueError("No semantic logits were returned by the decoder.")

        return {
            task: self.stitch_crop_logits(crop_logits, origins, img_sizes)
            for task, crop_logits in crop_logits_by_task.items()
        }

    @torch.compiler.disable
    def plot_semantic_heads(
        self,
        img: torch.Tensor,
        target: torch.Tensor,
        logits_by_task: Mapping[str, torch.Tensor],
        *,
        target_task: str,
        cmap: str = "tab20",
        max_columns: int = 4,
    ) -> Image.Image:
        """Plot one image, its current-task GT, and every task-head prediction."""
        expected_tasks = tuple(self.task_specs)
        missing_tasks = set(expected_tasks) - set(logits_by_task)
        unknown_tasks = set(logits_by_task) - set(expected_tasks)
        if missing_tasks or unknown_tasks:
            raise ValueError(
                "All-head diagnostics require exactly the configured tasks; "
                f"missing={sorted(missing_tasks)}, "
                f"unknown={sorted(unknown_tasks)}."
            )
        if target_task not in self.task_specs:
            raise KeyError(f"Unknown target task {target_task!r}.")
        if max_columns < 1:
            raise ValueError("max_columns must be positive.")

        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0.0, 1.0)

        target_np = target.detach().cpu().numpy()
        predictions: dict[str, np.ndarray] = {}
        for task in expected_tasks:
            logits = logits_by_task[task]
            expected_classes = self.task_specs[task].num_classes
            if logits.ndim != 3 or logits.shape[0] != expected_classes:
                raise ValueError(
                    f"Task {task!r} diagnostic logits must have shape "
                    f"({expected_classes}, H, W), got {tuple(logits.shape)}."
                )

            prediction = logits.argmax(dim=0).detach().cpu().numpy()
            if prediction.shape != target_np.shape:
                raise ValueError(
                    f"Task {task!r} prediction shape {prediction.shape} does "
                    f"not match target shape {target_np.shape}."
                )
            predictions[task] = prediction

        unique_classes = np.unique(
            np.concatenate(
                [
                    target_np.reshape(-1),
                    *(prediction.reshape(-1) for prediction in predictions.values()),
                ]
            )
        )
        colors = plt.get_cmap(cmap, len(unique_classes))(
            np.linspace(0.0, 1.0, len(unique_classes))
        )
        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0.0, 0.0, 0.0, 1.0]

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=-0.5, vmax=len(unique_classes) - 0.5)

        def encode_mask(mask: np.ndarray) -> np.ndarray:
            # unique_classes is sorted and contains every displayed value.
            return np.searchsorted(unique_classes, mask)

        panels: list[tuple[str, np.ndarray, bool]] = [
            ("Image", img_np, True),
            (f"GT ({target_task})", encode_mask(target_np), False),
            *[
                (f"Prediction ({task})", encode_mask(predictions[task]), False)
                for task in expected_tasks
            ],
        ]
        ncols = min(max_columns, len(panels))
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 5 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = axes.reshape(-1)

        for axis, (title, content, is_image) in zip(flat_axes, panels):
            axis.set_title(title)
            axis.axis("off")
            if is_image:
                axis.imshow(content)
            else:
                axis.imshow(
                    content,
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
        for axis in flat_axes[len(panels) :]:
            axis.axis("off")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=colors[index],
                lw=4,
                label=str(class_id),
            )
            for index, class_id in enumerate(unique_classes)
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(10, len(legend_handles)),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        image = Image.open(buffer).copy()
        buffer.close()
        return image

    @torch.compiler.disable
    def update_metric_stream(
        self,
        iou_metrics: nn.ModuleList,
        f1_metrics: nn.ModuleList,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_idx: int,
    ) -> None:
        for prediction, target in zip(predictions, targets):
            iou_metrics[dataloader_idx].update(
                prediction.unsqueeze(0),
                target.unsqueeze(0),
            )
            f1_metrics[dataloader_idx].update(
                prediction.unsqueeze(0),
                target.unsqueeze(0),
            )

    def finish_metric_stream(
        self,
        prefix: str,
        iou_metrics: nn.ModuleList,
        f1_metrics: nn.ModuleList,
    ) -> None:
        for dataloader_idx, task in enumerate(self.eval_task_names):
            iou_per_class = iou_metrics[dataloader_idx].compute()
            f1_per_class = f1_metrics[dataloader_idx].compute()
            iou_metrics[dataloader_idx].reset()
            f1_metrics[dataloader_idx].reset()

            for class_idx, value in enumerate(iou_per_class):
                self.log(
                    f"{prefix}_{dataloader_idx}_{task}_iou_{class_idx}",
                    value,
                    sync_dist=True,
                )
            for class_idx, value in enumerate(f1_per_class):
                self.log(
                    f"{prefix}_{dataloader_idx}_{task}_f1_{class_idx}",
                    value,
                    sync_dist=True,
                )

            self.log(
                f"{prefix}_{dataloader_idx}_{task}_miou",
                iou_per_class.mean(),
                sync_dist=True,
            )
            self.log(
                f"{prefix}_{dataloader_idx}_{task}_mf1",
                f1_per_class.mean(),
                sync_dist=True,
            )

    def eval_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        log_prefix: str,
    ):
        imgs, targets, _source_ids, _image_ids = self.unpack_batch(batch)
        task = self.eval_task_names[dataloader_idx]

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        plot_all_heads = batch_idx == 0
        crop_logits_by_task = self(
            crops,
            task=None if plot_all_heads else task,
        )
        logits_by_task = self.stitch_logits_by_task(
            crop_logits_by_task,
            origins,
            img_sizes,
        )
        logits = logits_by_task[task]
        target_maps = self._targets_to_per_pixel(targets)

        self.update_metric_stream(
            self.iou_metrics,
            self.f1_metrics,
            logits,
            target_maps,
            dataloader_idx,
        )

        if plot_all_heads:
            plot = self.plot_semantic_heads(
                imgs[0],
                target_maps[0],
                {
                    head_task: head_logits[0]
                    for head_task, head_logits in logits_by_task.items()
                },
                target_task=task,
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_{task}_all_heads_pred",
                plot,
                commit=False,
            )

        return logits

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def on_validation_epoch_end(self) -> None:
        self.finish_metric_stream("val", self.iou_metrics, self.f1_metrics)

    def on_test_epoch_end(self) -> None:
        self.finish_metric_stream("test", self.iou_metrics, self.f1_metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, targets, source_ids, image_ids = self.unpack_batch(batch)
        if not torch.is_tensor(imgs):
            imgs = torch.stack(list(imgs)).to(self.device)
        task = self.eval_task_names[dataloader_idx]

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops, task=task)[task]
        logits = self.stitch_crop_logits(crop_logits, origins, img_sizes)
        target_maps = self._targets_to_per_pixel(targets)

        outputs = []
        for index, logit in enumerate(logits):
            output = {
                "task": task,
                "logits": logit.detach().cpu(),
                "pred": logit.argmax(dim=0).detach().cpu(),
                "target": target_maps[index].detach().cpu(),
                "img": imgs[index].detach().cpu(),
            }
            if image_ids is not None:
                output["img_id"] = image_ids[index]
            if source_ids is not None:
                output["source_id"] = int(source_ids[index])
            outputs.append(output)

        return outputs

    def configure_optimizers(self):
        encoder_parameters = [
            parameter
            for parameter in self.network.encoder.parameters()
            if parameter.requires_grad
        ]
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in encoder_ids
        ]

        parameter_groups = []
        if base_parameters:
            parameter_groups.append({"params": base_parameters, "lr": self.lr})
        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": self.lr * self.lr_multiplier_encoder,
                }
            )
        if not parameter_groups:
            raise RuntimeError("The model has no trainable parameters.")

        optimizer = AdamW(parameter_groups, weight_decay=self.weight_decay)
        scheduler = PolynomialLR(
            optimizer,
            total_iters=int(self.trainer.estimated_stepping_batches),
            power=self.poly_lr_decay_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
