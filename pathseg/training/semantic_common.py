from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from pathseg.training.histo_loss import CrossEntropyDiceLoss
from pathseg.training.lightning_module import LightningModule


@dataclass(frozen=True, slots=True)
class SemanticTaskSpec:
    name: str
    num_classes: int
    loss_weight: float = 1.0
    class_weights: tuple[float, ...] | None = None
    loss_name: str = "cross_entropy"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Task name cannot be empty.")
        if "." in self.name:
            raise ValueError(
                f"Task name {self.name!r} cannot contain '.', because it is "
                "used as a ModuleDict key."
            )
        if self.num_classes < 2:
            raise ValueError(f"Task {self.name!r}: num_classes must be at least 2.")
        if self.loss_weight < 0:
            raise ValueError(f"Task {self.name!r}: loss_weight cannot be negative.")
        if self.class_weights is not None:
            if len(self.class_weights) != self.num_classes:
                raise ValueError(
                    f"Task {self.name!r}: expected {self.num_classes} class "
                    f"weights, got {len(self.class_weights)}."
                )
            if any(weight < 0 for weight in self.class_weights):
                raise ValueError(
                    f"Task {self.name!r}: class weights cannot be negative."
                )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> SemanticTaskSpec:
        values = dict(config)
        class_weights = values.get("class_weights")
        if class_weights is not None:
            values["class_weights"] = tuple(float(weight) for weight in class_weights)

        try:
            values["name"] = str(values["name"])
            values["num_classes"] = int(values["num_classes"])
            if "loss_weight" in values:
                values["loss_weight"] = float(values["loss_weight"])
            if "loss_name" in values:
                values["loss_name"] = str(values["loss_name"])
            return cls(**values)
        except KeyError as error:
            raise ValueError(
                f"Task configuration is missing required field {error.args[0]!r}: "
                f"{config!r}."
            ) from error
        except TypeError as error:
            name = values.get("name", "<unnamed>")
            raise ValueError(
                f"Invalid configuration for task {name!r}: {error}"
            ) from error


def parse_task_specs(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, SemanticTaskSpec]:
    if not tasks:
        raise ValueError("At least one semantic task must be configured.")
    if isinstance(tasks, Mapping):
        raise TypeError(
            "tasks must be an explicit list of task configurations, not a "
            "mapping keyed by task name."
        )

    parsed: dict[str, SemanticTaskSpec] = {}

    for index, config in enumerate(tasks):
        if not isinstance(config, Mapping):
            raise TypeError(
                f"tasks[{index}] must be a mapping, got {type(config).__name__}."
            )

        spec = SemanticTaskSpec.from_mapping(config)
        if spec.name in parsed:
            raise ValueError(f"Duplicate task name: {spec.name!r}.")
        parsed[spec.name] = spec

    if not any(spec.loss_weight > 0 for spec in parsed.values()):
        raise ValueError("At least one task must have a positive loss weight.")

    return parsed


@dataclass(frozen=True, slots=True)
class TaskRoute:
    """Indices for one task without a device-to-host synchronization.

    ``host_indices`` selects Python containers such as the raw target list.
    ``device_indices`` selects tensors that already live with the model. Both
    are created from CPU task-name metadata, so selecting Python targets never
    requires calling ``tolist()`` on a CUDA tensor.
    """

    host_indices: tuple[int, ...]
    device_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.device_indices.dtype != torch.long:
            raise TypeError("TaskRoute.device_indices must have dtype torch.long.")
        if self.device_indices.ndim != 1:
            raise ValueError("TaskRoute.device_indices must be one-dimensional.")
        if len(self.host_indices) != self.device_indices.numel():
            raise ValueError(
                "TaskRoute host/device indices must contain the same number of samples."
            )

    def __len__(self) -> int:
        return len(self.host_indices)


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

    raise ValueError(f"Unknown loss {spec.loss_name!r} for task {spec.name!r}.")


class SemanticLightningModule(LightningModule):
    """Shared multi-task semantic evaluation and optimization infrastructure.

    `eval_task_names[dataloader_idx]` identifies the head evaluated by each
    validation/test dataloader. Training batches may mix tasks and carry one
    string task name per sample. Training converts those names to paired host
    and device routes, encodes the mixed batch once, and decodes each head only
    for its own samples.
    """

    def __init__(
        self,
        *,
        network: nn.Module,
        tasks: list[dict[str, Any]],
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
        self.eval_task_names = list(self.task_specs.keys())

        self.ignore_idx = int(ignore_idx)
        self.poly_lr_decay_power = float(poly_lr_decay_power)

        self.criteria = nn.ModuleDict(
            {
                name: build_criterion(spec, self.ignore_idx)
                for name, spec in self.task_specs.items()
            }
        )

        # TODO: add metric for position info
        self.iou_metrics, self.f1_metrics = self._make_metric_streams()

    @property
    def num_classes_by_task(self) -> dict[str, int]:
        return {name: spec.num_classes for name, spec in self.task_specs.items()}

    def forward(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # Images arrive as uint8-like 0..255 tensors in the existing pipeline;
        # the encoder wrapper performs its own mean/std normalization.
        return self.network(imgs / 255.0, task=task)

    def routed_forward(
        self,
        imgs: torch.Tensor,
        routes: Mapping[str, TaskRoute],
    ) -> dict[str, torch.Tensor]:
        """Encode a mixed batch once and route feature maps to task heads."""
        if not routes:
            raise ValueError("At least one non-empty task route is required.")

        unknown_tasks = set(routes) - set(self.task_specs)
        if unknown_tasks:
            raise ValueError(f"Unknown routed tasks: {sorted(unknown_tasks)}.")

        input_size = tuple(imgs.shape[-2:])
        feature_maps = self.network.encode(imgs / 255.0)
        logits_by_task: dict[str, torch.Tensor] = {}

        for task, route in routes.items():
            if not route:
                raise ValueError(f"Task route {task!r} is empty.")
            if route.device_indices.device != imgs.device:
                raise ValueError(
                    f"Task route {task!r} is on "
                    f"{route.device_indices.device}, but images are on "
                    f"{imgs.device}."
                )

            task_feature_maps = tuple(
                feature_map.index_select(0, route.device_indices)
                for feature_map in feature_maps
            )
            task_logits = self.network.decode(task_feature_maps, task=task)
            if set(task_logits) != {task}:
                raise ValueError(
                    f"Decoding task {task!r} must return exactly that head; "
                    f"got {sorted(task_logits)}."
                )

            task_logits = self.network.ensure_input_resolution(
                task_logits,
                input_size,
            )
            logits = task_logits[task]
            if logits.shape[0] != len(route):
                raise ValueError(
                    f"Task {task!r} returned {logits.shape[0]} samples for "
                    f"a route containing {len(route)} samples."
                )
            logits_by_task[task] = logits

        return logits_by_task

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
        # TODO: remove if for one task it is the same
        if len(batch) == 2:
            imgs, targets = batch
            return imgs, targets, None, None
        if len(batch) == 3:
            imgs, targets, task_names = batch
            return imgs, targets, task_names, None
        if len(batch) == 4:
            return batch
        raise ValueError(
            "Expected (imgs, targets), (imgs, targets, task_names), or "
            "(imgs, targets, task_names, image_ids)."
        )

    @staticmethod
    def _normalize_task_names(
        task_names,
        *,
        batch_size: int,
    ) -> tuple[str, ...] | None:
        # TODO: rm or simplify
        if task_names is None:
            return None
        if isinstance(task_names, str):
            if batch_size != 1:
                raise ValueError(
                    "A scalar task name is valid only for a batch of size 1."
                )
            return (task_names,)

        if len(task_names) != batch_size:
            raise ValueError(
                f"Received {len(task_names)} task names for batch size {batch_size}."
            )

        normalized: list[str] = []
        for index, name in enumerate(task_names):
            if not isinstance(name, str):
                raise TypeError(
                    f"task_names[{index}] must be a string, got "
                    f"{type(name).__name__}. Numeric source IDs are no longer "
                    "supported."
                )
            if not name:
                raise ValueError(f"task_names[{index}] cannot be empty.")
            normalized.append(name)
        return tuple(normalized)

    def _evaluation_task(
        self,
        task_names,
        *,
        batch_size: int,
        dataloader_idx: int,
    ) -> str:
        if not 0 <= dataloader_idx < len(self.eval_task_names):
            raise IndexError(
                f"dataloader_idx={dataloader_idx} has no matching entry in "
                f"eval_task_names={self.eval_task_names}."
            )

        expected_task = self.eval_task_names[dataloader_idx]
        normalized = self._normalize_task_names(
            task_names,
            batch_size=batch_size,
        )
        if normalized is None:
            return expected_task

        batch_tasks = set(normalized)
        if len(batch_tasks) != 1:
            raise ValueError(
                "Every evaluation dataloader must produce task-homogeneous "
                f"batches, got tasks={sorted(batch_tasks)}."
            )

        batch_task = next(iter(batch_tasks))
        if batch_task != expected_task:
            raise ValueError(
                f"Evaluation dataloader {dataloader_idx} produced task "
                f"{batch_task!r}, but eval_task_names expects "
                f"{expected_task!r}."
            )
        return batch_task

    @staticmethod
    def _select_batch(value, route: TaskRoute):
        if torch.is_tensor(value):
            if value.device == route.device_indices.device:
                indices = route.device_indices
            else:
                # Covers CPU tensor targets without copying CUDA indices back
                # to the host.
                indices = torch.tensor(
                    route.host_indices,
                    dtype=torch.long,
                    device=value.device,
                )
            return value.index_select(0, indices)
        return [value[index] for index in route.host_indices]

    def task_routes(
        self,
        task_names,
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, TaskRoute]:
        """Build host and device indices directly from CPU task metadata."""
        # TODO: simplify
        normalized = self._normalize_task_names(
            task_names,
            batch_size=batch_size,
        )
        if normalized is None:
            if len(self.task_specs) == 1:
                only_task = next(iter(self.task_specs))
                normalized = (only_task,) * batch_size
            else:
                raise ValueError(
                    "Multi-task training batches must contain one task name per sample."
                )

        unknown_tasks = set(normalized) - set(self.task_specs)
        if unknown_tasks:
            raise ValueError(f"Batch contains unknown tasks: {sorted(unknown_tasks)}.")

        indices_by_task: dict[str, list[int]] = {task: [] for task in self.task_specs}
        for sample_index, task in enumerate(normalized):
            indices_by_task[task].append(sample_index)

        routes: dict[str, TaskRoute] = {}
        for task, indices in indices_by_task.items():
            if not indices:
                continue
            host_indices = tuple(indices)
            routes[task] = TaskRoute(
                host_indices=host_indices,
                device_indices=torch.tensor(
                    host_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )

        if not routes:
            raise ValueError(
                "The training batch did not contain samples for any configured task."
            )
        return routes

    def task_loss(
        self,
        task: str,
        logits: torch.Tensor,
        targets,
        route: TaskRoute,
    ) -> torch.Tensor:
        if task not in self.task_specs:
            raise KeyError(f"Unknown task {task!r}.")
        if not route:
            raise ValueError(f"Cannot compute loss for empty task route {task!r}.")
        if logits.shape[0] != len(route):
            raise ValueError(
                f"Task {task!r} received {logits.shape[0]} routed logits for "
                f"{len(route)} targets."
            )

        targets_subset = self._select_batch(targets, route)
        target_maps = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        target_tensor = torch.stack(target_maps).long().to(logits.device)

        return self.criteria[task](logits, target_tensor)

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
        imgs, targets, task_names, _image_ids = self.unpack_batch(batch)
        batch_size = imgs.shape[0] if torch.is_tensor(imgs) else len(imgs)
        task = self._evaluation_task(
            task_names,
            batch_size=batch_size,
            dataloader_idx=dataloader_idx,
        )

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
        target_maps = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

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
        imgs, targets, task_names, image_ids = self.unpack_batch(batch)
        batch_size = imgs.shape[0] if torch.is_tensor(imgs) else len(imgs)
        task = self._evaluation_task(
            task_names,
            batch_size=batch_size,
            dataloader_idx=dataloader_idx,
        )
        if not torch.is_tensor(imgs):
            imgs = torch.stack(list(imgs)).to(self.device)

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops, task=task)[task]
        logits = self.stitch_crop_logits(crop_logits, origins, img_sizes)
        target_maps = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        outputs = []
        for index, logit in enumerate(logits):
            output = {
                "task_name": task,
                "logits": logit.detach().cpu(),
                "pred": logit.argmax(dim=0).detach().cpu(),
                "target": target_maps[index].detach().cpu(),
                "img": imgs[index].detach().cpu(),
            }
            if image_ids is not None:
                output["img_id"] = image_ids[index]
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
