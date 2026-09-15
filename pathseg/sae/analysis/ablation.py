from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pathseg.sae.analysis.batches import unpack_multitask_batch
from pathseg.sae.analysis.cli import build_analysis_objects
from pathseg.sae.analysis.interventions import (
    ablate_reconstructed_tokens,
    decode_reconstructed_tokens,
)
from pathseg.sae.analysis.latent_selection import (
    LatentSpec,
    resolve_latent_specs,
    unique_latent_ids,
)
from pathseg.sae.analysis.runner import (
    _autocast_context,
    _forward_sae_output,
    _named_validation_loaders,
    _task_specs_from_lightning_module,
)
LOGGER = logging.getLogger(__name__)


def _write_csv(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_task_logits(decoded: Any, task: str) -> torch.Tensor:
    if torch.is_tensor(decoded):
        logits = decoded
    elif isinstance(decoded, Mapping):
        if task not in decoded:
            raise KeyError(
                f"Semantic decoder did not return logits for task {task!r}."
            )
        logits = decoded[task]
    else:
        raise TypeError(
            "Semantic decoder must return a tensor or a task-to-tensor mapping."
        )
    if not torch.is_tensor(logits) or logits.ndim != 4:
        raise TypeError("Crop logits must be a [num_crops,C,H,W] tensor.")
    return logits


def _stitched_predictions(
    lightning_module,
    decoded: Any,
    *,
    task: str,
    origins: Any,
    image_sizes: Any,
) -> list[torch.Tensor]:
    crop_logits = _select_task_logits(decoded, task)
    stitched = lightning_module.stitch_crop_logits(
        crop_logits,
        origins,
        image_sizes,
    )
    return [logit.argmax(dim=0).detach().cpu() for logit in stitched]


def _update_confusion(
    confusion: torch.Tensor,
    *,
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    num_classes: int,
    ignore_idx: int,
) -> int:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Received {len(predictions)} predictions for {len(targets)} targets."
        )

    valid_pixel_count = 0
    for prediction, target in zip(predictions, targets, strict=True):
        prediction = prediction.long().cpu()
        target = target.long().cpu()
        if prediction.shape != target.shape:
            raise ValueError(
                "Stitched prediction and target sizes differ: "
                f"{tuple(prediction.shape)} versus {tuple(target.shape)}."
            )
        valid = target != ignore_idx
        invalid_targets = valid & ((target < 0) | (target >= num_classes))
        if torch.any(invalid_targets):
            labels = torch.unique(target[invalid_targets]).tolist()
            raise ValueError(f"Target contains invalid class IDs: {labels}.")
        invalid_predictions = (prediction < 0) | (prediction >= num_classes)
        if torch.any(invalid_predictions):
            labels = torch.unique(prediction[invalid_predictions]).tolist()
            raise ValueError(f"Prediction contains invalid class IDs: {labels}.")

        target_valid = target[valid]
        prediction_valid = prediction[valid]
        indices = target_valid * num_classes + prediction_valid
        confusion += torch.bincount(
            indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        valid_pixel_count += int(valid.sum().item())
    return valid_pixel_count


def _pixel_flips(
    baseline: Sequence[torch.Tensor],
    ablated: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    ignore_idx: int,
) -> int:
    if not (len(baseline) == len(ablated) == len(targets)):
        raise ValueError("Baseline, ablated, and target batch sizes differ.")
    return sum(
        int(((base != changed) & (target.cpu() != ignore_idx)).sum().item())
        for base, changed, target in zip(
            baseline,
            ablated,
            targets,
            strict=True,
        )
    )


def _confusion_metrics(confusion: torch.Tensor) -> dict[str, torch.Tensor | float]:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    gt_pixels = confusion.sum(dim=1)
    predicted_pixels = confusion.sum(dim=0)
    union = gt_pixels + predicted_pixels - true_positive
    iou = torch.zeros_like(union)
    valid_union = union > 0
    iou[valid_union] = true_positive[valid_union] / union[valid_union]

    # Match MulticlassJaccardIndex(average=None).mean() from semantic
    # validation: every configured class is included, and a zero-union class
    # contributes zero. The denominator is therefore fixed across ablations.
    included = torch.ones_like(gt_pixels, dtype=torch.bool)
    mean_iou = float(iou.mean().item())
    return {
        "iou": iou,
        "gt_pixels": gt_pixels,
        "predicted_pixels": predicted_pixels,
        "included": included,
        "miou": mean_iou,
    }


def _json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _csv_float(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def _selection_metadata(
    specs: Sequence[LatentSpec],
) -> dict[int, str]:
    by_latent: dict[int, list[dict[str, Any]]] = {}
    for spec in specs:
        by_latent.setdefault(spec.latent_id, []).append(spec.as_dict())
    return {
        latent_id: json.dumps(values, separators=(",", ":"))
        for latent_id, values in by_latent.items()
    }


@torch.no_grad()
def run_ablation(
    *,
    lightning_module,
    data_module,
    latent_specs: Sequence[LatentSpec],
    output_dir: str | Path,
    tasks: Sequence[str] = (),
    device: str | torch.device | None = None,
    precision: str = "16-mixed",
    max_batches_per_loader: int | None = None,
) -> dict[str, Any]:
    """Measure full-image segmentation changes after individual zero-ablations."""

    if not latent_specs:
        raise ValueError("At least one latent specification is required.")
    if max_batches_per_loader is not None and max_batches_per_loader <= 0:
        raise ValueError("max_batches_per_loader must be positive.")

    segmenter = lightning_module.network
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    segmenter.to(device)
    segmenter.eval()

    task_specs = _task_specs_from_lightning_module(lightning_module)
    requested_tasks = tuple(dict.fromkeys(str(task) for task in tasks))
    unknown_tasks = set(requested_tasks).difference(task_specs)
    if unknown_tasks:
        raise ValueError(f"Unknown ablation tasks: {sorted(unknown_tasks)}.")
    enabled_tasks = set(requested_tasks or tuple(task_specs))

    latent_ids = unique_latent_ids(latent_specs)
    sae = getattr(segmenter, "sae", None)
    num_latents = int(getattr(sae, "num_latents", 0))
    if num_latents <= 0:
        raise TypeError("segmenter.sae must expose a positive num_latents.")
    invalid_latents = [
        latent_id
        for latent_id in latent_ids
        if not 0 <= latent_id < num_latents
    ]
    if invalid_latents:
        raise IndexError(f"Latent IDs are out of bounds: {invalid_latents}.")

    ignore_idx = int(lightning_module.ignore_idx)
    target_converter = getattr(lightning_module, "_targets_to_per_pixel", None)
    if not callable(target_converter):
        target_converter = None

    baseline_confusions = {
        task: torch.zeros(
            (task_specs[task].num_classes, task_specs[task].num_classes),
            dtype=torch.int64,
        )
        for task in enabled_tasks
    }
    ablated_confusions = {
        task: {
            latent_id: torch.zeros_like(baseline_confusions[task])
            for latent_id in latent_ids
        }
        for task in enabled_tasks
    }
    valid_pixels = {task: 0 for task in enabled_tasks}
    pixel_flips = {
        task: {latent_id: 0 for latent_id in latent_ids}
        for task in enabled_tasks
    }
    latent_token_count = {task: 0 for task in enabled_tasks}
    latent_firing_sum = {
        task: torch.zeros(len(latent_ids), dtype=torch.float64)
        for task in enabled_tasks
    }
    latent_activation_sum = {
        task: torch.zeros(len(latent_ids), dtype=torch.float64)
        for task in enabled_tasks
    }
    processed_batches = {task: 0 for task in enabled_tasks}
    processed_images = {task: 0 for task in enabled_tasks}

    id_tensor = torch.as_tensor(latent_ids, dtype=torch.long, device=device)
    for dataset_name, loader in _named_validation_loaders(data_module):
        for batch_index, raw_batch in enumerate(loader):
            if (
                max_batches_per_loader is not None
                and batch_index >= max_batches_per_loader
            ):
                break
            batch = unpack_multitask_batch(
                raw_batch,
                ignore_idx=ignore_idx,
                target_converter=target_converter,
            )
            batch_tasks = set(batch.task_names)
            if len(batch_tasks) != 1:
                raise ValueError(
                    "Ablation requires task-homogeneous validation batches, "
                    f"got {sorted(batch_tasks)} from {dataset_name!r}."
                )
            task = next(iter(batch_tasks))
            if task not in task_specs:
                raise KeyError(f"Unknown validation task {task!r}.")
            if task not in enabled_tasks:
                continue

            images = batch.images
            crops, origins, image_sizes = lightning_module.window_imgs_semantic(
                images
            )
            crops = crops.to(device, non_blocking=True)
            output = _forward_sae_output(
                segmenter,
                crops,
                device=device,
                precision=precision,
                input_scale=1.0 / 255.0,
            )
            latents = output["latents"]
            reconstructed_tokens = output["reconstructed_tokens"]
            feature_maps = output["feature_maps"]
            del output, crops

            selected_latents = latents.index_select(-1, id_tensor)
            latent_token_count[task] += int(
                selected_latents.shape[0] * selected_latents.shape[1]
            )
            latent_firing_sum[task] += (
                (selected_latents > 0)
                .sum(dim=(0, 1))
                .detach()
                .to(dtype=torch.float64, device="cpu")
            )
            latent_activation_sum[task] += (
                selected_latents.sum(dim=(0, 1))
                .detach()
                .to(dtype=torch.float64, device="cpu")
            )
            del selected_latents

            with _autocast_context(device, precision):
                baseline_decoded = decode_reconstructed_tokens(
                    segmenter,
                    reconstructed_tokens=reconstructed_tokens,
                    feature_maps=feature_maps,
                    task=task,
                )
            baseline_predictions = _stitched_predictions(
                lightning_module,
                baseline_decoded,
                task=task,
                origins=origins,
                image_sizes=image_sizes,
            )
            target_maps = [target.detach().cpu() for target in batch.targets]
            batch_valid_pixels = _update_confusion(
                baseline_confusions[task],
                predictions=baseline_predictions,
                targets=target_maps,
                num_classes=task_specs[task].num_classes,
                ignore_idx=ignore_idx,
            )
            valid_pixels[task] += batch_valid_pixels

            del baseline_decoded
            for latent_id in latent_ids:
                ablated_tokens = ablate_reconstructed_tokens(
                    sae,
                    latents=latents,
                    reconstructed_tokens=reconstructed_tokens,
                    latent_ids=latent_id,
                )
                with _autocast_context(device, precision):
                    ablated_decoded = decode_reconstructed_tokens(
                        segmenter,
                        reconstructed_tokens=ablated_tokens,
                        feature_maps=feature_maps,
                        task=task,
                    )
                ablated_predictions = _stitched_predictions(
                    lightning_module,
                    ablated_decoded,
                    task=task,
                    origins=origins,
                    image_sizes=image_sizes,
                )
                _update_confusion(
                    ablated_confusions[task][latent_id],
                    predictions=ablated_predictions,
                    targets=target_maps,
                    num_classes=task_specs[task].num_classes,
                    ignore_idx=ignore_idx,
                )
                pixel_flips[task][latent_id] += _pixel_flips(
                    baseline_predictions,
                    ablated_predictions,
                    target_maps,
                    ignore_idx=ignore_idx,
                )
                del ablated_tokens, ablated_decoded, ablated_predictions

            processed_batches[task] += 1
            processed_images[task] += len(target_maps)
            LOGGER.info(
                "Ablated %d latents on %s batch %d (%s, %d images)",
                len(latent_ids),
                dataset_name,
                batch_index,
                task,
                len(target_maps),
            )

    missing_tasks = [
        task for task in enabled_tasks if processed_batches[task] == 0
    ]
    if missing_tasks:
        raise RuntimeError(
            f"No validation batches were processed for tasks: {missing_tasks}."
        )

    selection = _selection_metadata(latent_specs)
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    json_tasks: dict[str, Any] = {}
    for task in sorted(enabled_tasks):
        task_spec = task_specs[task]
        baseline_metrics = _confusion_metrics(baseline_confusions[task])
        baseline_miou = float(baseline_metrics["miou"])
        task_json: dict[str, Any] = {
            "num_classes": task_spec.num_classes,
            "class_names": [
                task_spec.class_name(index)
                for index in range(task_spec.num_classes)
            ],
            "processed_batches": processed_batches[task],
            "processed_images": processed_images[task],
            "valid_pixels": valid_pixels[task],
            "baseline_miou": _json_float(baseline_miou),
            "latents": {},
        }
        for latent_position, latent_id in enumerate(latent_ids):
            ablated_metrics = _confusion_metrics(
                ablated_confusions[task][latent_id]
            )
            ablated_miou = float(ablated_metrics["miou"])
            delta_miou = ablated_miou - baseline_miou
            miou_drop = -delta_miou
            token_count = latent_token_count[task]
            firing_count = float(
                latent_firing_sum[task][latent_position].item()
            )
            activation_sum = float(
                latent_activation_sum[task][latent_position].item()
            )
            firing_rate = firing_count / max(token_count, 1)
            mean_active_activation = activation_sum / max(firing_count, 1.0)
            flip_rate = pixel_flips[task][latent_id] / max(
                valid_pixels[task], 1
            )
            row = {
                "latent_id": latent_id,
                "selected_for": selection[latent_id],
                "task": task,
                "processed_images": processed_images[task],
                "valid_pixels": valid_pixels[task],
                "crop_token_count": token_count,
                "firing_rate": firing_rate,
                "mean_active_activation": mean_active_activation,
                "pixel_flip_rate": flip_rate,
                "baseline_miou": _csv_float(baseline_miou),
                "ablated_miou": _csv_float(ablated_miou),
                "delta_miou": _csv_float(delta_miou),
                "miou_drop": _csv_float(miou_drop),
            }
            overall_rows.append(row)

            latent_json = {
                "selected_for": json.loads(selection[latent_id]),
                "firing_rate": firing_rate,
                "mean_active_activation": mean_active_activation,
                "pixel_flip_rate": flip_rate,
                "ablated_miou": _json_float(ablated_miou),
                "delta_miou": _json_float(delta_miou),
                "miou_drop": _json_float(miou_drop),
                "classes": [],
            }
            baseline_iou = baseline_metrics["iou"]
            ablated_iou = ablated_metrics["iou"]
            for class_index in range(task_spec.num_classes):
                base_value = float(baseline_iou[class_index].item())
                changed_value = float(ablated_iou[class_index].item())
                delta_value = changed_value - base_value
                class_row = {
                    "latent_id": latent_id,
                    "task": task,
                    "class_index": class_index,
                    "class_name": task_spec.class_name(class_index),
                    "included_in_miou": bool(
                        baseline_metrics["included"][class_index].item()
                    ),
                    "gt_pixels": int(
                        baseline_metrics["gt_pixels"][class_index].item()
                    ),
                    "baseline_predicted_pixels": int(
                        baseline_metrics["predicted_pixels"][class_index].item()
                    ),
                    "ablated_predicted_pixels": int(
                        ablated_metrics["predicted_pixels"][class_index].item()
                    ),
                    "baseline_iou": _csv_float(base_value),
                    "ablated_iou": _csv_float(changed_value),
                    "delta_iou": _csv_float(delta_value),
                    "iou_drop": _csv_float(-delta_value),
                }
                class_rows.append(class_row)
                latent_json["classes"].append(
                    {
                        "class_index": class_index,
                        "class_name": task_spec.class_name(class_index),
                        "included_in_miou": class_row["included_in_miou"],
                        "gt_pixels": class_row["gt_pixels"],
                        "baseline_iou": _json_float(base_value),
                        "ablated_iou": _json_float(changed_value),
                        "delta_iou": _json_float(delta_value),
                        "iou_drop": _json_float(-delta_value),
                    }
                )
            task_json["latents"][str(latent_id)] = latent_json
        json_tasks[task] = task_json

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_fields = list(overall_rows[0])
    class_fields = list(class_rows[0])
    _write_csv(
        output_dir / "ablation_overall.csv",
        rows=overall_rows,
        fieldnames=overall_fields,
    )
    _write_csv(
        output_dir / "ablation_per_class.csv",
        rows=class_rows,
        fieldnames=class_fields,
    )
    report = {
        "baseline": "SAE reconstruction",
        "intervention": "set one latent to zero at every crop token",
        "metric_scope": "stitched full validation images",
        "miou_class_inclusion": (
            "all configured classes; zero IoU when union is zero"
        ),
        "latent_specs": [spec.as_dict() for spec in latent_specs],
        "tasks": json_tasks,
    }
    with (output_dir / "ablation.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure stitched full-image segmentation changes after "
            "individual latent zero-ablation."
        )
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="YAML manifest containing a latents sequence.",
    )
    parser.add_argument(
        "--latent",
        action="append",
        default=[],
        metavar="ID[:TASK[:CLASS]]",
        help="Repeat for each latent. Task/class are selection metadata only.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Restrict evaluation to a task; repeat as needed.",
    )
    parser.add_argument(
        "--max-latents",
        type=int,
        default=64,
        help="Safety limit for individual interventions; set 0 for unlimited.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=("32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"),
    )
    parser.add_argument(
        "--max-batches-per-loader",
        type=int,
        default=None,
        help="Limit each validation loader for a quick smoke test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    latent_specs = resolve_latent_specs(
        manifest_path=args.manifest,
        values=args.latent,
    )
    num_unique = len(unique_latent_ids(latent_specs))
    if args.max_latents < 0:
        raise ValueError("max_latents cannot be negative.")
    if args.max_latents and num_unique > args.max_latents:
        raise ValueError(
            f"Requested {num_unique} individual ablations, exceeding "
            f"--max-latents={args.max_latents}. Increase the limit explicitly."
        )

    module, data_module = build_analysis_objects(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    run_ablation(
        lightning_module=module,
        data_module=data_module,
        latent_specs=latent_specs,
        output_dir=args.output_dir,
        tasks=args.task,
        device=args.device,
        precision=args.precision,
        max_batches_per_loader=args.max_batches_per_loader,
    )
    LOGGER.info("Saved ablation report to %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
