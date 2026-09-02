from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pathseg.sae.analysis.attribution import (
    compute_task_attributions,
    decoder_column_norms,
    resolve_task_heads,
    select_relevant_latents,
)
from pathseg.sae.analysis.batches import unpack_multitask_batch
from pathseg.sae.analysis.collector import (
    StreamingSAECollector,
    TopActivationCollector,
)
from pathseg.sae.analysis.report import save_analysis_report
from pathseg.sae.analysis.types import AnalysisResult, TaskSpec

LOGGER = logging.getLogger(__name__)


def _normalize_task_specs(
    tasks: Mapping[str, int | TaskSpec | Mapping[str, Any]],
) -> dict[str, TaskSpec]:
    result: dict[str, TaskSpec] = {}
    for task_name, value in tasks.items():
        if isinstance(value, TaskSpec):
            spec = value
            if spec.name != task_name:
                raise ValueError(
                    f"Task mapping key {task_name!r} does not match "
                    f"TaskSpec name {spec.name!r}."
                )
        elif isinstance(value, int):
            spec = TaskSpec(name=task_name, num_classes=value)
        elif isinstance(value, Mapping):
            class_names = tuple(str(name) for name in value.get("class_names", ()))
            spec = TaskSpec(
                name=task_name,
                num_classes=int(value["num_classes"]),
                class_names=class_names,
            )
        else:
            raise TypeError(
                f"Unsupported task specification for {task_name!r}: "
                f"{type(value).__name__}."
            )
        result[task_name] = spec
    if not result:
        raise ValueError("At least one task must be configured.")
    return result


def _named_validation_loaders(data_module: Any) -> list[tuple[str, Any]]:
    data_module.setup(stage="validate")
    loaders = data_module.val_dataloader()
    if not isinstance(loaders, (tuple, list)):
        loaders = [loaders]

    wrapped = getattr(data_module, "val_wrapped", None)
    if wrapped is not None:
        names = [str(name) for name, _ in wrapped]
    else:
        names = [f"validation_{index}" for index in range(len(loaders))]

    if len(names) != len(loaders):
        raise ValueError(
            f"Data module exposes {len(names)} validation dataset names but "
            f"returned {len(loaders)} loaders."
        )
    return list(zip(names, loaders, strict=True))


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision in {"32", "32-true"}:
        return nullcontext()
    if precision in {"16", "16-mixed"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision in {"bf16", "bf16-mixed"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(
        "precision must be one of '32', '32-true', '16', '16-mixed', "
        "'bf16', or 'bf16-mixed'."
    )


def _forward_sae_output(
    segmenter: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    input_scale: float,
) -> Mapping[str, Any]:
    forward_sae = getattr(segmenter, "forward_sae", None)
    if not callable(forward_sae):
        raise TypeError("segmenter must expose forward_sae(images).")

    with _autocast_context(device, precision):
        output = forward_sae(images * input_scale)
    if not isinstance(output, Mapping):
        raise TypeError("forward_sae must return a mapping.")
    required = {"latents", "reconstructed_tokens", "feature_maps"}
    missing = required.difference(output)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"forward_sae output is missing: {names}.")

    feature_maps = output["feature_maps"]
    if not isinstance(feature_maps, Sequence) or not feature_maps:
        raise TypeError("feature_maps must be a non-empty sequence.")
    latents = output["latents"]
    reconstructed_tokens = output["reconstructed_tokens"]
    if not torch.is_tensor(latents) or latents.ndim != 3:
        raise TypeError("latents must be a [B,N,L] tensor.")
    if not torch.is_tensor(reconstructed_tokens) or reconstructed_tokens.ndim != 3:
        raise TypeError("reconstructed_tokens must be a [B,N,D] tensor.")
    if latents.shape[:2] != reconstructed_tokens.shape[:2]:
        raise ValueError("Latent and reconstructed token grids do not match.")
    return output


def _forward_sae(
    segmenter: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    input_scale: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    output = _forward_sae_output(
        segmenter,
        images,
        device=device,
        precision=precision,
        input_scale=input_scale,
    )
    feature_maps = output["feature_maps"]
    spatial_size = tuple(int(value) for value in feature_maps[-1].shape[-2:])
    return output["latents"], spatial_size


def _task_specs_from_lightning_module(module: nn.Module) -> dict[str, TaskSpec]:
    raw_specs = getattr(module, "task_specs", None)
    if not isinstance(raw_specs, Mapping) or not raw_specs:
        raise TypeError(
            "Lightning module does not expose a non-empty task_specs mapping."
        )
    return {
        str(name): TaskSpec(
            name=str(name),
            num_classes=int(spec.num_classes),
            class_names=tuple(getattr(spec, "class_names", ()) or ()),
        )
        for name, spec in raw_specs.items()
    }


def _window_batch(
    module: nn.Module,
    batch,
    *,
    task_specs: Mapping[str, TaskSpec],
    ignore_idx: int,
):
    """Apply the training module's validation windowing to images and masks.

    Masks are converted to one-hot maps (plus an ignore channel), windowed with
    exactly the same routine, and converted back with argmax.  This works for
    both the default semantic windowing and channel-agnostic tilers.
    """

    crop_images: list[torch.Tensor] = []
    crop_targets: list[torch.Tensor] = []
    crop_tasks: list[str] = []
    crop_image_ids: list[str] = []
    crop_sample_ids: list[str] = []

    for sample_index in range(batch.images.shape[0]):
        task_name = batch.task_names[sample_index]
        if task_name not in task_specs:
            raise KeyError(f"Unknown validation task {task_name!r}.")
        spec = task_specs[task_name]
        image = batch.images[sample_index]
        target = batch.targets[sample_index].long()
        valid = target != ignore_idx
        invalid = valid & ((target < 0) | (target >= spec.num_classes))
        if torch.any(invalid):
            labels = torch.unique(target[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"Task {task_name!r} contains invalid target labels: {labels}."
            )

        safe_target = target.masked_fill(~valid, spec.num_classes)
        target_channels = F.one_hot(
            safe_target,
            num_classes=spec.num_classes + 1,
        ).permute(2, 0, 1).float()

        image_windows, _, _ = module.window_imgs_semantic((image,))
        target_windows, _, _ = module.window_imgs_semantic((target_channels,))
        if image_windows.shape[0] != target_windows.shape[0]:
            raise RuntimeError(
                "Image and target windowing returned different numbers of crops."
            )
        if image_windows.shape[-2:] != target_windows.shape[-2:]:
            raise RuntimeError("Image and target crop sizes do not match.")

        target_labels = target_windows.argmax(dim=1).long()
        target_labels[target_labels == spec.num_classes] = ignore_idx
        source_image_id = batch.image_ids[sample_index]

        for crop_index in range(image_windows.shape[0]):
            crop_images.append(image_windows[crop_index])
            crop_targets.append(target_labels[crop_index])
            crop_tasks.append(task_name)
            crop_image_ids.append(source_image_id)
            crop_sample_ids.append(f"{source_image_id}::crop_{crop_index}")

    from pathseg.sae.analysis.batches import AnalysisBatch

    return AnalysisBatch(
        images=torch.stack(crop_images),
        targets=torch.stack(crop_targets),
        task_names=tuple(crop_tasks),
        image_ids=tuple(crop_image_ids),
        sample_ids=tuple(crop_sample_ids),
    )


def _iter_batches(
    named_loaders: Sequence[tuple[str, Any]],
    *,
    ignore_idx: int,
    target_converter: Callable[[Any], Any] | None = None,
    max_batches_per_loader: int | None,
):
    for dataset_name, loader in named_loaders:
        for batch_index, raw_batch in enumerate(loader):
            if (
                max_batches_per_loader is not None
                and batch_index >= max_batches_per_loader
            ):
                break
            yield dataset_name, batch_index, unpack_multitask_batch(
                raw_batch,
                ignore_idx=ignore_idx,
                target_converter=target_converter,
            )


@torch.no_grad()
def analyze_sae(
    *,
    segmenter: nn.Module | None = None,
    lightning_module: nn.Module | None = None,
    data_module: Any,
    tasks: Mapping[str, int | TaskSpec | Mapping[str, Any]] | None = None,
    ignore_idx: int | None = None,
    output_dir: str | Path,
    heads: Mapping[str, nn.Module] | None = None,
    input_scale: float | None = None,
    window_inputs: bool = True,
    device: str | torch.device | None = None,
    precision: str = "16-mixed",
    top_latents_per_class: int = 10,
    global_top_latents: int = 32,
    max_selected_latents: int = 128,
    top_examples_per_latent: int = 16,
    max_examples_per_image: int = 2,
    max_batches_per_loader: int | None = None,
) -> AnalysisResult:
    """Analyze an already-loaded SAE semantic segmenter on validation data.

    The function makes one streaming statistics pass and, when requested, a
    second pass that stores top-activation metadata for selected latents.
    """

    if lightning_module is not None:
        if segmenter is not None:
            raise ValueError("Pass either segmenter or lightning_module, not both.")
        segmenter = getattr(lightning_module, "network", None)
        if not isinstance(segmenter, nn.Module):
            raise TypeError("lightning_module.network must be an nn.Module.")
        if tasks is None:
            task_specs = _task_specs_from_lightning_module(lightning_module)
        else:
            task_specs = _normalize_task_specs(tasks)
        if ignore_idx is None:
            ignore_idx = int(getattr(lightning_module, "ignore_idx"))
        if input_scale is None:
            input_scale = 1.0 / 255.0
    else:
        if segmenter is None:
            raise ValueError("segmenter or lightning_module is required.")
        if tasks is None or ignore_idx is None:
            raise ValueError(
                "tasks and ignore_idx are required when no Lightning module is passed."
            )
        task_specs = _normalize_task_specs(tasks)
        if input_scale is None:
            input_scale = 1.0

    assert segmenter is not None
    assert ignore_idx is not None
    assert input_scale is not None
    sae = getattr(segmenter, "sae", None)
    if sae is None:
        raise TypeError("segmenter must expose its SAE as segmenter.sae.")
    num_latents = getattr(sae, "num_latents", None)
    if num_latents is None:
        decoder = getattr(sae, "decoder", None)
        weight = getattr(decoder, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise TypeError("Cannot infer the number of SAE latents.")
        num_latents = weight.shape[1]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    segmenter.to(device)
    segmenter.eval()
    named_loaders = _named_validation_loaders(data_module)
    target_converter = None
    if lightning_module is not None:
        candidate = getattr(lightning_module, "_targets_to_per_pixel", None)
        if callable(candidate):
            target_converter = candidate

    collector = StreamingSAECollector(
        num_latents=int(num_latents),
        tasks=task_specs,
        ignore_idx=ignore_idx,
    )

    LOGGER.info(
        "Collecting SAE statistics from %d validation loaders.",
        len(named_loaders),
    )
    for dataset_name, _, batch in _iter_batches(
        named_loaders,
        ignore_idx=ignore_idx,
        target_converter=target_converter,
        max_batches_per_loader=max_batches_per_loader,
    ):
        if lightning_module is not None and window_inputs:
            batch = _window_batch(
                lightning_module,
                batch,
                task_specs=task_specs,
                ignore_idx=ignore_idx,
            )
        images = batch.images.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        latents, spatial_size = _forward_sae(
            segmenter,
            images,
            device=device,
            precision=precision,
            input_scale=input_scale,
        )
        collector.update(
            latents=latents,
            targets=targets,
            task_names=batch.task_names,
            dataset_name=dataset_name,
            spatial_size=spatial_size,
        )

    result = collector.finalize()
    result.decoder_norms = decoder_column_norms(sae)

    if heads is None and lightning_module is not None:
        decoder = getattr(segmenter, "decoder", None)
        if not isinstance(decoder, nn.Module):
            raise TypeError("segmenter.decoder must be an nn.Module.")
        input_dim = getattr(sae, "input_dim", None)
        if input_dim is None:
            decoder_weight = getattr(getattr(sae, "decoder", None), "weight", None)
            if not torch.is_tensor(decoder_weight) or decoder_weight.ndim != 2:
                raise TypeError("Cannot infer SAE input dimension.")
            input_dim = decoder_weight.shape[0]
        heads = resolve_task_heads(
            decoder,
            task_specs=task_specs,
            input_dim=int(input_dim),
        )
    heads = dict(heads or {})
    unknown_heads = set(heads) - set(task_specs)
    if unknown_heads:
        raise KeyError(
            f"Heads were provided for unknown tasks: {sorted(unknown_heads)}"
        )
    attributions = compute_task_attributions(
        result=result,
        task_specs=task_specs,
        sae=sae,
        heads=heads,
    )

    selected_latents = select_relevant_latents(
        result=result,
        attributions=attributions,
        per_class=top_latents_per_class,
        global_importance=global_top_latents,
        max_latents=max_selected_latents,
    )

    if top_examples_per_latent > 0 and selected_latents:
        LOGGER.info(
            "Collecting top examples for %d selected latents.",
            len(selected_latents),
        )
        example_collector = TopActivationCollector(
            latent_ids=selected_latents,
            tasks=task_specs,
            ignore_idx=ignore_idx,
            examples_per_latent=top_examples_per_latent,
            max_per_image=max_examples_per_image,
        )
        for dataset_name, _, batch in _iter_batches(
            named_loaders,
            ignore_idx=ignore_idx,
            target_converter=target_converter,
            max_batches_per_loader=max_batches_per_loader,
        ):
            if lightning_module is not None and window_inputs:
                batch = _window_batch(
                    lightning_module,
                    batch,
                    task_specs=task_specs,
                    ignore_idx=ignore_idx,
                )
            images = batch.images.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)
            latents, spatial_size = _forward_sae(
                segmenter,
                images,
                device=device,
                precision=precision,
                input_scale=input_scale,
            )
            example_collector.update(
                latents=latents,
                targets=targets,
                task_names=batch.task_names,
                image_ids=batch.image_ids,
                sample_ids=batch.sample_ids,
                dataset_name=dataset_name,
                spatial_size=spatial_size,
            )
        result.top_activations = example_collector.finalize()

    save_analysis_report(
        output_dir,
        result=result,
        task_specs=task_specs,
        attributions=attributions,
    )
    return result
