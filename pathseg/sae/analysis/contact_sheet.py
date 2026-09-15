from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from pathseg.sae.analysis.batches import unpack_multitask_batch
from pathseg.sae.analysis.cli import build_analysis_objects
from pathseg.sae.analysis.latent_selection import (
    LatentSpec,
    load_top_activation_records,
    resolve_latent_specs,
    select_top_activation_records,
)
from pathseg.sae.analysis.runner import (
    _forward_sae,
    _named_validation_loaders,
    _task_specs_from_lightning_module,
    _window_batch,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ContactExample:
    record: dict[str, Any]
    image: torch.Tensor
    target: torch.Tensor
    activation_map: torch.Tensor
    recomputed_activation: float


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "latent"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu()
    if image.ndim != 3 or image.shape[0] not in {1, 3, 4}:
        raise ValueError(
            "Contact-sheet images must have shape [C,H,W] with 1, 3, or 4 "
            f"channels, got {tuple(image.shape)}."
        )
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] == 4:
        image = image[:3]
    if float(image.max().item()) <= 1.5:
        image = image * 255.0
    image = image.clamp(0, 255).byte().permute(1, 2, 0)
    return image.numpy()


def _upsample_map(
    activation_map: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    upsampled = F.interpolate(
        activation_map[None, None].float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return upsampled.detach().cpu().numpy()


def _activation_overlay(
    rgb: np.ndarray,
    activation_map: torch.Tensor,
    *,
    vmax: float,
) -> Image.Image:
    heatmap = _upsample_map(activation_map, rgb.shape[:2])
    normalized = np.clip(heatmap / max(vmax, 1e-8), 0.0, 1.0)
    color = colormaps["magma"](normalized)[..., :3] * 255.0
    alpha = (0.15 + 0.60 * normalized)[..., None]
    overlay = rgb * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _target_overlay(
    rgb: np.ndarray,
    target: torch.Tensor,
    *,
    class_index: int | None,
) -> Image.Image:
    if class_index is None:
        return Image.fromarray(rgb)
    mask = target.detach().cpu().numpy() == int(class_index)
    overlay = rgb.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.55 + np.array(
        [0.0, 255.0, 255.0], dtype=np.float32
    ) * 0.45

    if mask.any():
        interior = mask.copy()
        interior[1:] &= mask[:-1]
        interior[:-1] &= mask[1:]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        boundary = mask & ~interior
        overlay[boundary] = np.array([255.0, 255.0, 0.0])
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _token_center(
    record: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    height, width = image_size
    center_y = (float(record["token_y"]) + 0.5) * height / int(
        record["grid_height"]
    )
    center_x = (float(record["token_x"]) + 0.5) * width / int(
        record["grid_width"]
    )
    return center_y, center_x


def _mark_token(image: Image.Image, center: tuple[float, float]) -> None:
    center_y, center_x = center
    draw = ImageDraw.Draw(image)
    radius = max(4, round(min(image.size) * 0.012))
    x = round(center_x)
    y = round(center_y)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(0, 255, 255),
        width=max(2, radius // 3),
    )


def _context_box(
    record: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    context_tokens: int,
) -> tuple[int, int, int, int]:
    height, width = image_size
    center_y, center_x = _token_center(record, image_size=image_size)
    box_height = max(
        1,
        round(context_tokens * height / int(record["grid_height"])),
    )
    box_width = max(
        1,
        round(context_tokens * width / int(record["grid_width"])),
    )
    top = round(center_y - box_height / 2)
    left = round(center_x - box_width / 2)
    top = min(max(top, 0), max(height - box_height, 0))
    left = min(max(left, 0), max(width - box_width, 0))
    return left, top, min(left + box_width, width), min(top + box_height, height)


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def _example_triptych(
    example: ContactExample,
    *,
    panel_size: int,
    context_tokens: int,
    vmax: float,
    expected_class: int | None,
) -> Image.Image:
    rgb = _tensor_to_rgb(example.image)
    image_size = rgb.shape[:2]
    center = _token_center(example.record, image_size=image_size)

    original = Image.fromarray(rgb.copy())
    _mark_token(original, center)
    heatmap = _activation_overlay(
        rgb,
        example.activation_map,
        vmax=vmax,
    )
    _mark_token(heatmap, center)

    target_class = (
        expected_class
        if expected_class is not None
        else example.record.get("target_class")
    )
    target_overlay = _target_overlay(
        rgb,
        example.target,
        class_index=target_class,
    )
    context_box = _context_box(
        example.record,
        image_size=image_size,
        context_tokens=context_tokens,
    )
    zoom = target_overlay.crop(context_box)

    panels = [
        _resize_square(original, panel_size),
        _resize_square(heatmap, panel_size),
        _resize_square(zoom, panel_size),
    ]
    header_height = 68
    canvas = Image.new(
        "RGB",
        (panel_size * 3, panel_size + header_height),
        color=(20, 20, 20),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel_size, header_height))

    record = example.record
    source = (
        f"{record['dataset_name']} | sample={record['sample_id']}"
    )
    values = (
        f"target={record.get('target_class')} | "
        f"token=({record['token_y']},{record['token_x']}) | "
        f"stored={float(record['activation']):.3f} | "
        f"recomputed={example.recomputed_activation:.3f}"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), source[:100], fill=(245, 245, 245), font=_font(15))
    draw.text((8, 25), values[:110], fill=(215, 215, 215), font=_font(13))
    draw.text(
        (8, 46),
        "original + token | activation heatmap | local target overlay",
        fill=(180, 180, 180),
        font=_font(13),
    )
    return canvas


def render_contact_sheet(
    spec: LatentSpec,
    examples: Sequence[ContactExample],
    *,
    output_path: str | Path,
    panel_size: int = 224,
    context_tokens: int = 9,
    examples_per_row: int = 2,
) -> Path:
    if not examples:
        raise ValueError(f"Latent {spec.latent_id} has no contact examples.")
    if panel_size <= 0 or context_tokens <= 0 or examples_per_row <= 0:
        raise ValueError("Panel, context, and row sizes must be positive.")
    if context_tokens % 2 == 0:
        raise ValueError("context_tokens must be odd.")

    vmax = max(
        float(example.activation_map.max().item())
        for example in examples
    )
    tiles = [
        _example_triptych(
            example,
            panel_size=panel_size,
            context_tokens=context_tokens,
            vmax=vmax,
            expected_class=spec.class_index,
        )
        for example in examples
    ]
    tile_width, tile_height = tiles[0].size
    rows = math.ceil(len(tiles) / examples_per_row)
    title_height = 60
    sheet = Image.new(
        "RGB",
        (tile_width * examples_per_row, title_height + tile_height * rows),
        color=(12, 12, 12),
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, examples_per_row)
        sheet.paste(
            tile,
            (column * tile_width, title_height + row * tile_height),
        )

    title_parts = [f"latent {spec.latent_id}"]
    if spec.task is not None:
        title_parts.append(f"task={spec.task}")
    if spec.class_index is not None:
        title_parts.append(f"class={spec.class_index}")
    if spec.label is not None:
        title_parts.append(spec.label)
    title_parts.append(f"shared activation scale: 0 to {vmax:.3f}")
    ImageDraw.Draw(sheet).text(
        (12, 15),
        " | ".join(title_parts),
        fill=(255, 255, 255),
        font=_font(20),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


@torch.no_grad()
def collect_contact_examples(
    *,
    lightning_module,
    data_module,
    selected_records: Mapping[
        tuple[int, str | None, int | None],
        Sequence[Mapping[str, Any]],
    ],
    device: str | torch.device | None = None,
    precision: str = "16-mixed",
    window_inputs: bool = True,
) -> dict[tuple[int, str | None, int | None], list[ContactExample]]:
    segmenter = lightning_module.network
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    segmenter.to(device)
    segmenter.eval()

    task_specs = _task_specs_from_lightning_module(lightning_module)
    ignore_idx = int(lightning_module.ignore_idx)
    target_converter = getattr(lightning_module, "_targets_to_per_pixel", None)
    if not callable(target_converter):
        target_converter = None

    requests_by_sample: dict[
        tuple[str, str],
        list[tuple[tuple[int, str | None, int | None], int, dict[str, Any]]],
    ] = defaultdict(list)
    pending: set[tuple[tuple[int, str | None, int | None], int]] = set()
    for spec_key, records in selected_records.items():
        for record_index, raw_record in enumerate(records):
            record = dict(raw_record)
            request_id = spec_key, record_index
            sample_key = str(record["dataset_name"]), str(record["sample_id"])
            requests_by_sample[sample_key].append(
                (spec_key, record_index, record)
            )
            pending.add(request_id)

    collected: dict[
        tuple[int, str | None, int | None],
        dict[int, ContactExample],
    ] = {key: {} for key in selected_records}

    for dataset_name, loader in _named_validation_loaders(data_module):
        if not any(key[0] == dataset_name for key in requests_by_sample):
            continue
        for raw_batch in loader:
            batch = unpack_multitask_batch(
                raw_batch,
                ignore_idx=ignore_idx,
                target_converter=target_converter,
            )
            if window_inputs:
                batch = _window_batch(
                    lightning_module,
                    batch,
                    task_specs=task_specs,
                    ignore_idx=ignore_idx,
                )

            matching_indices = [
                index
                for index, sample_id in enumerate(batch.sample_ids)
                if (dataset_name, sample_id) in requests_by_sample
            ]
            if not matching_indices:
                continue

            index_tensor = torch.as_tensor(matching_indices, dtype=torch.long)
            selected_images = batch.images.index_select(0, index_tensor)
            selected_targets = batch.targets.index_select(0, index_tensor)
            latents, spatial_size = _forward_sae(
                segmenter,
                selected_images.to(device, non_blocking=True),
                device=device,
                precision=precision,
                input_scale=1.0 / 255.0,
            )
            grid_height, grid_width = spatial_size
            latent_maps = latents.transpose(1, 2).reshape(
                latents.shape[0],
                latents.shape[-1],
                grid_height,
                grid_width,
            )

            for local_index, batch_index in enumerate(matching_indices):
                sample_key = dataset_name, batch.sample_ids[batch_index]
                for spec_key, record_index, record in requests_by_sample[sample_key]:
                    request_id = spec_key, record_index
                    if request_id not in pending:
                        continue
                    if (
                        int(record["grid_height"]) != grid_height
                        or int(record["grid_width"]) != grid_width
                    ):
                        raise ValueError(
                            f"Stored grid for latent {spec_key[0]} is "
                            f"{record['grid_height']}x{record['grid_width']}, but "
                            f"the replayed model produced {grid_height}x{grid_width}."
                        )
                    latent_id = spec_key[0]
                    if not 0 <= latent_id < latent_maps.shape[1]:
                        raise IndexError(f"Latent ID {latent_id} is out of bounds.")
                    token_y = int(record["token_y"])
                    token_x = int(record["token_x"])
                    activation_map = latent_maps[
                        local_index,
                        latent_id,
                    ].detach().float().cpu()
                    recomputed = float(activation_map[token_y, token_x].item())
                    collected[spec_key][record_index] = ContactExample(
                        record=record,
                        image=selected_images[local_index].detach().cpu(),
                        target=selected_targets[local_index].detach().cpu(),
                        activation_map=activation_map,
                        recomputed_activation=recomputed,
                    )
                    pending.remove(request_id)

            if not pending:
                break
        if not pending:
            break

    if pending:
        missing = []
        for spec_key, record_index in sorted(pending, key=str):
            record = selected_records[spec_key][record_index]
            missing.append(
                f"latent={spec_key[0]} dataset={record['dataset_name']} "
                f"sample={record['sample_id']}"
            )
        raise RuntimeError(
            "Could not replay the following top activations: "
            + "; ".join(missing[:20])
        )
    return {
        spec_key: [
            collected[spec_key][index]
            for index in range(len(selected_records[spec_key]))
        ]
        for spec_key in selected_records
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create latent contact sheets from top-activation records."
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="Directory containing top_activations.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to ANALYSIS_DIR/contact_sheets.",
    )
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
        help="Repeat for each requested latent.",
    )
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--panel-size", type=int, default=224)
    parser.add_argument("--context-tokens", type=int, default=9)
    parser.add_argument("--examples-per-row", type=int, default=2)
    parser.add_argument(
        "--include-class-mismatches",
        action="store_true",
        help="Keep other target classes when a manifest class is specified.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=("32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"),
    )
    parser.add_argument("--no-window-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    specs = resolve_latent_specs(
        manifest_path=args.manifest,
        values=args.latent,
    )
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    records = load_top_activation_records(
        analysis_dir / "top_activations.jsonl"
    )
    selected_records = {
        spec.key: select_top_activation_records(
            records,
            spec,
            max_examples=args.examples,
            include_class_mismatches=args.include_class_mismatches,
        )
        for spec in specs
    }
    empty = [spec for spec in specs if not selected_records[spec.key]]
    if empty:
        values = ", ".join(str(spec.key) for spec in empty)
        raise ValueError(f"No matching top activations for: {values}")

    module, data_module = build_analysis_objects(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    examples = collect_contact_examples(
        lightning_module=module,
        data_module=data_module,
        selected_records=selected_records,
        device=args.device,
        precision=args.precision,
        window_inputs=not args.no_window_inputs,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir / "contact_sheets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output: list[dict[str, Any]] = []
    for spec in specs:
        name_parts = [f"latent_{spec.latent_id}"]
        if spec.task is not None:
            name_parts.append(spec.task)
        if spec.class_index is not None:
            name_parts.append(f"class_{spec.class_index}")
        output_path = output_dir / f"{_slug('_'.join(name_parts))}.png"
        render_contact_sheet(
            spec,
            examples[spec.key],
            output_path=output_path,
            panel_size=args.panel_size,
            context_tokens=args.context_tokens,
            examples_per_row=args.examples_per_row,
        )
        manifest_output.append(
            {
                **spec.as_dict(),
                "contact_sheet": str(output_path),
                "examples": [
                    {
                        **example.record,
                        "recomputed_activation": example.recomputed_activation,
                    }
                    for example in examples[spec.key]
                ],
            }
        )
        LOGGER.info("Saved %s", output_path)

    with (output_dir / "contact_sheets.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump({"latents": manifest_output}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
