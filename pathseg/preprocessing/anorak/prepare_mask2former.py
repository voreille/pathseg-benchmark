#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import click
import numpy as np
import torch
from PIL import Image


# Connected components: prefer scipy, fallback to opencv, otherwise error
def _connected_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Args:
        binary: (H, W) bool/uint8

    Returns:
        labels: (H, W) int32, where 0 is background and 1..num are components
        num: number of components
    """
    binary_u8 = binary.astype(np.uint8) * 255

    # Try scipy first
    try:
        import scipy.ndimage as ndi  # type: ignore

        structure = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )  # 4-connectivity
        labels, num = ndi.label(binary_u8 > 0, structure=structure)
        return labels.astype(np.int32), int(num)
    except Exception:
        pass

    # Try opencv
    try:
        import cv2  # type: ignore

        num, labels = cv2.connectedComponents(binary_u8, connectivity=4)
        # opencv returns num labels including background
        return labels.astype(np.int32), int(num - 1)
    except Exception:
        raise RuntimeError(
            "No connected-components backend found. Install scipy or opencv-python.\n"
            "  pip install scipy\n"
            "or\n"
            "  pip install opencv-python"
        )


def _load_mapping(path: Optional[Path]) -> Optional[Dict[int, int]]:
    if path is None:
        return None
    mapping = json.loads(path.read_text())
    # allow keys as str in json
    return {int(k): int(v) for k, v in mapping.items()}


def _apply_mapping(
    mask: np.ndarray, mapping: Dict[int, int], ignore_idx: int
) -> np.ndarray:
    # map every pixel value via LUT (fast)
    max_val = int(mask.max()) if mask.size else 0
    lut_size = max(max_val, max(mapping.keys(), default=0), ignore_idx) + 1
    lut = np.arange(lut_size, dtype=np.int32)
    for k, v in mapping.items():
        if k < lut_size:
            lut[k] = v
    # extend LUT if needed
    if mask.max() >= lut_size:
        # rare; grow LUT
        new_size = int(mask.max()) + 1
        lut2 = np.arange(new_size, dtype=np.int32)
        lut2[:lut_size] = lut
        lut = lut2
        for k, v in mapping.items():
            lut[k] = v
    out = lut[mask.astype(np.int32)]
    # preserve ignore_idx pixels
    out[mask == ignore_idx] = ignore_idx
    return out.astype(mask.dtype)


def semantic_to_instances(
    sem: np.ndarray,
    num_classes: int,
    ignore_idx: int,
    return_background: bool,
    min_area: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert semantic mask (H,W) to instance masks + labels.

    Returns:
        masks_u8: (N,H,W) uint8 in {0,1}
        labels_i64: (N,) int64 (class ids)
    """
    H, W = sem.shape
    masks: List[np.ndarray] = []
    labels: List[int] = []

    class_ids = np.unique(sem)
    for class_id in class_ids:
        class_id = int(class_id)
        if class_id == ignore_idx:
            continue
        if (not return_background) and class_id == 0:
            continue
        if class_id < 0 or class_id >= num_classes:
            # skip unexpected labels (except ignore)
            continue

        binary = sem == class_id
        if not np.any(binary):
            continue

        cc_labels, cc_num = _connected_components(binary)
        if cc_num == 0:
            continue

        for k in range(1, cc_num + 1):
            comp = cc_labels == k
            area = int(comp.sum())
            if area < min_area:
                continue
            masks.append(comp.astype(np.uint8))
            labels.append(class_id)

    if len(masks) == 0:
        return np.zeros((0, H, W), dtype=np.uint8), np.zeros((0,), dtype=np.int64)

    masks_u8 = np.stack(masks, axis=0)  # (N,H,W)
    labels_i64 = np.asarray(labels, dtype=np.int64)
    return masks_u8, labels_i64


def _infer_image_ids(images_dir: Path, masks_dir: Path) -> List[str]:
    # require a semantic mask png with same stem
    ids = []
    for p in sorted(masks_dir.glob("*.png")):
        stem = p.stem
        # image can have any extension
        if any(images_dir.glob(f"{stem}.*")):
            ids.append(stem)
    return ids


@click.command()
@click.argument(
    "root_dir", type=click.Path(path_type=Path, exists=True, file_okay=False)
)
@click.option(
    "--images-subdir",
    default="images",
    show_default=True,
    help="Subdir under root containing images.",
)
@click.option(
    "--masks-subdir",
    default="masks_semantic",
    show_default=True,
    help="Subdir under root containing semantic masks.",
)
@click.option(
    "--out-subdir",
    default="targets_instance",
    show_default=True,
    help="Output subdir under root.",
)
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["npz", "pt"]),
    default="pt",
    show_default=True,
)
@click.option(
    "--num-classes",
    type=int,
    default=7,
    required=False,
    help="Number of classes in semantic mask (labels 0..num_classes-1).",
)
@click.option(
    "--ignore-idx",
    type=int,
    default=255,
    show_default=True,
    help="Ignore label value in semantic masks.",
)
@click.option(
    "--return-background/--no-return-background",
    default=True,
    show_default=True,
    help="Whether to keep class 0 instances.",
)
@click.option(
    "--min-area",
    type=int,
    default=1,
    show_default=True,
    help="Minimum pixel area for an instance component.",
)
@click.option(
    "--class-mapping",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help='Optional JSON mapping, e.g. {"3": 1, "4": 2} to remap semantic labels before splitting.',
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    show_default=True,
    help="Overwrite existing outputs.",
)
def main(
    root_dir: Path,
    images_subdir: str,
    masks_subdir: str,
    out_subdir: str,
    out_format: str,
    num_classes: int,
    ignore_idx: int,
    return_background: bool,
    min_area: int,
    class_mapping: Optional[Path],
    overwrite: bool,
):
    """
    Convert semantic masks in ROOT_DIR/masks_semantic/*.png into instance targets for Mask2Former.

    Expects:
      ROOT_DIR/images/<id>.<ext>
      ROOT_DIR/masks_semantic/<id>.png

    Produces:
      ROOT_DIR/targets_instance/<id>.npz  (masks uint8 [N,H,W], labels int64 [N])
    or
      ROOT_DIR/targets_instance/<id>.pt   (dict with masks BoolTensor, labels LongTensor)
    """
    images_dir = root_dir / images_subdir
    masks_dir = root_dir / masks_subdir
    out_dir = root_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise click.ClickException(f"Images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise click.ClickException(f"Masks dir not found: {masks_dir}")

    mapping = _load_mapping(class_mapping)

    image_ids = _infer_image_ids(images_dir, masks_dir)
    if len(image_ids) == 0:
        raise click.ClickException(
            f"No matching pairs found in {images_dir} and {masks_dir}"
        )

    click.echo(f"Found {len(image_ids)} image/mask pairs.")
    click.echo(f"Writing to: {out_dir}  (format={out_format})")

    n_written = 0
    for image_id in image_ids:
        mask_path = masks_dir / f"{image_id}.png"
        out_path = out_dir / f"{image_id}.{out_format}"

        if out_path.exists() and not overwrite:
            continue

        sem = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        if mapping is not None:
            sem = _apply_mapping(sem, mapping, ignore_idx=ignore_idx)

        masks_u8, labels_i64 = semantic_to_instances(
            sem=sem,
            num_classes=num_classes,
            ignore_idx=ignore_idx,
            return_background=return_background,
            min_area=min_area,
        )

        if out_format == "npz":
            np.savez_compressed(out_path, masks=masks_u8, labels=labels_i64)
        else:
            # store torch-friendly dict
            d = {
                "masks": torch.from_numpy(masks_u8.astype(np.bool_)),
                "labels": torch.from_numpy(labels_i64),
            }
            torch.save(d, out_path)

        n_written += 1

    click.echo(
        f"Done. Wrote {n_written} files (skipped {len(image_ids) - n_written} existing)."
    )


if __name__ == "__main__":
    main()
