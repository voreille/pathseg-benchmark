from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


def load_label_map_ids(label_map_path: Path) -> set[int]:
    """
    Label map is expected to be dict[str, int] (class_name -> class_id).
    Returns the set of valid IDs.
    """
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict):
        raise ValueError(f"Label map {label_map_path} must be a dict[str, int]")

    ids = set(label_map.values())
    if not all(isinstance(x, int) for x in ids):
        raise ValueError(f"Label map {label_map_path} contains non-integer IDs")

    return ids


def load_label_mask(mask_path: Path) -> np.ndarray:
    """
    Load a semantic segmentation mask as integer class IDs.

    Important:
    - Do NOT convert 'P' to 'L' (palette -> grayscale destroys IDs).
    - Fail fast on visualization-style masks (values near 255).
    """
    with Image.open(mask_path) as im:
        if im.mode == "P":
            mask = np.array(im, dtype=np.int32)  # indices = class ids
        elif im.mode == "L":
            mask = np.array(im, dtype=np.int32)
        else:
            raise ValueError(
                f"Unsupported mask mode {im.mode} for {mask_path}. "
                "Expected 'P' or 'L'. If RGB/RGBA, need a color->id mapping."
            )

    u = np.unique(mask)
    # IGNITE visualization masks often have values in [240..255] (wrong artifact)
    if u.size > 0 and u.min() >= 240:
        raise RuntimeError(
            f"Mask {mask_path} looks like a visualization mask (min={u.min()}, max={u.max()}, "
            f"unique_head={u[:10]}). Use the label-map annotations (integer IDs)."
        )

    return mask


def check_mask(
    mask_path: Path,
    *,
    sample_id: str,
    valid_label_ids: set[int],
) -> None:
    """
    Validate mask labels against `valid_label_ids`.

    Policy:
    - If invalid IDs are found:
        * If this is the known IGNITE glitch (patient69_he_roi3 with a stray 255 pixel),
          fix in place by mapping invalid IDs -> 0.
        * Otherwise raise.

    Correction is applied to the staged (copied/moved) file.
    """
    mask = load_label_mask(mask_path)

    labels = set(int(x) for x in np.unique(mask)) - {0}
    invalid = labels - valid_label_ids
    if not invalid:
        return

    raise ValueError(
        f"Mask {mask_path} (sample_id={sample_id}) contains IDs not defined in label_map.json: "
        f"{sorted(invalid)}"
    )


def stage_file(src: Path, dst: Path, *, mode: str = "copy") -> None:
    """
    Stage a file into the preprocessed dataset.

    mode:
      - "copy": keep raw data intact
      - "move": save space when using an automated download+prepare pipeline
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_suffix = src.suffix.lower()
    dst_suffix = dst.suffix.lower()

    convert_to_png = src_suffix in {".jpg", ".jpeg"} and dst_suffix == ".png"

    if convert_to_png:
        tmp_dst = dst.with_suffix(src.suffix)
    else:
        tmp_dst = dst

    if mode == "copy":
        shutil.copy2(src, tmp_dst)
    elif mode == "move":
        shutil.move(src, tmp_dst)
    else:
        raise ValueError(f"Unknown mode={mode!r}, expected 'copy' or 'move'")

    if convert_to_png:
        with Image.open(tmp_dst) as im:
            im = im.convert("RGB")
            im.save(dst)
        tmp_dst.unlink()


def mpp_to_nominal_magnification(
    mpp: float,
    *,
    ref_mag: float = 20.0,
    ref_mpp: float = 0.5,
    allowed_magnifications: Tuple[int, ...] = (40, 20, 10, 5),
    max_relative_error: float = 0.35,
) -> Optional[int]:
    """
    Infer nominal magnification from MPP using proportionality:
        magnification ≈ ref_mag * ref_mpp / mpp
    Then snap to closest allowed magnification.
    """
    if mpp <= 0:
        return None

    estimated_mag = ref_mag * ref_mpp / mpp
    best_mag = -1
    best_rel_error = float("inf")

    for mag in allowed_magnifications:
        rel_error = abs(estimated_mag - mag) / mag
        if rel_error < best_rel_error:
            best_rel_error = rel_error
            best_mag = mag

    return int(best_mag) if best_rel_error <= max_relative_error else None


def get_shape_from_image(image_path: Path) -> Tuple[int, int]:
    """
    Get image shape (height, width) from image file.
    """
    with Image.open(image_path) as im:
        w, h = im.size
    return h, w


def _build_lut(
    *,
    src_label_map: Dict[str, int],
    dst_label_map: Dict[str, int],
    label_translation: Dict[str, str],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build a 256-entry LUT mapping old pixel values -> new pixel values.
    Returns (lut, mapping_dict) where mapping_dict is {old_id: new_id}.
    """
    missing = set(label_translation.values()) - set(dst_label_map.keys())
    if missing:
        raise ValueError(
            f"label_translation targets not in dst_label_map: {sorted(missing)}"
        )

    mapping: Dict[int, int] = {}
    for name, src_id in src_label_map.items():
        dst_name = label_translation.get(name, name)
        if dst_name not in dst_label_map:
            raise ValueError(
                f"Source label {name!r} maps to {dst_name!r}, but {dst_name!r} not in dst_label_map"
            )
        mapping[src_id] = dst_label_map[dst_name]

    lut = np.arange(256, dtype=np.uint8)
    for old_id, new_id in mapping.items():
        if not (0 <= old_id <= 255 and 0 <= new_id <= 255):
            raise ValueError(f"Label ids must be in [0,255], got {old_id}->{new_id}")
        lut[old_id] = np.uint8(new_id)

    return lut, mapping


def translate_mask(
    src: Path,
    dst: Path,
    *,
    src_label_map: Dict[str, int],
    dst_label_map: Dict[str, int],
    label_translation: Dict[str, str],
    mode: str = "inplace",  # "inplace" or "copy"
) -> None:
    """
    Translate an 8-bit single-channel mask by remapping pixel values.

    mode:
      - "copy": keep src, write to dst (src != dst)
      - "inplace": overwrite src (src == dst), safely via a temp file + atomic replace
    """
    src = src.resolve()
    dst = dst.resolve()

    if mode == "copy" and src == dst:
        raise ValueError("In 'copy' mode, src and dst must be different paths")
    if mode == "inplace" and src != dst:
        raise ValueError("In 'inplace' mode, src and dst must be the same path")
    if mode not in {"copy", "inplace"}:
        raise ValueError(f"Unknown mode={mode!r}, expected 'copy' or 'inplace'")

    dst.parent.mkdir(parents=True, exist_ok=True)

    lut, mapping = _build_lut(
        src_label_map=src_label_map,
        dst_label_map=dst_label_map,
        label_translation=label_translation,
    )

    # Read mask
    with Image.open(src) as im:
        im = im.convert("L")  # ensure 8-bit single channel
        arr = np.array(im, dtype=np.uint8)

    # Sanity check: only expected IDs appear
    present = set(np.unique(arr).tolist())
    expected = set(mapping.keys())
    unknown = present - expected
    if unknown:
        raise ValueError(
            f"{src} contains pixel values not covered by src_label_map: {sorted(unknown)}"
        )

    # Apply translation
    out = lut[arr]
    out_im = Image.fromarray(out, mode="L")

    if mode == "copy":
        out_im.save(dst)
        return

    # mode == "inplace": write to temp file then atomically replace
    # Keep temp in same directory so os.replace() is atomic on the same filesystem.
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        out_im.save(tmp, format="PNG")
        os.replace(tmp, dst)  # atomic replace
    finally:
        # If something failed before os.replace, clean up temp if it exists
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
