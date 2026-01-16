from __future__ import annotations

import ast
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pathseg.preprocessing.utils import (
    load_label_map_ids,
    load_label_mask,
    mpp_to_nominal_magnification,
    stage_file,
)

logger = logging.getLogger(__name__)

DESTINATION_LABEL_MAP = {
    "Background": 0,
    "Tumor epithelium": 1,
    "Reactive epithelium": 2,
    "Stroma": 3,
    "Inflammation": 4,
    "Alveolar tissue": 5,
    "Fatty tissue": 6,
    "Necrotic tissue": 7,
    "Erythrocytes": 8,
    "Bronchial epithelium": 9,
    "Mucus/Plasma/Fluids": 10,
    "Cartilage/Bone": 11,
    "Macrophages": 12,
    "Muscle": 13,
    "Liver": 14,
    "Keratinization": 15,
    "Ignore": 255,
}
# Just specify the one that needs to be changes called as
LABEL_TRANSLATION = {
    "Unannotated": "Ignore",
}


# IGNITE specifics (keep constants simple while you bootstrap the benchmark)
STAIN = "H&E"
TASK = "he_tissue_segmentation"

# Known dataset quirk: one mask has a single stray 255 pixel
KNOWN_GLITCH_SAMPLE_ID = "patient69_he_roi3"
KNOWN_GLITCH_INVALID_IDS = {255}


# ----------------------------
# Small utilities
# ----------------------------
def _parse_shape(x) -> Tuple[int, int]:
    """CSV 'shape' field is typically a string like '(H, W)'."""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, str):
        t = ast.literal_eval(x)
        return int(t[0]), int(t[1])
    raise ValueError(f"Unrecognized shape format: {x!r}")


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


def _compute_mpp_from_area(area_mm2: float, h: int, w: int) -> float:
    """
    Assumes pixels are square.
    area_mm2 is physical area of the ROI in mm^2.
    """
    if area_mm2 <= 0:
        raise ValueError(f"area_mm2 must be > 0, got {area_mm2}")
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid shape (h,w)=({h},{w})")

    pixel_area_mm2 = area_mm2 / (h * w)
    pixel_side_mm = math.sqrt(pixel_area_mm2)
    return float(pixel_side_mm * 1000.0)  # mm -> microns


def _check_and_fix_mask_inplace(
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

    labels = set(int(x) for x in np.unique(mask))
    invalid = labels - valid_label_ids
    if not invalid:
        return

    # Strict: only auto-fix the known glitch (keeps the benchmark transparent)
    if sample_id == KNOWN_GLITCH_SAMPLE_ID and invalid == KNOWN_GLITCH_INVALID_IDS:
        count = int((mask == 255).sum())
        logger.warning(
            f"Known IGNITE glitch: {mask_path} (sample_id={sample_id}) contains invalid label 255 "
            f"(count={count}). Replacing with Unannotated value == 0."
        )
        mask[mask == 255] = 0
        Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)

        # Re-check to be safe
        mask2 = load_label_mask(mask_path)
        labels2 = set(int(x) for x in np.unique(mask2)) - {0}
        invalid2 = labels2 - valid_label_ids
        if invalid2:
            raise RuntimeError(
                f"Mask still invalid after correction: {mask_path}, invalid={sorted(invalid2)}"
            )
        return

    raise ValueError(
        f"Mask {mask_path} (sample_id={sample_id}) contains IDs not defined in label_map.json: "
        f"{sorted(invalid)}"
    )


# ----------------------------
# CLI
# ----------------------------
@click.command()
@click.option(
    "--raw-data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to unzipped IGNITE raw data directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for the preprocessed dataset (dataset root).",
)
@click.option(
    "--dataset-id",
    type=str,
    default="ignite",
    show_default=True,
    help="Dataset identifier written into metadata.csv.",
)
@click.option(
    "--min-area",
    type=float,
    default=500.0,
    show_default=True,
    help="Filter class entries with area in [µm^2] below this threshold in class_index.",
)
@click.option(
    "--mode",
    type=click.Choice(["copy", "move"], case_sensitive=False),
    default="copy",
    show_default=True,
    help="File staging mode: 'copy' keeps raw files; 'move' saves space.",
)
def main(
    raw_data_dir: Path,
    output_dir: Path,
    dataset_id: str,
    min_area: float,
    mode: str,
) -> None:
    """
    Prepare the IGNITE dataset for PathSeg experiments.

    Expected raw layout:
      raw_data_dir/
        data_overview.csv
        he_label_map.json
        annotations/he/*.png
        images/he/*.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stain_folder = "he"

    # Load and filter overview
    df = pd.read_csv(raw_data_dir / "data_overview.csv")
    df = df[df["stain"].str.upper() == STAIN.upper()]
    df = df[df["task"] == TASK]

    # Output structure
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks_semantic"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Stage label map and load valid IDs from the staged copy (important for mode=move)
    src_label_map_file = raw_data_dir / f"{stain_folder}_label_map.json"
    if not src_label_map_file.exists():
        raise FileNotFoundError(f"Missing label map: {src_label_map_file}")
    dst_label_map_file = output_dir / "label_map.json"

    with open(src_label_map_file, "r") as f_src:
        src_label_map = json.load(f_src)

    stage_file(src_label_map_file, output_dir / "src_label_map.json", mode=mode)

    with open(dst_label_map_file, "w") as f_dst:
        json.dump(DESTINATION_LABEL_MAP, f_dst, indent=2)

    valid_label_ids = load_label_map_ids(src_label_map_file)

    rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(r["name"])

        img_name = Path(r["image_path"]).name
        msk_name = Path(r["annotation_path"]).name

        src_img = raw_data_dir / "images" / stain_folder / img_name
        src_msk = raw_data_dir / "annotations" / stain_folder / msk_name

        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        if not src_msk.exists():
            raise FileNotFoundError(f"Missing mask: {src_msk}")

        dst_img = images_dir / img_name
        dst_msk = masks_dir / msk_name

        # Stage files
        stage_file(src_img, dst_img, mode=mode)
        stage_file(src_msk, dst_msk, mode=mode)

        # Validate/correct the staged mask in place (if needed)
        _check_and_fix_mask_inplace(
            dst_msk, sample_id=sample_id, valid_label_ids=valid_label_ids
        )

        translate_mask(
            dst_msk,
            dst_msk,
            src_label_map=src_label_map,
            dst_label_map=DESTINATION_LABEL_MAP,
            label_translation=LABEL_TRANSLATION,
            mode="inplace",
        )

        # Metadata (from CSV)
        h, w = _parse_shape(r["shape"])
        mpp = _compute_mpp_from_area(float(r["area_mm2"]), h=h, w=w)
        magnification = mpp_to_nominal_magnification(mpp)
        group = r.get("patient_id", None)

        rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "group": group,
                "image_relpath": str(Path("images") / img_name),
                "mask_relpath": str(Path("masks_semantic") / msk_name),
                "width": int(w),
                "height": int(h),
                "mpp_x": mpp,
                "mpp_y": mpp,
                "magnification": magnification if magnification is not None else "",
                # Keep these if present (else empty)
                "split": str(r.get("split", "")),
                "validation_fold": str(r.get("validation_fold", "")),
                "patient_id": r.get("patient_id", ""),
                "roi_id": r.get("roi_id", ""),
                "source": r.get("source", ""),
                "scanner": r.get("scanner", ""),
                "histological_subtype": r.get("histological_subtype", ""),
            }
        )

        # Class -> bbox index (from corrected staged mask)
        mask = load_label_mask(dst_msk)
        mask_ids = set(int(x) for x in np.unique(mask)) - {0}

        for cid in sorted(mask_ids):
            ys, xs = (mask == cid).nonzero()
            if xs.size == 0:
                continue

            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            class_area = int((mask == cid).sum()) * (mpp * mpp)  # µm^2

            if class_area < min_area:
                continue

            class_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "dataset_class_id": int(cid),
                    "group": group,
                    "area_um2": class_area,
                    "image_relpath": str(Path("images") / img_name),
                    "mask_relpath": str(Path("masks_semantic") / msk_name),
                    "mpp": mpp,
                    "bbox_xmin": xmin,
                    "bbox_ymin": ymin,
                    "bbox_xmax": xmax,
                    "bbox_ymax": ymax,
                    "width": int(mask.shape[1]),
                    "height": int(mask.shape[0]),
                }
            )

    # Write dataset manifests
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    try:
        metadata.to_parquet(output_dir / "metadata.parquet", index=False)
    except Exception:
        pass

    class_index = pd.DataFrame(class_rows)
    class_index.to_parquet(output_dir / "class_index.parquet", index=False)

    # Convenience mapping: class -> sample_ids
    class_to_samples: Dict[str, List[str]] = (
        class_index.groupby("dataset_class_id")["sample_id"]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
    )
    with open(output_dir / "class_to_samples.json", "w") as f:
        json.dump(class_to_samples, f, indent=2)

    click.echo(f"Prepared {len(metadata)} samples into: {output_dir}")
    click.echo(f"Wrote: {output_dir / 'metadata.csv'}")
    click.echo(f"Wrote: {output_dir / 'class_index.parquet'}")
    click.echo(f"Wrote: {output_dir / 'class_to_samples.json'}")


if __name__ == "__main__":
    main()
