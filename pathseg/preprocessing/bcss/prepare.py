from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathseg.preprocessing.utils import (
    check_mask,
    load_label_mask,
    mpp_to_nominal_magnification,
    stage_file,
    get_shape_from_image,
    translate_mask,
)

logger = logging.getLogger(__name__)

DST_LABEL_MAP = {
    "Background": 0,  # Rest
    "Invasive tumor": 1,
    "Tumor-associated stroma": 2,
    "In-situ tumor": 3,
    "Healthy gland": 4,
    "Necrosis not in-situ": 5,
    "Inflamed stroma": 6,
    "Ignore": 255,  # this class contains regions of several tissue compartments that are not specifically annotated in the other categories; examples are healthy stroma, erythrocytes, adipose tissue, skin, nipple, etc.
}
SRC_LABEL_MAP = {
    "Unannotated": 0,
    "Invasive tumor": 1,
    "Tumor-associated stroma": 2,
    "In-situ tumor": 3,
    "Healthy gland": 4,
    "Necrosis not in-situ": 5,
    "Inflamed stroma": 6,
    "Rest": 7,  # this class contains regions of several tissue compartments that are not specifically annotated in the other categories; examples are healthy stroma, erythrocytes, adipose tissue, skin, nipple, etc.
}

LABEL_TRANSLATION = {
    "Unannotated": "Ignore",
    "Rest": "Background",
}
MPP = 0.5  # microns per pixel at 20x magnification


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
    default="bcss",
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
    src_images_dir = raw_data_dir / "images"
    src_masks_dir = raw_data_dir / "masks"

    src_image_paths = [
        p
        for p in src_images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    src_image_paths = sorted(src_image_paths, key=lambda p: p.stem)
    sample_ids = [p.stem for p in src_image_paths]
    src_mask_paths = [p for p in src_masks_dir.iterdir() if p.stem in sample_ids]
    src_mask_paths = sorted(src_mask_paths, key=lambda p: p.stem)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks_semantic"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Write label map
    dst_label_map_file = output_dir / "label_map.json"
    with open(dst_label_map_file, "w") as f:
        json.dump(DST_LABEL_MAP, f, indent=2)

    with open(output_dir / "src_label_map.json", "w") as f:
        json.dump(SRC_LABEL_MAP, f, indent=2)

    valid_label_ids = set(SRC_LABEL_MAP.values())

    rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for sample_idx, sample_id in tqdm(
        enumerate(sample_ids), total=len(sample_ids), desc="Preparing samples"
    ):
        src_img = src_image_paths[sample_idx]
        src_msk = src_mask_paths[sample_idx]

        h, w = get_shape_from_image(src_img)
        h_mask, w_mask = get_shape_from_image(src_msk)
        if (h, w) != (h_mask, w_mask):
            logger.warning(
                f"Image and mask shape mismatch for sample_id={sample_id}: "
                f"image shape={(h, w)}, mask shape={(h_mask, w_mask)} skipping sample for now"
            )
            continue


        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        if not src_msk.exists():
            raise FileNotFoundError(f"Missing mask: {src_msk}")

        dst_img_name = f"{sample_id}.png"
        dst_msk_name = f"{sample_id}.png"

        dst_img = images_dir / dst_img_name
        dst_msk = masks_dir / dst_msk_name

        # Stage files
        stage_file(src_img, dst_img, mode=mode)
        stage_file(src_msk, dst_msk, mode=mode)

        # Validate/correct the staged mask in place (if needed)
        check_mask(dst_msk, sample_id=sample_id, valid_label_ids=valid_label_ids)
        translate_mask(
            dst_msk,
            dst_msk,
            src_label_map=SRC_LABEL_MAP,
            dst_label_map=DST_LABEL_MAP,
            label_translation=LABEL_TRANSLATION,
        )

        # Metadata (from CSV)
        mpp = MPP  # fixed for BCSS from TIGER, TODO: check more thoroughly
        magnification = mpp_to_nominal_magnification(mpp)
        group = "-".join(sample_id.split("-")[:3])  # TCGA Patient ID

        rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "image_relpath": str(Path("images") / dst_img_name),
                "mask_relpath": str(Path("masks_semantic") / dst_msk_name),
                "width": int(w),
                "height": int(h),
                "mpp_x": mpp,
                "mpp_y": mpp,
                "magnification": magnification,
                "group": group,
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
                    "image_relpath": str(Path("images") / dst_img_name),
                    "mask_relpath": str(Path("masks_semantic") / dst_msk_name),
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
