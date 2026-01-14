from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _load_label_map(prepared_dir: Path) -> Dict[str, int]:
    p = prepared_dir / "label_map.json"
    with open(p, "r") as f:
        lm = json.load(f)
    if not isinstance(lm, dict):
        raise ValueError("label_map.json must be a dict[str,int]")
    # normalize to int
    return {str(k): int(v) for k, v in lm.items()}


def _invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    inv: Dict[int, str] = {}
    for name, cid in label_map.items():
        inv[int(cid)] = str(name)
    return inv


def _color_for_class(cid: int) -> Tuple[int, int, int]:
    """
    Deterministic vivid-ish color without extra deps.
    """
    rng = random.Random(cid * 99991)
    return (rng.randint(40, 240), rng.randint(40, 240), rng.randint(40, 240))


def _load_image(prepared_dir: Path, relpath: str) -> Image.Image:
    p = prepared_dir / relpath
    im = Image.open(p).convert("RGB")
    return im


def _load_mask(prepared_dir: Path, relpath: str) -> np.ndarray:
    p = prepared_dir / relpath
    with Image.open(p) as im:
        # canonical in your pipeline: L
        return np.array(im, dtype=np.int32)


def _alpha_blend_mask(
    image: Image.Image,
    mask: np.ndarray,
    classes: List[int],
    alpha: float = 0.35,
) -> Image.Image:
    """
    Colorize mask for selected classes and alpha blend.
    """
    overlay = Image.new("RGB", image.size, (0, 0, 0))
    ov = np.array(overlay, dtype=np.uint8)

    for cid in classes:
        color = _color_for_class(cid)
        m = mask == cid
        ov[m, 0] = color[0]
        ov[m, 1] = color[1]
        ov[m, 2] = color[2]

    overlay = Image.fromarray(ov, mode="RGB")
    return Image.blend(image, overlay, alpha)


def _draw_bboxes_and_legend(
    image: Image.Image,
    rows: pd.DataFrame,
    id_to_name: Dict[int, str],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # try to load a default font; fall back gracefully
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw bboxes + labels
    for _, r in rows.iterrows():
        cid = int(r["dataset_class_id"])
        color = _color_for_class(cid)

        x0 = int(r["bbox_xmin"])
        y0 = int(r["bbox_ymin"])
        x1 = int(r["bbox_xmax"])
        y1 = int(r["bbox_ymax"])

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        name = id_to_name.get(cid, str(cid))
        text = f"{cid}:{name}"
        tx = max(0, min(w - 1, x0))
        ty = max(0, min(h - 12, y0 - 12))
        draw.text((tx, ty), text, fill=color, font=font)

    # Legend block (top-left)
    # list unique classes for this image
    uniq = sorted(set(int(x) for x in rows["dataset_class_id"].tolist()))
    pad = 6
    line_h = 14
    box_w = min(w, 520)
    box_h = pad * 2 + line_h * (len(uniq) + 1)

    # semi-transparent background for legend
    legend_bg = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
    image_rgba = image.convert("RGBA")
    image_rgba.paste(legend_bg, (0, 0), legend_bg)
    draw = ImageDraw.Draw(image_rgba)

    draw.text((pad, pad), "Legend", fill=(255, 255, 255, 255), font=font)

    y = pad + line_h
    for cid in uniq:
        color = _color_for_class(cid)
        name = id_to_name.get(cid, str(cid))
        draw.rectangle([pad, y + 3, pad + 10, y + 13], fill=color + (255,))
        draw.text((pad + 16, y), f"{cid}: {name}", fill=(255, 255, 255, 255), font=font)
        y += line_h

    return image_rgba.convert("RGB")


@click.command()
@click.option(
    "--prepared-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option(
    "--n",
    type=int,
    default=20,
    show_default=True,
    help="Number of random samples to export.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--overlay/--no-overlay",
    default=True,
    show_default=True,
    help="Overlay selected class masks.",
)
@click.option(
    "--alpha", type=float, default=0.35, show_default=True, help="Mask overlay alpha."
)
@click.option(
    "--max-classes",
    type=int,
    default=8,
    show_default=True,
    help="Max classes to overlay per sample.",
)
def main(
    prepared_dir: Path,
    out_dir: Path,
    n: int,
    seed: int,
    overlay: bool,
    alpha: float,
    max_classes: int,
):
    """
    Export visual sanity-check images:
    - original RGB
    - optional mask overlay (only classes present)
    - class bboxes with per-class colors
    - legend using label_map.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(prepared_dir / "metadata.csv")
    idx = pd.read_parquet(prepared_dir / "class_index.parquet")

    label_map = _load_label_map(prepared_dir)
    id_to_name = _invert_label_map(label_map)

    rng = random.Random(seed)
    sample_ids = meta["sample_id"].tolist()
    if len(sample_ids) == 0:
        raise ValueError("metadata.csv contains no samples")

    chosen = rng.sample(sample_ids, k=min(n, len(sample_ids)))

    for sample_id in chosen:
        mrow = meta[meta["sample_id"] == sample_id].iloc[0]
        image_rel = str(mrow["image_relpath"])
        mask_rel = str(mrow["mask_relpath"])

        rows = idx[idx["sample_id"] == sample_id]
        if len(rows) == 0:
            # still export image; maybe empty mask
            im = _load_image(prepared_dir, image_rel)
            im.save(out_dir / f"{sample_id}_NO_CLASS_INDEX.png")
            continue

        im = _load_image(prepared_dir, image_rel)

        classes = sorted(set(int(c) for c in rows["dataset_class_id"].tolist()))
        classes_for_overlay = classes[:max_classes]

        if overlay:
            mask = _load_mask(prepared_dir, mask_rel)
            im = _alpha_blend_mask(im, mask, classes_for_overlay, alpha=alpha)

        im = _draw_bboxes_and_legend(im, rows, id_to_name)
        im.save(out_dir / f"{sample_id}.png")

    click.echo(f"Wrote {len(chosen)} inspection images to: {out_dir}")


if __name__ == "__main__":
    main()
