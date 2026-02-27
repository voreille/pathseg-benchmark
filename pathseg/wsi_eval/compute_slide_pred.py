"""
Option 1 (geometry-correct): stitch directly onto the GT/mask grid.

- Tiles are defined in level-0 coordinates (x0,y0).
- For each tile, map its level-0 bounding box to GT pixels using per-WSI factors.
- Resize tile logits to exactly that GT-rectangle size (bilinear).
- Accumulate logits with a weight map, then divide at the end.

This avoids “drift” because the placement rectangle is defined by mapped endpoints.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from histoseg_plugin.storage.factory import build_tiling_store_from_dir
from pathseg.utils.load_experiment import load_experiment


# -------------------------
# Dataset + collate
# -------------------------
class TileDataset(Dataset):
    """
    Reads RGB tiles from an OpenSlide at coords.

    coords must be level-0 pixel coords (x, y) of the tile top-left.
    attrs must include:
      - patch_size: tile size in pixels at patch_level
      - patch_level: openslide level used for read_region
    """

    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords,
        attrs: Dict[str, Any],
        transforms=None,
    ):
        self.wsi = wsi
        self.coords = coords
        self.tile_size_lvl = int(attrs["patch_size"])  # pixels at tile_level
        self.tile_level = int(attrs["patch_level"])  # openslide level index
        if transforms is None:
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, y0 = self.coords[idx]
        tile = self.wsi.read_region(
            (int(x0), int(y0)),
            self.tile_level,
            (self.tile_size_lvl, self.tile_size_lvl),
        ).convert("RGB")

        if self.transforms is not None:
            tile = self.transforms(tile)

        return tile, torch.tensor([x0, y0], dtype=torch.int64)



# -------------------------
# Metadata helpers
# -------------------------
def load_metadata_map(csv_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    if "wsi_id" not in df.columns:
        raise ValueError("metadata CSV must contain a 'wsi_id' column")
    df = df.set_index("wsi_id")
    return df.to_dict(orient="index")


def get_mask_grid_and_factors(
    meta_row: Dict[str, Any],
    level0_w: int,
    level0_h: int,
) -> Tuple[int, int, float, float]:
    """
    Returns (mask_h, mask_w, fx, fy)
    fx, fy = level-0 pixels per 1 mask pixel (can be non-integer)
    """
    mask_w = int(meta_row["mask_width"])
    mask_h = int(meta_row["mask_height"])

    # Prefer explicit factors if present in CSV
    if (
        "mask_width_downsample_factor" in meta_row
        and "mask_height_downsample_factor" in meta_row
    ):
        fx = float(meta_row["mask_width_downsample_factor"])
        fy = float(meta_row["mask_height_downsample_factor"])
        if fx <= 0 or fy <= 0:
            raise ValueError("mask_*_downsample_factor must be > 0")
        return mask_h, mask_w, fx, fy

    # Fallback: infer from level-0 dims
    fx = level0_w / mask_w
    fy = level0_h / mask_h
    return mask_h, mask_w, fx, fy


# -------------------------
# Stitching (Option 1)
# -------------------------
def bbox_level0_to_mask_rect(
    x0: int,
    y0: int,
    extent0_w: float,
    extent0_h: float,
    fx: float,
    fy: float,
) -> Tuple[int, int, int, int]:
    """
    Map a level-0 tile bounding box [x0, x1) x [y0, y1) to a mask pixel rectangle.
    Uses floor for start, ceil for end -> stable tessellation.
    Returns (x0m, y0m, x1m, y1m) in mask pixels (end exclusive).
    """
    x1 = x0 + extent0_w
    y1 = y0 + extent0_h

    x0m = int(math.floor(x0 / fx))
    y0m = int(math.floor(y0 / fy))
    x1m = int(math.ceil(x1 / fx))
    y1m = int(math.ceil(y1 / fy))

    # Ensure at least 1 pixel
    if x1m <= x0m:
        x1m = x0m + 1
    if y1m <= y0m:
        y1m = y0m + 1

    return x0m, y0m, x1m, y1m


def accumulate_tile_into_canvas(
    sum_map: torch.Tensor,  # (C,H,W) float32
    w_map: torch.Tensor,  # (H,W) float32
    tile_logits: torch.Tensor,  # (C, h, w) float32/float16 ok
    x0m: int,
    y0m: int,
    x1m: int,
    y1m: int,
):
    """
    Adds tile_logits into sum_map at the rectangle, with uniform weights.
    Handles clipping at canvas borders by cropping both the target rect and tile.
    """
    C, H, W = sum_map.shape

    # Clip target rect to canvas bounds
    tx0 = max(0, x0m)
    ty0 = max(0, y0m)
    tx1 = min(W, x1m)
    ty1 = min(H, y1m)

    if tx1 <= tx0 or ty1 <= ty0:
        return  # fully outside

    # Compute corresponding crop in tile space
    # Original requested size:
    out_w = x1m - x0m
    out_h = y1m - y0m

    # Offsets due to clipping
    ox0 = tx0 - x0m
    oy0 = ty0 - y0m
    ox1 = ox0 + (tx1 - tx0)
    oy1 = oy0 + (ty1 - ty0)

    tile_crop = tile_logits[:, oy0:oy1, ox0:ox1]  # (C, clipped_h, clipped_w)

    sum_map[:, ty0:ty1, tx0:tx1] += tile_crop
    w_map[ty0:ty1, tx0:tx1] += 1.0


def finalize_canvas(
    sum_map: torch.Tensor, w_map: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    sum_map: (C,H,W)
    w_map:   (H,W)
    returns avg_logits: (C,H,W)
    """
    w = torch.clamp(w_map, min=eps)
    return sum_map / w.unsqueeze(0)


# -------------------------
# CLI
# -------------------------
def get_autocast_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported autocast dtype: {name}")


@click.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--slides-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default=None,
    help="CSV with wsi_id, mask_width, mask_height (and optionally mask_*_downsample_factor).",
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--gpu-id", type=int, default=0, show_default=True)
@click.option("--no-autocast", is_flag=True, default=False)
@click.option(
    "--autocast-dtype",
    type=click.Choice(["float16", "bfloat16"], case_sensitive=False),
    default="float16",
    show_default=True,
)
@click.option(
    "--save-format",
    type=click.Choice(["pt", "npy"], case_sensitive=False),
    default="pt",
    show_default=True,
    help="How to save final avg logits on mask grid.",
)
@click.option(
    "--save-argmax-png",
    is_flag=True,
    default=False,
    help="Also save argmax segmentation as PNG.",
)
def main(
    tiles_dir: Path,
    slides_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    model_config: Path,
    metadata_csv: Optional[Path],
    batch_size: int,
    num_workers: int,
    gpu_id: int,
    no_autocast: bool,
    autocast_dtype: str,
    save_format: str,
    save_argmax_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not no_autocast)
    amp_dtype = get_autocast_dtype(autocast_dtype)

    meta_map = load_metadata_map(metadata_csv)

    bundle = load_experiment(
        config_path=str(model_config.resolve()),
        ckpt_path=str(checkpoint_path.resolve()),
        stage="predict",
        device=device,
        eval_mode=True,
    )
    model = bundle["model"].network.to(device).eval()

    tiling_store = build_tiling_store_from_dir(
        slides_root=slides_dir, root_dir=tiles_dir
    )

    slide_paths = sorted(slides_dir.glob("*.tif"))
    if not slide_paths:
        raise FileNotFoundError(f"No .tif slides found in {slides_dir}")

    for wsi_path in slide_paths:
        wsi_id = wsi_path.stem
        if wsi_id not in meta_map:
            raise KeyError(f"{wsi_id} not found in metadata CSV (need mask dims).")

        coords, _cont_idx, attrs = tiling_store.load_coords(wsi_id)

        wsi = openslide.OpenSlide(str(wsi_path))
        try:
            level0_w, level0_h = wsi.dimensions  # level-0 size
            mask_h, mask_w, fx, fy = get_mask_grid_and_factors(
                meta_map[wsi_id], level0_w, level0_h
            )

            ds = TileDataset(wsi=wsi, coords=coords, attrs=attrs)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            # How much area the tile covers in level-0 pixels
            level_ds = float(wsi.level_downsamples[ds.tile_level])
            extent0 = (
                ds.tile_size_lvl * level_ds
            )  # assume square; use extent0_w/extent0_h below

            # Allocate stitch buffers on CPU (safe for big WSIs)
            # You can put them on GPU if mask is small and you want speed.
            # We'll infer C from first batch.
            sum_map = None
            w_map = torch.zeros((mask_h, mask_w), dtype=torch.float32)

            for tiles, xy0 in loader:
                tiles = tiles.to(device, non_blocking=True)

                with torch.inference_mode():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            logits = model(tiles)
                    else:
                        logits = model(tiles)

                # Unwrap common output formats
                if isinstance(logits, dict):
                    logits = logits.get("logits", None)
                    if logits is None:
                        raise ValueError("Model returned dict without 'logits' key.")
                if logits.ndim != 4:
                    raise ValueError(
                        f"Expected logits (B,C,H,W), got {tuple(logits.shape)}"
                    )

                # Allocate sum_map after knowing C
                if sum_map is None:
                    C = int(logits.shape[1])
                    sum_map = torch.zeros((C, mask_h, mask_w), dtype=torch.float32)

                # Move logits to CPU for stitching to avoid GPU memory growth
                logits_cpu = logits.detach().float().cpu()  # (B,C,H,W)
                xy0_cpu = xy0.cpu().tolist()

                B = logits_cpu.shape[0]
                for b in range(B):
                    x0, y0 = xy0_cpu[b]

                    # Map tile bbox to mask rectangle
                    x0m, y0m, x1m, y1m = bbox_level0_to_mask_rect(
                        x0=x0,
                        y0=y0,
                        extent0_w=extent0,
                        extent0_h=extent0,
                        fx=fx,
                        fy=fy,
                    )
                    out_w = x1m - x0m
                    out_h = y1m - y0m

                    # Resize this tile's logits to exactly that rectangle
                    # (1,C,H,W) -> (1,C,out_h,out_w) -> (C,out_h,out_w)
                    tile_logits = F.interpolate(
                        logits_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    accumulate_tile_into_canvas(
                        sum_map=sum_map,
                        w_map=w_map,
                        tile_logits=tile_logits,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )

            if sum_map is None:
                raise RuntimeError(f"No tiles processed for {wsi_id}.")

            avg_logits = finalize_canvas(sum_map, w_map)  # (C,mask_h,mask_w)

            out_path = output_dir / f"{wsi_id}_logits_maskgrid.{save_format.lower()}"
            if save_format.lower() == "pt":
                torch.save(
                    {
                        "wsi_id": wsi_id,
                        "avg_logits": avg_logits,  # float32 CPU tensor
                        "weight_map": w_map,
                        "mask_h": mask_h,
                        "mask_w": mask_w,
                        "level0_w": level0_w,
                        "level0_h": level0_h,
                        "fx": fx,
                        "fy": fy,
                    },
                    out_path,
                )
            else:
                np.save(out_path, avg_logits.numpy())

            print(f"[OK] {wsi_id}: stitched -> {out_path}")

            if save_argmax_png:
                # (C,H,W) -> (H,W)
                pred = torch.argmax(avg_logits, dim=0).to(torch.uint8).cpu().numpy()

                from PIL import Image

                png_path = output_dir / f"{wsi_id}_argmax.png"
                img = Image.fromarray(pred, mode="L")
                img.save(png_path)

                print(f"[OK] saved argmax PNG -> {png_path}")

        finally:
            wsi.close()


if __name__ == "__main__":
    main()
