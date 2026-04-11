from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import click
import numpy as np
import openslide
import torch
import torch.nn.functional as F
import yaml
from histoseg_plugin.storage.factory import build_tiling_store_from_dir
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


HEAD_A_LABEL_MAP = {
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

HEAD_B_LABEL_MAP = {
    "background": 0,
    "cribriform": 1,
    "micropapillary": 2,
    "solid": 3,
    "papillary": 4,
    "acinar": 5,
    "lepidic": 6,
}


def import_from_string(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class TileDataset(Dataset):
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords: Sequence[Tuple[int, int]],
        attrs: Dict[str, Any],
        transforms: Optional[Any] = None,
    ):
        self.wsi = wsi
        self.coords = list(coords)
        self.tile_size_lvl = int(attrs["patch_size"])
        self.tile_level = int(attrs["patch_level"])
        self.transforms = transforms or T.ToTensor()

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, y0 = self.coords[idx]
        tile = self.wsi.read_region(
            (int(x0), int(y0)),
            self.tile_level,
            (self.tile_size_lvl, self.tile_size_lvl),
        ).convert("RGB")
        tile = self.transforms(tile)
        return tile, torch.tensor([x0, y0], dtype=torch.long)


def build_tile_transform(img_size: tuple[int, int]) -> Any:
    return T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
        ]
    )


def get_mpp(wsi: openslide.OpenSlide) -> float:
    mpp_x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is None or mpp_y is None:
        raise ValueError(
            "WSI is missing MPP properties (openslide.mpp-x / openslide.mpp-y)."
        )

    if abs(float(mpp_x) - float(mpp_y)) > 1e-8:
        raise ValueError(
            f"Non-square pixels not supported (mpp_x={mpp_x}, mpp_y={mpp_y})."
        )

    return float(mpp_x)


def bbox_level0_to_mask_rect(
    x0: int,
    y0: int,
    extent0_w: float,
    extent0_h: float,
    fx: float,
    fy: float,
) -> Tuple[int, int, int, int]:
    x1 = x0 + extent0_w
    y1 = y0 + extent0_h

    x0m = int(math.floor(x0 / fx))
    y0m = int(math.floor(y0 / fy))
    x1m = int(math.ceil(x1 / fx))
    y1m = int(math.ceil(y1 / fy))

    if x1m <= x0m:
        x1m = x0m + 1
    if y1m <= y0m:
        y1m = y0m + 1

    return x0m, y0m, x1m, y1m


def accumulate_tile_into_canvas(
    sum_map: torch.Tensor,
    w_map: torch.Tensor,
    tile_logits: torch.Tensor,
    x0m: int,
    y0m: int,
    x1m: int,
    y1m: int,
) -> None:
    _, h, w = sum_map.shape

    tx0 = max(0, x0m)
    ty0 = max(0, y0m)
    tx1 = min(w, x1m)
    ty1 = min(h, y1m)
    if tx1 <= tx0 or ty1 <= ty0:
        return

    ox0 = tx0 - x0m
    oy0 = ty0 - y0m
    ox1 = ox0 + (tx1 - tx0)
    oy1 = oy0 + (ty1 - ty0)

    tile_crop = tile_logits[:, oy0:oy1, ox0:ox1]
    sum_map[:, ty0:ty1, tx0:tx1] += tile_crop
    w_map[ty0:ty1, tx0:tx1] += 1.0


def finalize_canvas(
    sum_map: torch.Tensor,
    w_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return sum_map / torch.clamp(w_map, min=eps).unsqueeze(0)


def get_autocast_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported autocast dtype: {name}")


def save_png_mask(mask: torch.Tensor, path: Path) -> None:
    arr = mask.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def load_model_and_conformal(
    model_dir: Path,
) -> tuple[torch.nn.Module, tuple[int, int] | None, dict[str, Any] | None]:
    model_yaml = model_dir / "model.yaml"
    weights_pt = model_dir / "weights.pt"
    conformal_yaml = model_dir / "conformal.yaml"

    if not model_yaml.exists():
        raise FileNotFoundError(f"Missing model yaml: {model_yaml}")
    if not weights_pt.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_pt}")

    with model_yaml.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_factory = model_cfg["model"]["factory"]
    model_init_args = model_cfg["model"]["init_args"]

    cls = import_from_string(model_factory)
    model = cls(**model_init_args)
    state_dict = torch.load(weights_pt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    img_size = model_init_args.get("img_size", None)

    conformal_cfg = None
    if conformal_yaml.exists():
        with conformal_yaml.open("r", encoding="utf-8") as f:
            conformal_cfg = yaml.safe_load(f)

    return model, img_size, conformal_cfg


def get_head_b_thresholds(
    conformal_cfg: dict[str, Any] | None,
    num_classes: int,
    bg_idx: int = 0,
) -> torch.Tensor | None:
    """
    Returns thresholds tensor of shape [C].
    Background is always set to NaN because it is excluded from conformal logic.
    Supports thresholds stored either for:
      - all classes [C]
      - foreground classes only [C-1]
    """
    if conformal_cfg is None:
        return None

    heads = conformal_cfg.get("heads", {})
    if "b" not in heads:
        return None

    thresholds = heads["b"].get("thresholds", None)
    if thresholds is None:
        return None

    vals = []
    for t in thresholds:
        vals.append(float("nan") if t is None else float(t))

    if len(vals) == num_classes:
        out = torch.tensor(vals, dtype=torch.float32)
    elif len(vals) == num_classes - 1:
        out = torch.full((num_classes,), float("nan"), dtype=torch.float32)
        fg_ids = [c for c in range(num_classes) if c != bg_idx]
        for cls_id, t in zip(fg_ids, vals):
            out[cls_id] = t
    else:
        raise ValueError(
            f"Unexpected number of head B thresholds: got {len(vals)}, "
            f"expected {num_classes} or {num_classes - 1}"
        )

    out[bg_idx] = float("nan")
    return out


def compute_covered_mask(w_map: torch.Tensor) -> torch.Tensor:
    return w_map > 0


def compute_head_a_prediction(
    avg_logits_a: torch.Tensor,
    covered_mask_a: torch.Tensor,
    bg_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Head A:
      - uncovered pixels => forced background
      - covered pixels => standard softmax over all classes
    """
    probs_a = torch.softmax(avg_logits_a, dim=0)  # [C,H,W]
    probs_a = probs_a.clone()

    uncovered = ~covered_mask_a
    if uncovered.any():
        probs_a[:, uncovered] = 0.0
        probs_a[bg_idx, uncovered] = 1.0

    pred_a = torch.argmax(probs_a, dim=0).to(torch.uint8)

    return {
        "probs_a": probs_a,
        "pred_a": pred_a,
        "covered_mask_a": covered_mask_a,
    }


def compute_conformal_head_b(
    avg_logits_b: torch.Tensor,
    thresholds_b: torch.Tensor | None,
    covered_mask_b: torch.Tensor,
    support_mask: torch.Tensor | None = None,
    bg_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """
    avg_logits_b: [C,H,W]
    thresholds_b: [C] with NaN for excluded classes, typically bg=NaN
    covered_mask_b: [H,W] bool
    support_mask: optional extra mask, e.g. from head A
    """

    if avg_logits_b.ndim != 3:
        raise ValueError(
            f"avg_logits_b must be [C,H,W], got {tuple(avg_logits_b.shape)}"
        )

    c, h, w = avg_logits_b.shape

    probs_b = torch.softmax(avg_logits_b, dim=0)  # full softmax over all classes
    pred_b_argmax = torch.argmax(probs_b, dim=0).to(torch.uint8)

    valid_mask = covered_mask_b.clone()
    if support_mask is not None:
        valid_mask = valid_mask & support_mask

    # where we actually apply conformal:
    # only where model predicts non-bg
    apply_conformal_mask = valid_mask & (pred_b_argmax != bg_idx)

    # default safe prediction = argmax
    pred_b_safe = pred_b_argmax.clone()
    pred_b_safe[~covered_mask_b] = bg_idx

    possible_b = torch.zeros_like(probs_b, dtype=torch.bool)
    set_size_b = torch.zeros((h, w), dtype=torch.uint8, device=avg_logits_b.device)

    if thresholds_b is None:
        return {
            "probs_b": probs_b,
            "pred_b_argmax": pred_b_argmax,
            "pred_b_safe": pred_b_safe,
            "possible_b": possible_b,
            "set_size_b": set_size_b,
            "apply_conformal_mask": apply_conformal_mask,
        }

    tau = 1.0 - thresholds_b
    valid_cls = ~torch.isnan(tau)

    fg_ids = [i for i in range(c) if i != bg_idx]

    for cls_id in fg_ids:
        if bool(valid_cls[cls_id]):
            possible_b[cls_id] = apply_conformal_mask & (probs_b[cls_id] >= tau[cls_id])

    set_size_b = possible_b.sum(dim=0).to(torch.uint8)

    # only pixels under conformal control can become ambiguous
    pred_b_safe[apply_conformal_mask] = 255

    singleton = apply_conformal_mask & (set_size_b == 1)
    if singleton.any():
        pred_b_safe[singleton] = torch.argmax(
            possible_b[:, singleton].float(), dim=0
        ).to(torch.uint8)

    # optional choice:
    # if no FG class passes threshold, fall back to background
    empty = apply_conformal_mask & (set_size_b == 0)
    pred_b_safe[empty] = bg_idx

    return {
        "probs_b": probs_b,
        "pred_b_argmax": pred_b_argmax,
        "pred_b_safe": pred_b_safe,
        "possible_b": possible_b,
        "set_size_b": set_size_b,
        "apply_conformal_mask": apply_conformal_mask,
    }


def compute_area_stats_from_safe_possible(
    pred_b_safe: torch.Tensor,
    possible_b: torch.Tensor,
) -> dict[str, Any]:
    """
    pred_b_safe: [H,W] with 255 for ambiguous FG pixels
    possible_b: [C,H,W] bool
    """
    stats: dict[str, Any] = {}
    num_classes = int(possible_b.shape[0])

    for cls_id in range(num_classes):
        safe_area = int((pred_b_safe == cls_id).sum().item())
        possible_area = int(possible_b[cls_id].sum().item())
        stats[str(cls_id)] = {
            "safe_area_px": safe_area,
            "possible_area_px": possible_area,
        }

    stats["ambiguous"] = {
        "safe_area_px": int((pred_b_safe == 255).sum().item()),
    }

    return stats


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
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--gpu-id", type=int, default=0, show_default=True)
@click.option("--output-target-mpp", type=float, default=4.0, show_default=True)
@click.option("--no-autocast", is_flag=True, default=False)
@click.option(
    "--autocast-dtype",
    type=click.Choice(["float16", "bfloat16"], case_sensitive=False),
    default="float16",
    show_default=True,
)
@click.option("--save-pt", is_flag=True, default=False)
@click.option("--save-argmax-png", is_flag=True, default=False)
@click.option("--save-safe-png", is_flag=True, default=True)
def main(
    tiles_dir: Path,
    slides_dir: Path,
    output_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    gpu_id: int,
    output_target_mpp: float,
    no_autocast: bool,
    autocast_dtype: str,
    save_pt: bool,
    save_argmax_png: bool,
    save_safe_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not no_autocast)
    amp_dtype = get_autocast_dtype(autocast_dtype)

    model, tile_size, conformal_cfg = load_model_and_conformal(model_dir.resolve())
    model = model.to(device).eval()

    num_head_b_classes = len(HEAD_B_LABEL_MAP)
    thresholds_b = get_head_b_thresholds(
        conformal_cfg=conformal_cfg,
        num_classes=num_head_b_classes,
        bg_idx=0,
    )
    if thresholds_b is not None:
        thresholds_b = thresholds_b.to(device)

    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    elif isinstance(tile_size, (list, tuple)):
        tile_size = tuple(tile_size)
    else:
        raise TypeError(f"Unsupported img_size type: {type(tile_size)}")

    tile_transform = build_tile_transform(tile_size)

    tiling_store = build_tiling_store_from_dir(
        slides_root=slides_dir,
        root_dir=tiles_dir,
    )

    slide_paths = sorted(slides_dir.glob("*.tif"))
    if not slide_paths:
        raise FileNotFoundError(f"No .tif slides found in {slides_dir}")

    for wsi_path in slide_paths:
        wsi_id = wsi_path.stem
        coords, _cont_idx, attrs = tiling_store.load_coords(wsi_id)

        wsi = openslide.OpenSlide(str(wsi_path))
        try:
            mpp = get_mpp(wsi)
            level0_w, level0_h = wsi.level_dimensions[0]

            mask_h = int(math.ceil(level0_h * mpp / output_target_mpp))
            mask_w = int(math.ceil(level0_w * mpp / output_target_mpp))

            thumbnail = None
            if save_argmax_png or save_safe_png:
                thumbnail = wsi.get_thumbnail((mask_w, mask_h))
                mask_w, mask_h = thumbnail.size

            fx = level0_w / mask_w
            fy = level0_h / mask_h

            ds = TileDataset(
                wsi=wsi,
                coords=coords,
                attrs=attrs,
                transforms=tile_transform,
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            level_downsample = float(wsi.level_downsamples[ds.tile_level])
            tile_extent0 = ds.tile_size_lvl * level_downsample

            sum_map_a: Optional[torch.Tensor] = None
            sum_map_b: Optional[torch.Tensor] = None
            w_map_a = torch.zeros((mask_h, mask_w), dtype=torch.float32)
            w_map_b = torch.zeros((mask_h, mask_w), dtype=torch.float32)

            for tiles, xy0 in tqdm(loader, desc=f"Processing {wsi_id}"):
                tiles = tiles.to(device, non_blocking=True)

                with torch.inference_mode():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            out = model(tiles)
                    else:
                        out = model(tiles)

                required_keys = {"logits_a", "logits_b"}
                missing = required_keys - set(out.keys())
                if missing:
                    raise KeyError(
                        f"Model output missing required keys: {sorted(missing)}"
                    )

                logits_a_cpu = out["logits_a"].detach().float().cpu()
                logits_b_cpu = out["logits_b"].detach().float().cpu()
                xy0_cpu = xy0.cpu().tolist()

                if sum_map_a is None:
                    sum_map_a = torch.zeros(
                        (int(logits_a_cpu.shape[1]), mask_h, mask_w),
                        dtype=torch.float32,
                    )

                if sum_map_b is None:
                    sum_map_b = torch.zeros(
                        (int(logits_b_cpu.shape[1]), mask_h, mask_w),
                        dtype=torch.float32,
                    )

                for b, (x0, y0) in enumerate(xy0_cpu):
                    x0m, y0m, x1m, y1m = bbox_level0_to_mask_rect(
                        x0=int(x0),
                        y0=int(y0),
                        extent0_w=tile_extent0,
                        extent0_h=tile_extent0,
                        fx=fx,
                        fy=fy,
                    )
                    out_w = x1m - x0m
                    out_h = y1m - y0m

                    tile_logits_a = F.interpolate(
                        logits_a_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    tile_logits_b = F.interpolate(
                        logits_b_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    accumulate_tile_into_canvas(
                        sum_map=sum_map_a,
                        w_map=w_map_a,
                        tile_logits=tile_logits_a,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )
                    accumulate_tile_into_canvas(
                        sum_map=sum_map_b,
                        w_map=w_map_b,
                        tile_logits=tile_logits_b,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )

            if sum_map_a is None or sum_map_b is None:
                raise RuntimeError(f"No tiles processed for {wsi_id}.")

            avg_logits_a = finalize_canvas(sum_map_a, w_map_a)
            avg_logits_b = finalize_canvas(sum_map_b, w_map_b)

            covered_mask_a = compute_covered_mask(w_map_a)
            covered_mask_b = compute_covered_mask(w_map_b)

            head_a = compute_head_a_prediction(
                avg_logits_a=avg_logits_a,
                covered_mask_a=covered_mask_a,
                bg_idx=0,
            )
            pred_a = head_a["pred_a"]

            # hard FG support from head A, but only where head B is covered
            valid_fg_mask_b = covered_mask_b & (pred_a != 0)

            conformal_b = compute_conformal_head_b(
                avg_logits_b=avg_logits_b.to(device),
                thresholds_b=thresholds_b,
                covered_mask_b=covered_mask_b.to(device),
                bg_idx=0,
            )

            pred_b_argmax = conformal_b["pred_b_argmax"].cpu()
            pred_b_safe = conformal_b["pred_b_safe"].cpu()
            possible_b = conformal_b["possible_b"].cpu()
            set_size_b = conformal_b["set_size_b"].cpu()
            probs_b = conformal_b["probs_b"].cpu()

            area_stats_b = compute_area_stats_from_safe_possible(
                pred_b_safe=pred_b_safe,
                possible_b=possible_b,
            )

            payload = {
                "wsi_id": wsi_id,
                "avg_logits_a": avg_logits_a,
                "avg_logits_b": avg_logits_b.cpu(),
                "pred_a": pred_a,
                "pred_b_argmax": pred_b_argmax,
                "pred_b_safe": pred_b_safe,
                "possible_b": possible_b,
                "set_size_b": set_size_b,
                "covered_mask_a": covered_mask_a,
                "covered_mask_b": covered_mask_b,
                "valid_fg_mask_b": valid_fg_mask_b,
                "area_stats_b": area_stats_b,
                "mask_h": mask_h,
                "mask_w": mask_w,
                "level0_w": level0_w,
                "level0_h": level0_h,
                "fx": fx,
                "fy": fy,
                "output_target_mpp": float(output_target_mpp),
                "head_a_label_map": HEAD_A_LABEL_MAP,
                "head_b_label_map": HEAD_B_LABEL_MAP,
            }

            if save_pt:
                out_path = output_dir / f"{wsi_id}_stitched.pt"
                torch.save(payload, out_path)
                print(f"[OK] {wsi_id}: stitched -> {out_path}")

            if save_argmax_png:
                save_png_mask(pred_a, output_dir / f"{wsi_id}_pred_head_a.png")
                save_png_mask(
                    pred_b_argmax, output_dir / f"{wsi_id}_pred_head_b_argmax.png"
                )

            if save_safe_png:
                save_png_mask(
                    pred_b_safe, output_dir / f"{wsi_id}_pred_head_b_safe.png"
                )

            if thumbnail is not None:
                thumb_path = output_dir / f"{wsi_id}_thumbnail.png"
                thumbnail.save(thumb_path)

            stats_path = output_dir / f"{wsi_id}_area_stats_b.yaml"
            with stats_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(area_stats_b, f, sort_keys=False)

        finally:
            wsi.close()


if __name__ == "__main__":
    main()
