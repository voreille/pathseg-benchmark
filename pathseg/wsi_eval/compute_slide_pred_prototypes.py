from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import click
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from histoseg_plugin.storage.factory import build_tiling_store_from_dir
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms.v2 import functional as Fv2
from tqdm import tqdm

from pathseg.utils.load_experiment import load_experiment
from pathseg.utils.reinhard_normalization import ReinhardNormalize


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

PATTERN_MAP = {
    "background": 0,
    "Micropapillary": 1,
    "Solid": 2,
    "solid": 2,
    "Papillary": 3,
    "Acinar": 4,
    "Lepidic": 5,
}


class CenterPad(torch.nn.Module):
    def __init__(self, img_size: tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        self.center_crop = T.CenterCrop(img_size)

    def pad(self, img: torch.Tensor) -> torch.Tensor:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]
        return Fv2.pad(img, padding, padding_mode="edge")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = self.pad(img)
        return self.center_crop(img)


def build_tile_transform(
    img_size: tuple[int, int],
    use_reinhard: bool,
    reinhard_target_image: Optional[Path],
) -> Callable:
    transforms = []

    if use_reinhard:
        if reinhard_target_image is None:
            raise ValueError(
                "reinhard_target_image must be provided when use_reinhard=True"
            )
        transforms.append(
            ReinhardNormalize(
                target_image_path=str(reinhard_target_image),
                return_tensor=True,
            )
        )
    else:
        transforms.append(T.ToTensor())

    transforms.append(CenterPad(img_size=img_size))
    return T.Compose(transforms)


def load_support(
    df: pd.DataFrame,
    root: str | Path,
    transform: Optional[Callable] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
    df = df.reset_index(drop=True).copy()
    root = Path(root)

    required_cols = {"png_filename", "annotation_class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

    images_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    image_ids: list[str] = []

    for _, row in df.iterrows():
        image_path = root / row["png_filename"]
        image_id = image_path.stem
        label = str(row["annotation_class"])

        image = Image.open(image_path).convert("RGB")
        if transform is not None:
            image = transform(image)
        else:
            image = T.ToTensor()(image)

        images_list.append(image)
        labels_list.append(PATTERN_MAP[label])
        image_ids.append(image_id)

    images = torch.stack(images_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    if device is not None:
        images = images.to(device)
        labels = labels.to(device)

    return images, labels, image_ids


class TileDataset(Dataset):
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords: Sequence[Tuple[int, int]],
        attrs: Dict[str, Any],
        transforms: Optional[Callable] = None,
    ):
        self.wsi = wsi
        self.coords = coords
        self.tile_size_lvl = int(attrs["patch_size"])
        self.tile_level = int(attrs["patch_level"])
        self.transforms = transforms or T.Compose([T.ToTensor()])

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


def get_mpp(wsi: openslide.OpenSlide) -> float:
    mpp_x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is None or mpp_y is None:
        raise ValueError("WSI is missing MPP properties (openslide.mpp-x / mpp-y).")

    if float(mpp_x) != float(mpp_y):
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


def predict_from_stitched_threshold(
    avg_logits_preseg: torch.Tensor,
    avg_logits_pattern: torch.Tensor,
    fg_threshold: float,
) -> torch.Tensor:
    fg_prob = torch.sigmoid(avg_logits_preseg)[0]
    pred_patterns = torch.argmax(avg_logits_pattern, dim=0) + 1
    pred = pred_patterns.clone()
    pred[fg_prob < fg_threshold] = 0
    return pred


def predict_from_stitched_probabilistic(
    avg_logits_preseg: torch.Tensor,
    avg_logits_pattern: torch.Tensor,
) -> torch.Tensor:
    fg_prob = torch.sigmoid(avg_logits_preseg)[0]
    pattern_prob = torch.softmax(avg_logits_pattern, dim=0)

    p_bg = (1.0 - fg_prob).unsqueeze(0)
    p_fg = fg_prob.unsqueeze(0) * pattern_prob
    full_prob = torch.cat([p_bg, p_fg], dim=0)

    return torch.argmax(full_prob, dim=0)


def build_prediction_map(
    avg_logits_preseg: torch.Tensor,
    avg_logits_pattern: torch.Tensor,
    inference_mode: str,
    fg_threshold: float,
) -> torch.Tensor:
    if inference_mode == "threshold":
        return predict_from_stitched_threshold(
            avg_logits_preseg=avg_logits_preseg,
            avg_logits_pattern=avg_logits_pattern,
            fg_threshold=fg_threshold,
        )
    if inference_mode == "probabilistic":
        return predict_from_stitched_probabilistic(
            avg_logits_preseg=avg_logits_preseg,
            avg_logits_pattern=avg_logits_pattern,
        )
    raise ValueError(f"Unsupported inference_mode={inference_mode}")


def save_png_mask(mask: torch.Tensor, path: Path) -> None:
    arr = mask.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


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
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--support-dir",
    type=click.Path(file_okay=False, path_type=Path, exists=True),
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
@click.option(
    "--save-format",
    type=click.Choice(["pt", "npy"], case_sensitive=False),
    default="pt",
    show_default=True,
)
@click.option(
    "--save-argmax-png",
    is_flag=True,
    default=False,
    help="Save PNG masks for final pred, head_a, and head_b_aux when available.",
)
@click.option(
    "--inference-mode",
    type=click.Choice(["threshold", "probabilistic"], case_sensitive=False),
    default="threshold",
    show_default=True,
)
@click.option(
    "--fg-threshold",
    type=float,
    default=0.5,
    show_default=True,
)
@click.option(
    "--use-reinhard/--no-use-reinhard",
    default=False,
    show_default=True,
    help="Apply Reinhard normalization to tiles and support images.",
)
@click.option(
    "--reinhard-target-image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Target image used for Reinhard normalization.",
)
def main(
    tiles_dir: Path,
    slides_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    support_dir: Path,
    batch_size: int,
    num_workers: int,
    gpu_id: int,
    output_target_mpp: float,
    no_autocast: bool,
    autocast_dtype: str,
    save_format: str,
    save_argmax_png: bool,
    inference_mode: str,
    fg_threshold: float,
    use_reinhard: bool,
    reinhard_target_image: Optional[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not no_autocast)
    amp_dtype = get_autocast_dtype(autocast_dtype)

    checkpoint_dir = checkpoint_dir.resolve()
    model_config = checkpoint_dir / "loading_config.yaml"
    checkpoint_paths = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    if len(checkpoint_paths) > 1:
        print(
            f"Warning: multiple .ckpt files found in {checkpoint_dir}, using {checkpoint_paths[0].name}"
        )
    checkpoint_path = checkpoint_paths[0]

    bundle = load_experiment(
        config_path=str(model_config.resolve()),
        ckpt_path=str(checkpoint_path.resolve()),
        stage="fit",
        device=device,
        eval_mode=True,
    )
    pl_model = bundle["model"]
    model = pl_model.network.to(device).eval()

    if not hasattr(pl_model, "img_size"):
        raise AttributeError("pl_model must expose img_size")

    tile_size = pl_model.img_size
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    elif isinstance(tile_size, (list, tuple)):
        tile_size = tuple(tile_size)
    else:
        raise TypeError(f"Unsupported img_size type: {type(tile_size)}")

    tile_transform = build_tile_transform(
        img_size=tile_size,
        use_reinhard=use_reinhard,
        reinhard_target_image=reinhard_target_image,
    )

    support_images, support_labels, _support_image_ids = load_support(
        df=pd.read_csv(support_dir / "metadata.csv"),
        root=support_dir,
        transform=tile_transform,
        device=device,
    )

    prototype_ctx = model.fit_prototypes_from_image_labels(
        images=support_images,
        image_labels=support_labels,
    )

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
            if save_argmax_png:
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

            sum_map_preseg: Optional[torch.Tensor] = None
            sum_map_pattern: Optional[torch.Tensor] = None
            sum_map_a: Optional[torch.Tensor] = None
            sum_map_b_aux: Optional[torch.Tensor] = None

            w_map_preseg = torch.zeros((mask_h, mask_w), dtype=torch.float32)
            w_map_pattern = torch.zeros((mask_h, mask_w), dtype=torch.float32)
            w_map_a = torch.zeros((mask_h, mask_w), dtype=torch.float32)
            w_map_b_aux = torch.zeros((mask_h, mask_w), dtype=torch.float32)

            has_b_aux = False

            for tiles, xy0 in tqdm(loader, desc=f"Processing {wsi_id}"):
                tiles = tiles.to(device, non_blocking=True)

                with torch.inference_mode():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            out = model(tiles, ctx=prototype_ctx)
                    else:
                        out = model(tiles, ctx=prototype_ctx)

                required_keys = {"logits_a", "logits_preseg", "logits_b"}
                missing = required_keys - set(out.keys())
                if missing:
                    raise KeyError(
                        f"Model output missing required keys: {sorted(missing)}"
                    )

                logits_a_cpu = out["logits_a"].detach().float().cpu()
                logits_preseg_cpu = out["logits_preseg"].detach().float().cpu()
                logits_pattern_cpu = out["logits_b"].detach().float().cpu()
                logits_b_aux_cpu = (
                    out["logits_b_aux"].detach().float().cpu()
                    if "logits_b_aux" in out
                    else None
                )
                if logits_b_aux_cpu is not None:
                    has_b_aux = True

                xy0_cpu = xy0.cpu().tolist()

                if sum_map_preseg is None:
                    sum_map_preseg = torch.zeros(
                        (1, mask_h, mask_w), dtype=torch.float32
                    )

                if sum_map_pattern is None:
                    sum_map_pattern = torch.zeros(
                        (int(logits_pattern_cpu.shape[1]), mask_h, mask_w),
                        dtype=torch.float32,
                    )

                if sum_map_a is None:
                    sum_map_a = torch.zeros(
                        (int(logits_a_cpu.shape[1]), mask_h, mask_w),
                        dtype=torch.float32,
                    )

                if logits_b_aux_cpu is not None and sum_map_b_aux is None:
                    sum_map_b_aux = torch.zeros(
                        (int(logits_b_aux_cpu.shape[1]), mask_h, mask_w),
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

                    tile_logits_preseg = F.interpolate(
                        logits_preseg_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    tile_logits_pattern = F.interpolate(
                        logits_pattern_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    tile_logits_a = F.interpolate(
                        logits_a_cpu[b : b + 1],
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

                    accumulate_tile_into_canvas(
                        sum_map=sum_map_preseg,
                        w_map=w_map_preseg,
                        tile_logits=tile_logits_preseg,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )
                    accumulate_tile_into_canvas(
                        sum_map=sum_map_pattern,
                        w_map=w_map_pattern,
                        tile_logits=tile_logits_pattern,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )
                    accumulate_tile_into_canvas(
                        sum_map=sum_map_a,
                        w_map=w_map_a,
                        tile_logits=tile_logits_a,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )

                    if logits_b_aux_cpu is not None and sum_map_b_aux is not None:
                        tile_logits_b_aux = F.interpolate(
                            logits_b_aux_cpu[b : b + 1],
                            size=(out_h, out_w),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        accumulate_tile_into_canvas(
                            sum_map=sum_map_b_aux,
                            w_map=w_map_b_aux,
                            tile_logits=tile_logits_b_aux,
                            x0m=x0m,
                            y0m=y0m,
                            x1m=x1m,
                            y1m=y1m,
                        )

            if sum_map_preseg is None or sum_map_pattern is None or sum_map_a is None:
                raise RuntimeError(f"No tiles processed for {wsi_id}.")

            avg_logits_preseg = finalize_canvas(sum_map_preseg, w_map_preseg)
            avg_logits_pattern = finalize_canvas(sum_map_pattern, w_map_pattern)
            avg_logits_a = finalize_canvas(sum_map_a, w_map_a)
            avg_logits_b_aux = (
                finalize_canvas(sum_map_b_aux, w_map_b_aux)
                if has_b_aux and sum_map_b_aux is not None
                else None
            )

            pred_final = build_prediction_map(
                avg_logits_preseg=avg_logits_preseg,
                avg_logits_pattern=avg_logits_pattern,
                inference_mode=inference_mode.lower(),
                fg_threshold=fg_threshold,
            )

            pred_a = torch.argmax(avg_logits_a, dim=0).to(torch.uint8)
            pred_b_aux = (
                torch.argmax(avg_logits_b_aux, dim=0).to(torch.uint8)
                if avg_logits_b_aux is not None
                else None
            )

            out_path = output_dir / f"{wsi_id}_stitched.{save_format.lower()}"
            payload = {
                "wsi_id": wsi_id,
                "avg_logits_preseg": avg_logits_preseg,
                "avg_logits_pattern": avg_logits_pattern,
                "avg_logits_a": avg_logits_a,
                "avg_logits_b_aux": avg_logits_b_aux,
                "weight_map_preseg": w_map_preseg,
                "weight_map_pattern": w_map_pattern,
                "weight_map_a": w_map_a,
                "weight_map_b_aux": w_map_b_aux if has_b_aux else None,
                "pred_final": pred_final.to(torch.uint8),
                "pred_a": pred_a,
                "pred_b_aux": pred_b_aux,
                "mask_h": mask_h,
                "mask_w": mask_w,
                "level0_w": level0_w,
                "level0_h": level0_h,
                "fx": fx,
                "fy": fy,
                "output_target_mpp": float(output_target_mpp),
                "inference_mode": inference_mode.lower(),
                "fg_threshold": float(fg_threshold),
                "use_reinhard": bool(use_reinhard),
                "reinhard_target_image": (
                    str(reinhard_target_image)
                    if reinhard_target_image is not None
                    else None
                ),
                "head_a_label_map": HEAD_A_LABEL_MAP,
                "head_b_label_map": HEAD_B_LABEL_MAP,
            }

            if save_format.lower() == "pt":
                torch.save(payload, out_path)
            else:
                np_payload = {
                    k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in payload.items()
                }
                np.save(out_path, np_payload, allow_pickle=True)

            print(f"[OK] {wsi_id}: stitched -> {out_path}")

            if save_argmax_png:
                save_png_mask(pred_final, output_dir / f"{wsi_id}_pred_final.png")
                save_png_mask(pred_a, output_dir / f"{wsi_id}_pred_head_a.png")
                print(
                    f"[OK] saved final pred PNG -> {output_dir / f'{wsi_id}_pred_final.png'}"
                )
                print(
                    f"[OK] saved head_a pred PNG -> {output_dir / f'{wsi_id}_pred_head_a.png'}"
                )

                if pred_b_aux is not None:
                    save_png_mask(
                        pred_b_aux,
                        output_dir / f"{wsi_id}_pred_head_b_aux.png",
                    )
                    print(
                        f"[OK] saved head_b_aux pred PNG -> {output_dir / f'{wsi_id}_pred_head_b_aux.png'}"
                    )

                if thumbnail is not None:
                    thumb_path = output_dir / f"{wsi_id}_thumbnail.png"
                    thumbnail.save(thumb_path)
                    print(f"[OK] saved thumbnail PNG -> {thumb_path}")

        finally:
            wsi.close()


if __name__ == "__main__":
    main()
