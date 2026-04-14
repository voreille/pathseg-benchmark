from __future__ import annotations

import csv
import importlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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

SAFE_AMBIGUOUS_ID = 255
FG_UNION_PATTERN_ID = -2
FG_UNION_PATTERN_NAME = "__fg_union__"
EMPTY_CONFORMAL_SET_ID = -3
EMPTY_CONFORMAL_SET_NAME = "empty_conformal_set"
MULTI_CLASS_CONFORMAL_SET_NAME = "multi_class_conformal_set"


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


def normalize_img_size(img_size: Any) -> tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return (int(img_size[0]), int(img_size[1]))
    raise TypeError(f"Unsupported img_size type: {type(img_size)}")


def get_head_b_thresholds(
    conformal_cfg: dict[str, Any] | None,
    num_classes: int,
    bg_idx: int = 0,
) -> torch.Tensor | None:
    if conformal_cfg is None:
        return None

    heads = conformal_cfg.get("heads", {})
    if "b" not in heads:
        return None

    thresholds = heads["b"].get("thresholds", None)
    if thresholds is None:
        return None

    vals = [float("nan") if t is None else float(t) for t in thresholds]

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
    probs_a = torch.softmax(avg_logits_a, dim=0)
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
    if avg_logits_b.ndim != 3:
        raise ValueError(
            f"avg_logits_b must be [C,H,W], got {tuple(avg_logits_b.shape)}"
        )

    c, h, w = avg_logits_b.shape

    probs_b = torch.softmax(avg_logits_b, dim=0)
    pred_b_argmax = torch.argmax(probs_b, dim=0).to(torch.uint8)

    valid_mask = covered_mask_b.clone()
    if support_mask is not None:
        valid_mask = valid_mask & support_mask

    apply_conformal_mask = valid_mask & (pred_b_argmax != bg_idx)

    pred_b_safe = pred_b_argmax.clone()
    pred_b_safe[~covered_mask_b] = bg_idx

    possible_b = torch.zeros_like(probs_b, dtype=torch.bool)
    set_size_b = torch.zeros((h, w), dtype=torch.uint8, device=avg_logits_b.device)

    if thresholds_b is None:
        empty_conformal_set_mask = torch.zeros(
            (h, w), dtype=torch.bool, device=avg_logits_b.device
        )
        multi_class_conformal_set_mask = torch.zeros(
            (h, w), dtype=torch.bool, device=avg_logits_b.device
        )
        return {
            "probs_b": probs_b,
            "pred_b_argmax": pred_b_argmax,
            "pred_b_safe": pred_b_safe,
            "possible_b": possible_b,
            "set_size_b": set_size_b,
            "apply_conformal_mask": apply_conformal_mask,
            "empty_conformal_set_mask": empty_conformal_set_mask,
            "multi_class_conformal_set_mask": multi_class_conformal_set_mask,
        }

    tau = 1.0 - thresholds_b
    valid_cls = ~torch.isnan(tau)
    fg_ids = [i for i in range(c) if i != bg_idx]

    for cls_id in fg_ids:
        if bool(valid_cls[cls_id]):
            possible_b[cls_id] = apply_conformal_mask & (probs_b[cls_id] >= tau[cls_id])

    set_size_b = possible_b.sum(dim=0).to(torch.uint8)

    pred_b_safe[apply_conformal_mask] = SAFE_AMBIGUOUS_ID

    singleton = apply_conformal_mask & (set_size_b == 1)
    if singleton.any():
        pred_b_safe[singleton] = torch.argmax(
            possible_b[:, singleton].float(), dim=0
        ).to(torch.uint8)

    empty_conformal_set_mask = apply_conformal_mask & (set_size_b == 0)
    pred_b_safe[empty_conformal_set_mask] = bg_idx

    multi_class_conformal_set_mask = apply_conformal_mask & (set_size_b > 1)

    return {
        "probs_b": probs_b,
        "pred_b_argmax": pred_b_argmax,
        "pred_b_safe": pred_b_safe,
        "possible_b": possible_b,
        "set_size_b": set_size_b,
        "apply_conformal_mask": apply_conformal_mask,
        "empty_conformal_set_mask": empty_conformal_set_mask,
        "multi_class_conformal_set_mask": multi_class_conformal_set_mask,
    }


def compute_area_stats_from_safe_possible(
    pred_b_argmax: torch.Tensor,
    pred_b_safe: torch.Tensor,
    possible_b: torch.Tensor,
    covered_mask_b: torch.Tensor,
    apply_conformal_mask: torch.Tensor,
    set_size_b: torch.Tensor,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    num_classes = int(possible_b.shape[0])

    for cls_id in range(num_classes):
        stats[str(cls_id)] = {
            "argmax_area_px": int(
                ((pred_b_argmax == cls_id) & covered_mask_b).sum().item()
            ),
            "safe_area_px": int(
                ((pred_b_safe == cls_id) & covered_mask_b).sum().item()
            ),
            "max_possible_area_px": int(possible_b[cls_id].sum().item()),
        }

    stats["conformal_status"] = {
        "singleton_conformal_set_area_px": int(
            (apply_conformal_mask & (set_size_b == 1)).sum().item()
        ),
        "empty_conformal_set_area_px": int(
            (apply_conformal_mask & (set_size_b == 0)).sum().item()
        ),
        "multi_class_conformal_set_area_px": int(
            (apply_conformal_mask & (set_size_b > 1)).sum().item()
        ),
    }
    return stats


@dataclass(frozen=True)
class StatRow:
    wsi_id: str
    stat_type: str
    prediction_type: str
    compartment_id: int
    compartment_name: str
    pattern_id: int
    pattern_name: str
    area_px: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingRow:
    wsi_id: str
    num_tiles: int
    num_batches: int
    elapsed_s: float
    elapsed_min: float
    tiles_per_s: float
    batches_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    batch_size: int
    num_workers: int
    output_target_mpp: float
    save_pt: bool
    save_argmax_png: bool
    save_safe_png: bool


@dataclass(frozen=True)
class SlideGeometry:
    wsi_id: str
    level0_w: int
    level0_h: int
    mask_w: int
    mask_h: int
    fx: float
    fy: float
    mpp: float
    thumbnail: Image.Image | None


@dataclass
class SlideOutputs:
    wsi_id: str
    avg_logits_a: torch.Tensor
    avg_logits_b: torch.Tensor
    pred_a: torch.Tensor
    pred_b_argmax: torch.Tensor
    pred_b_safe: torch.Tensor
    possible_b: torch.Tensor
    set_size_b: torch.Tensor
    apply_conformal_mask: torch.Tensor
    empty_conformal_set_mask: torch.Tensor
    multi_class_conformal_set_mask: torch.Tensor
    covered_mask_a: torch.Tensor
    covered_mask_b: torch.Tensor
    valid_fg_mask_b: torch.Tensor
    area_stats_b: dict[str, Any]
    slide_stat_rows: list[StatRow]
    timing_row: TimingRow
    geometry: SlideGeometry


def make_id_to_name(
    label_map: dict[str, int], ignore_ids: set[int] | None = None
) -> dict[int, str]:
    ignore_ids = ignore_ids or set()
    out: dict[int, str] = {}
    for name, idx in label_map.items():
        if idx in ignore_ids:
            continue
        out[int(idx)] = str(name)
    return out


def masked_bincount(
    values: torch.Tensor,
    mask: torch.Tensor,
    minlength: int,
) -> torch.Tensor:
    values = values.long()
    mask = mask.bool()
    if values.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: values={tuple(values.shape)} mask={tuple(mask.shape)}"
        )
    if not mask.any():
        return torch.zeros((minlength,), dtype=torch.long)
    return torch.bincount(values[mask].reshape(-1), minlength=minlength)


def masked_joint_bincount(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    mask: torch.Tensor,
    num_a: int,
    num_b: int,
) -> torch.Tensor:
    values_a = values_a.long()
    values_b = values_b.long()
    mask = mask.bool()

    if not (values_a.shape == values_b.shape == mask.shape):
        raise ValueError(
            f"Shape mismatch: a={tuple(values_a.shape)} b={tuple(values_b.shape)} mask={tuple(mask.shape)}"
        )

    if not mask.any():
        return torch.zeros((num_a, num_b), dtype=torch.long)

    a = values_a[mask].reshape(-1)
    b = values_b[mask].reshape(-1)
    encoded = a * num_b + b
    counts = torch.bincount(encoded, minlength=num_a * num_b)
    return counts.reshape(num_a, num_b)


def build_head_a_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    roi_mask: torch.Tensor,
    a_id_to_name: dict[int, str],
) -> list[StatRow]:
    num_a = max(a_id_to_name.keys()) + 1
    counts = masked_bincount(pred_a, roi_mask, minlength=num_a)

    rows: list[StatRow] = []
    for comp_id, comp_name in sorted(a_id_to_name.items()):
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="head_a",
                prediction_type="argmax",
                compartment_id=comp_id,
                compartment_name=comp_name,
                pattern_id=-1,
                pattern_name="",
                area_px=int(counts[comp_id].item()),
            )
        )
    return rows


def build_head_b_rows(
    wsi_id: str,
    pred_b: torch.Tensor,
    roi_mask: torch.Tensor,
    b_id_to_name: dict[int, str],
    prediction_type: str,
    include_multi_class_conformal_set: bool = False,
) -> list[StatRow]:
    max_id = (
        SAFE_AMBIGUOUS_ID
        if include_multi_class_conformal_set
        else max(b_id_to_name.keys())
    )
    counts = masked_bincount(pred_b, roi_mask, minlength=max_id + 1)

    rows: list[StatRow] = []
    for pat_id, pat_name in sorted(b_id_to_name.items()):
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="head_b",
                prediction_type=prediction_type,
                compartment_id=-1,
                compartment_name="",
                pattern_id=pat_id,
                pattern_name=pat_name,
                area_px=int(counts[pat_id].item()),
            )
        )

    if include_multi_class_conformal_set:
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="head_b",
                prediction_type=prediction_type,
                compartment_id=-1,
                compartment_name="",
                pattern_id=SAFE_AMBIGUOUS_ID,
                pattern_name=MULTI_CLASS_CONFORMAL_SET_NAME,
                area_px=int(counts[SAFE_AMBIGUOUS_ID].item()),
            )
        )
    return rows


def build_head_b_max_possible_rows(
    wsi_id: str,
    possible_b: torch.Tensor,
    roi_mask: torch.Tensor,
    b_id_to_name: dict[int, str],
) -> list[StatRow]:
    rows: list[StatRow] = []
    for pat_id, pat_name in sorted(b_id_to_name.items()):
        area = int((possible_b[pat_id] & roi_mask).sum().item())
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="head_b",
                prediction_type="max_possible",
                compartment_id=-1,
                compartment_name="",
                pattern_id=pat_id,
                pattern_name=pat_name,
                area_px=area,
            )
        )
    return rows


def build_head_b_conformal_status_rows(
    wsi_id: str,
    apply_conformal_mask: torch.Tensor,
    set_size_b: torch.Tensor,
) -> list[StatRow]:
    singleton_area = int((apply_conformal_mask & (set_size_b == 1)).sum().item())
    empty_area = int((apply_conformal_mask & (set_size_b == 0)).sum().item())
    multi_area = int((apply_conformal_mask & (set_size_b > 1)).sum().item())

    return [
        StatRow(
            wsi_id=wsi_id,
            stat_type="head_b_conformal_status",
            prediction_type="safe",
            compartment_id=-1,
            compartment_name="",
            pattern_id=-1,
            pattern_name="singleton_conformal_set",
            area_px=singleton_area,
        ),
        StatRow(
            wsi_id=wsi_id,
            stat_type="head_b_conformal_status",
            prediction_type="safe",
            compartment_id=-1,
            compartment_name="",
            pattern_id=EMPTY_CONFORMAL_SET_ID,
            pattern_name=EMPTY_CONFORMAL_SET_NAME,
            area_px=empty_area,
        ),
        StatRow(
            wsi_id=wsi_id,
            stat_type="head_b_conformal_status",
            prediction_type="safe",
            compartment_id=-1,
            compartment_name="",
            pattern_id=SAFE_AMBIGUOUS_ID,
            pattern_name=MULTI_CLASS_CONFORMAL_SET_NAME,
            area_px=multi_area,
        ),
    ]


def build_a_by_b_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    roi_mask: torch.Tensor,
    a_id_to_name: dict[int, str],
    b_id_to_name: dict[int, str],
    prediction_type: str,
    include_multi_class_conformal_set: bool = False,
) -> list[StatRow]:
    num_a = max(a_id_to_name.keys()) + 1
    num_b = (
        SAFE_AMBIGUOUS_ID + 1
        if include_multi_class_conformal_set
        else max(b_id_to_name.keys()) + 1
    )

    counts = masked_joint_bincount(
        values_a=pred_a,
        values_b=pred_b,
        mask=roi_mask,
        num_a=num_a,
        num_b=num_b,
    )

    rows: list[StatRow] = []
    for comp_id, comp_name in sorted(a_id_to_name.items()):
        for pat_id, pat_name in sorted(b_id_to_name.items()):
            rows.append(
                StatRow(
                    wsi_id=wsi_id,
                    stat_type="a_by_b",
                    prediction_type=prediction_type,
                    compartment_id=comp_id,
                    compartment_name=comp_name,
                    pattern_id=pat_id,
                    pattern_name=pat_name,
                    area_px=int(counts[comp_id, pat_id].item()),
                )
            )

        if include_multi_class_conformal_set:
            rows.append(
                StatRow(
                    wsi_id=wsi_id,
                    stat_type="a_by_b",
                    prediction_type=prediction_type,
                    compartment_id=comp_id,
                    compartment_name=comp_name,
                    pattern_id=SAFE_AMBIGUOUS_ID,
                    pattern_name=MULTI_CLASS_CONFORMAL_SET_NAME,
                    area_px=int(counts[comp_id, SAFE_AMBIGUOUS_ID].item()),
                )
            )
    return rows


def build_a_by_b_max_possible_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    possible_b: torch.Tensor,
    roi_mask: torch.Tensor,
    a_id_to_name: dict[int, str],
    b_id_to_name: dict[int, str],
) -> list[StatRow]:
    rows: list[StatRow] = []
    num_a = max(a_id_to_name.keys()) + 1

    for pat_id, pat_name in sorted(b_id_to_name.items()):
        counts = masked_bincount(
            pred_a,
            roi_mask & possible_b[pat_id],
            minlength=num_a,
        )
        for comp_id, comp_name in sorted(a_id_to_name.items()):
            rows.append(
                StatRow(
                    wsi_id=wsi_id,
                    stat_type="a_by_b",
                    prediction_type="max_possible",
                    compartment_id=comp_id,
                    compartment_name=comp_name,
                    pattern_id=pat_id,
                    pattern_name=pat_name,
                    area_px=int(counts[comp_id].item()),
                )
            )
    return rows


def build_a_by_b_binary_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    roi_mask: torch.Tensor,
    a_id_to_name: dict[int, str],
    bg_idx: int,
    prediction_type: str,
    multi_class_id: int | None = None,
) -> list[StatRow]:
    rows: list[StatRow] = []
    num_a = max(a_id_to_name.keys()) + 1

    bg_mask = roi_mask & (pred_b == bg_idx)
    fg_mask = roi_mask & (pred_b != bg_idx)

    bg_counts = masked_bincount(pred_a, bg_mask, minlength=num_a)
    fg_counts = masked_bincount(pred_a, fg_mask, minlength=num_a)

    multi_counts = None
    if multi_class_id is not None:
        multi_mask = roi_mask & (pred_b == multi_class_id)
        fg_mask = roi_mask & (pred_b != bg_idx) & (pred_b != multi_class_id)
        fg_counts = masked_bincount(pred_a, fg_mask, minlength=num_a)
        multi_counts = masked_bincount(pred_a, multi_mask, minlength=num_a)

    for comp_id, comp_name in sorted(a_id_to_name.items()):
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="a_by_b_binary",
                prediction_type=prediction_type,
                compartment_id=comp_id,
                compartment_name=comp_name,
                pattern_id=bg_idx,
                pattern_name="background",
                area_px=int(bg_counts[comp_id].item()),
            )
        )
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="a_by_b_binary",
                prediction_type=prediction_type,
                compartment_id=comp_id,
                compartment_name=comp_name,
                pattern_id=FG_UNION_PATTERN_ID,
                pattern_name=FG_UNION_PATTERN_NAME,
                area_px=int(fg_counts[comp_id].item()),
            )
        )

        if multi_counts is not None:
            rows.append(
                StatRow(
                    wsi_id=wsi_id,
                    stat_type="a_by_b_binary",
                    prediction_type=prediction_type,
                    compartment_id=comp_id,
                    compartment_name=comp_name,
                    pattern_id=SAFE_AMBIGUOUS_ID,
                    pattern_name=MULTI_CLASS_CONFORMAL_SET_NAME,
                    area_px=int(multi_counts[comp_id].item()),
                )
            )

    return rows


def build_a_by_b_binary_max_possible_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    possible_b: torch.Tensor,
    roi_mask: torch.Tensor,
    a_id_to_name: dict[int, str],
) -> list[StatRow]:
    rows: list[StatRow] = []
    num_a = max(a_id_to_name.keys()) + 1

    fg_union_mask = roi_mask & possible_b.any(dim=0)
    bg_mask = roi_mask & ~possible_b.any(dim=0)

    bg_counts = masked_bincount(pred_a, bg_mask, minlength=num_a)
    fg_counts = masked_bincount(pred_a, fg_union_mask, minlength=num_a)

    for comp_id, comp_name in sorted(a_id_to_name.items()):
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="a_by_b_binary",
                prediction_type="max_possible",
                compartment_id=comp_id,
                compartment_name=comp_name,
                pattern_id=0,
                pattern_name="background",
                area_px=int(bg_counts[comp_id].item()),
            )
        )
        rows.append(
            StatRow(
                wsi_id=wsi_id,
                stat_type="a_by_b_binary",
                prediction_type="max_possible",
                compartment_id=comp_id,
                compartment_name=comp_name,
                pattern_id=FG_UNION_PATTERN_ID,
                pattern_name=FG_UNION_PATTERN_NAME,
                area_px=int(fg_counts[comp_id].item()),
            )
        )

    return rows


def build_slide_stats_rows(
    wsi_id: str,
    pred_a: torch.Tensor,
    pred_b_argmax: torch.Tensor,
    pred_b_safe: torch.Tensor,
    possible_b: torch.Tensor,
    apply_conformal_mask: torch.Tensor,
    set_size_b: torch.Tensor,
    covered_mask_a: torch.Tensor,
    covered_mask_b: torch.Tensor,
    head_a_label_map: dict[str, int],
    head_b_label_map: dict[str, int],
) -> list[StatRow]:
    a_id_to_name = make_id_to_name(head_a_label_map, ignore_ids={255})
    b_id_to_name = make_id_to_name(head_b_label_map)

    roi_a = covered_mask_a.bool()
    roi_b = covered_mask_b.bool()
    roi_ab = (covered_mask_a & covered_mask_b).bool()

    rows: list[StatRow] = []
    rows.extend(build_head_a_rows(wsi_id, pred_a, roi_a, a_id_to_name))
    rows.extend(
        build_head_b_rows(
            wsi_id, pred_b_argmax, roi_b, b_id_to_name, prediction_type="argmax"
        )
    )
    rows.extend(
        build_head_b_rows(
            wsi_id,
            pred_b_safe,
            roi_b,
            b_id_to_name,
            prediction_type="safe",
            include_multi_class_conformal_set=True,
        )
    )
    rows.extend(
        build_head_b_max_possible_rows(
            wsi_id=wsi_id,
            possible_b=possible_b,
            roi_mask=roi_b,
            b_id_to_name=b_id_to_name,
        )
    )
    rows.extend(
        build_head_b_conformal_status_rows(
            wsi_id=wsi_id,
            apply_conformal_mask=apply_conformal_mask,
            set_size_b=set_size_b,
        )
    )
    rows.extend(
        build_a_by_b_rows(
            wsi_id,
            pred_a,
            pred_b_argmax,
            roi_ab,
            a_id_to_name,
            b_id_to_name,
            prediction_type="argmax",
            include_multi_class_conformal_set=False,
        )
    )
    rows.extend(
        build_a_by_b_rows(
            wsi_id,
            pred_a,
            pred_b_safe,
            roi_ab,
            a_id_to_name,
            b_id_to_name,
            prediction_type="safe",
            include_multi_class_conformal_set=True,
        )
    )
    rows.extend(
        build_a_by_b_max_possible_rows(
            wsi_id=wsi_id,
            pred_a=pred_a,
            possible_b=possible_b,
            roi_mask=roi_ab,
            a_id_to_name=a_id_to_name,
            b_id_to_name=b_id_to_name,
        )
    )
    rows.extend(
        build_a_by_b_binary_rows(
            wsi_id=wsi_id,
            pred_a=pred_a,
            pred_b=pred_b_argmax,
            roi_mask=roi_ab,
            a_id_to_name=a_id_to_name,
            bg_idx=0,
            prediction_type="argmax",
            multi_class_id=None,
        )
    )
    rows.extend(
        build_a_by_b_binary_rows(
            wsi_id=wsi_id,
            pred_a=pred_a,
            pred_b=pred_b_safe,
            roi_mask=roi_ab,
            a_id_to_name=a_id_to_name,
            bg_idx=0,
            prediction_type="safe",
            multi_class_id=SAFE_AMBIGUOUS_ID,
        )
    )
    rows.extend(
        build_a_by_b_binary_max_possible_rows(
            wsi_id=wsi_id,
            pred_a=pred_a,
            possible_b=possible_b,
            roi_mask=roi_ab,
            a_id_to_name=a_id_to_name,
        )
    )
    return rows


def write_stat_rows_csv(path: Path, rows: Iterable[StatRow]) -> None:
    fieldnames = list(StatRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def append_timing_row_csv(path: Path, row: TimingRow) -> None:
    fieldnames = list(TimingRow.__dataclass_fields__.keys())
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row.to_dict())


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_runtime_config(
    batch_size: int,
    num_workers: int,
    gpu_id: int,
    output_target_mpp: float,
    no_autocast: bool,
    autocast_dtype: str,
    save_pt: bool,
    save_argmax_png: bool,
    save_safe_png: bool,
) -> RuntimeConfig:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not no_autocast)
    amp_dtype = get_autocast_dtype(autocast_dtype)

    return RuntimeConfig(
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        output_target_mpp=output_target_mpp,
        save_pt=save_pt,
        save_argmax_png=save_argmax_png,
        save_safe_png=save_safe_png,
    )


def build_slide_geometry(
    wsi: openslide.OpenSlide,
    wsi_id: str,
    output_target_mpp: float,
    need_thumbnail: bool,
) -> SlideGeometry:
    mpp = get_mpp(wsi)
    level0_w, level0_h = wsi.level_dimensions[0]

    mask_h = int(math.ceil(level0_h * mpp / output_target_mpp))
    mask_w = int(math.ceil(level0_w * mpp / output_target_mpp))

    thumbnail = None
    if need_thumbnail:
        thumbnail = wsi.get_thumbnail((mask_w, mask_h))
        mask_w, mask_h = thumbnail.size

    fx = level0_w / mask_w
    fy = level0_h / mask_h

    return SlideGeometry(
        wsi_id=wsi_id,
        level0_w=level0_w,
        level0_h=level0_h,
        mask_w=mask_w,
        mask_h=mask_h,
        fx=fx,
        fy=fy,
        mpp=mpp,
        thumbnail=thumbnail,
    )


def run_model_on_tiles(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for tiles, xy0 in tqdm(loader, desc="Tiles", leave=False):
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
            raise KeyError(f"Model output missing required keys: {sorted(missing)}")

        logits_a_cpu = out["logits_a"].detach().float().cpu()
        logits_b_cpu = out["logits_b"].detach().float().cpu()
        yield logits_a_cpu, logits_b_cpu, xy0.cpu()


def stitch_slide_logits(
    loader: DataLoader,
    model: torch.nn.Module,
    ds: TileDataset,
    geometry: SlideGeometry,
    runtime_cfg: RuntimeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    level_downsample = float(ds.wsi.level_downsamples[ds.tile_level])
    tile_extent0 = ds.tile_size_lvl * level_downsample

    sum_map_a: Optional[torch.Tensor] = None
    sum_map_b: Optional[torch.Tensor] = None
    w_map_a = torch.zeros((geometry.mask_h, geometry.mask_w), dtype=torch.float32)
    w_map_b = torch.zeros((geometry.mask_h, geometry.mask_w), dtype=torch.float32)

    for logits_a_cpu, logits_b_cpu, xy0_cpu in run_model_on_tiles(
        model=model,
        loader=loader,
        device=runtime_cfg.device,
        use_amp=runtime_cfg.use_amp,
        amp_dtype=runtime_cfg.amp_dtype,
    ):
        if sum_map_a is None:
            sum_map_a = torch.zeros(
                (int(logits_a_cpu.shape[1]), geometry.mask_h, geometry.mask_w),
                dtype=torch.float32,
            )

        if sum_map_b is None:
            sum_map_b = torch.zeros(
                (int(logits_b_cpu.shape[1]), geometry.mask_h, geometry.mask_w),
                dtype=torch.float32,
            )

        for b, (x0, y0) in enumerate(xy0_cpu.tolist()):
            x0m, y0m, x1m, y1m = bbox_level0_to_mask_rect(
                x0=int(x0),
                y0=int(y0),
                extent0_w=tile_extent0,
                extent0_h=tile_extent0,
                fx=geometry.fx,
                fy=geometry.fy,
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
        raise RuntimeError(f"No tiles processed for {geometry.wsi_id}.")

    avg_logits_a = finalize_canvas(sum_map_a, w_map_a)
    avg_logits_b = finalize_canvas(sum_map_b, w_map_b)
    return avg_logits_a, avg_logits_b, w_map_a, w_map_b


def build_payload(
    outputs: SlideOutputs,
    output_target_mpp: float,
) -> dict[str, Any]:
    return {
        "wsi_id": outputs.wsi_id,
        "avg_logits_a": outputs.avg_logits_a,
        "avg_logits_b": outputs.avg_logits_b,
        "pred_a": outputs.pred_a,
        "pred_b_argmax": outputs.pred_b_argmax,
        "pred_b_safe": outputs.pred_b_safe,
        "possible_b": outputs.possible_b,
        "max_possible_b": outputs.possible_b,
        "set_size_b": outputs.set_size_b,
        "apply_conformal_mask": outputs.apply_conformal_mask,
        "empty_conformal_set_mask": outputs.empty_conformal_set_mask,
        "multi_class_conformal_set_mask": outputs.multi_class_conformal_set_mask,
        "covered_mask_a": outputs.covered_mask_a,
        "covered_mask_b": outputs.covered_mask_b,
        "valid_fg_mask_b": outputs.valid_fg_mask_b,
        "area_stats_b": outputs.area_stats_b,
        "mask_h": outputs.geometry.mask_h,
        "mask_w": outputs.geometry.mask_w,
        "level0_w": outputs.geometry.level0_w,
        "level0_h": outputs.geometry.level0_h,
        "fx": outputs.geometry.fx,
        "fy": outputs.geometry.fy,
        "output_target_mpp": float(output_target_mpp),
        "head_a_label_map": HEAD_A_LABEL_MAP,
        "head_b_label_map": HEAD_B_LABEL_MAP,
        "timing": outputs.timing_row.to_dict(),
    }


def write_slide_outputs(
    output_dir: Path,
    outputs: SlideOutputs,
    output_target_mpp: float,
    save_pt: bool,
    save_argmax_png: bool,
    save_safe_png: bool,
) -> None:
    if save_pt:
        out_path = output_dir / f"{outputs.wsi_id}_stitched.pt"
        torch.save(build_payload(outputs, output_target_mpp), out_path)

    if save_argmax_png:
        save_png_mask(outputs.pred_a, output_dir / f"{outputs.wsi_id}_pred_head_a.png")
        save_png_mask(
            outputs.pred_b_argmax,
            output_dir / f"{outputs.wsi_id}_pred_head_b_argmax.png",
        )

    if save_safe_png:
        save_png_mask(
            outputs.pred_b_safe,
            output_dir / f"{outputs.wsi_id}_pred_head_b_safe.png",
        )
        save_png_mask(
            outputs.empty_conformal_set_mask.to(torch.uint8),
            output_dir / f"{outputs.wsi_id}_empty_conformal_set.png",
        )
        save_png_mask(
            outputs.multi_class_conformal_set_mask.to(torch.uint8),
            output_dir / f"{outputs.wsi_id}_multi_class_conformal_set.png",
        )

    if outputs.geometry.thumbnail is not None:
        thumb_path = output_dir / f"{outputs.wsi_id}_thumbnail.png"
        outputs.geometry.thumbnail.save(thumb_path)

    stats_path = output_dir / f"{outputs.wsi_id}_area_stats_b.yaml"
    with stats_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(outputs.area_stats_b, f, sort_keys=False)

    slide_stats_csv_path = output_dir / f"{outputs.wsi_id}_slide_stats.csv"
    write_stat_rows_csv(slide_stats_csv_path, outputs.slide_stat_rows)

    append_timing_row_csv(output_dir / "slide_timing.csv", outputs.timing_row)


def process_single_slide(
    wsi_path: Path,
    tiling_store: Any,
    model: torch.nn.Module,
    thresholds_b: torch.Tensor | None,
    tile_transform: Any,
    runtime_cfg: RuntimeConfig,
) -> SlideOutputs:
    wsi_id = wsi_path.stem
    coords, _cont_idx, attrs = tiling_store.load_coords(wsi_id)

    wsi = openslide.OpenSlide(str(wsi_path))
    try:
        geometry = build_slide_geometry(
            wsi=wsi,
            wsi_id=wsi_id,
            output_target_mpp=runtime_cfg.output_target_mpp,
            need_thumbnail=(runtime_cfg.save_argmax_png or runtime_cfg.save_safe_png),
        )

        ds = TileDataset(
            wsi=wsi,
            coords=coords,
            attrs=attrs,
            transforms=tile_transform,
        )
        loader = DataLoader(
            ds,
            batch_size=runtime_cfg.batch_size,
            shuffle=False,
            num_workers=runtime_cfg.num_workers,
            pin_memory=(runtime_cfg.device.type == "cuda"),
        )

        num_tiles = len(ds)
        num_batches = len(loader)

        sync_if_needed(runtime_cfg.device)
        t0 = time.perf_counter()

        avg_logits_a, avg_logits_b, w_map_a, w_map_b = stitch_slide_logits(
            loader=loader,
            model=model,
            ds=ds,
            geometry=geometry,
            runtime_cfg=runtime_cfg,
        )

        sync_if_needed(runtime_cfg.device)
        elapsed_s = time.perf_counter() - t0

        covered_mask_a = compute_covered_mask(w_map_a)
        covered_mask_b = compute_covered_mask(w_map_b)

        head_a = compute_head_a_prediction(
            avg_logits_a=avg_logits_a,
            covered_mask_a=covered_mask_a,
            bg_idx=0,
        )
        pred_a = head_a["pred_a"]

        valid_fg_mask_b = covered_mask_b & (pred_a != 0)

        conformal_b = compute_conformal_head_b(
            avg_logits_b=avg_logits_b.to(runtime_cfg.device),
            thresholds_b=thresholds_b,
            covered_mask_b=covered_mask_b.to(runtime_cfg.device),
            bg_idx=0,
        )

        pred_b_argmax = conformal_b["pred_b_argmax"].cpu()
        pred_b_safe = conformal_b["pred_b_safe"].cpu()
        possible_b = conformal_b["possible_b"].cpu()
        set_size_b = conformal_b["set_size_b"].cpu()
        apply_conformal_mask = conformal_b["apply_conformal_mask"].cpu()
        empty_conformal_set_mask = conformal_b["empty_conformal_set_mask"].cpu()
        multi_class_conformal_set_mask = conformal_b[
            "multi_class_conformal_set_mask"
        ].cpu()

        area_stats_b = compute_area_stats_from_safe_possible(
            pred_b_argmax=pred_b_argmax,
            pred_b_safe=pred_b_safe,
            possible_b=possible_b,
            covered_mask_b=covered_mask_b,
            apply_conformal_mask=apply_conformal_mask,
            set_size_b=set_size_b,
        )

        slide_stat_rows = build_slide_stats_rows(
            wsi_id=wsi_id,
            pred_a=pred_a,
            pred_b_argmax=pred_b_argmax,
            pred_b_safe=pred_b_safe,
            possible_b=possible_b,
            apply_conformal_mask=apply_conformal_mask,
            set_size_b=set_size_b,
            covered_mask_a=covered_mask_a,
            covered_mask_b=covered_mask_b,
            head_a_label_map=HEAD_A_LABEL_MAP,
            head_b_label_map=HEAD_B_LABEL_MAP,
        )

        timing_row = TimingRow(
            wsi_id=wsi_id,
            num_tiles=num_tiles,
            num_batches=num_batches,
            elapsed_s=elapsed_s,
            elapsed_min=elapsed_s / 60.0,
            tiles_per_s=(num_tiles / elapsed_s) if elapsed_s > 0 else 0.0,
            batches_per_s=(num_batches / elapsed_s) if elapsed_s > 0 else 0.0,
        )

        return SlideOutputs(
            wsi_id=wsi_id,
            avg_logits_a=avg_logits_a,
            avg_logits_b=avg_logits_b.cpu(),
            pred_a=pred_a,
            pred_b_argmax=pred_b_argmax,
            pred_b_safe=pred_b_safe,
            possible_b=possible_b,
            set_size_b=set_size_b,
            apply_conformal_mask=apply_conformal_mask,
            empty_conformal_set_mask=empty_conformal_set_mask,
            multi_class_conformal_set_mask=multi_class_conformal_set_mask,
            covered_mask_a=covered_mask_a,
            covered_mask_b=covered_mask_b,
            valid_fg_mask_b=valid_fg_mask_b,
            area_stats_b=area_stats_b,
            slide_stat_rows=slide_stat_rows,
            timing_row=timing_row,
            geometry=geometry,
        )
    finally:
        wsi.close()


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

    runtime_cfg = build_runtime_config(
        batch_size=batch_size,
        num_workers=num_workers,
        gpu_id=gpu_id,
        output_target_mpp=output_target_mpp,
        no_autocast=no_autocast,
        autocast_dtype=autocast_dtype,
        save_pt=save_pt,
        save_argmax_png=save_argmax_png,
        save_safe_png=save_safe_png,
    )

    model, img_size, conformal_cfg = load_model_and_conformal(model_dir.resolve())
    model = model.to(runtime_cfg.device).eval()

    thresholds_b = get_head_b_thresholds(
        conformal_cfg=conformal_cfg,
        num_classes=len(HEAD_B_LABEL_MAP),
        bg_idx=0,
    )
    if thresholds_b is not None:
        thresholds_b = thresholds_b.to(runtime_cfg.device)

    tile_transform = build_tile_transform(normalize_img_size(img_size))

    tiling_store = build_tiling_store_from_dir(
        slides_root=slides_dir,
        root_dir=tiles_dir,
    )

    slide_paths = sorted(slides_dir.glob("*.tif"))
    if not slide_paths:
        raise FileNotFoundError(f"No .tif slides found in {slides_dir}")

    for wsi_path in slide_paths:
        outputs = process_single_slide(
            wsi_path=wsi_path,
            tiling_store=tiling_store,
            model=model,
            thresholds_b=thresholds_b,
            tile_transform=tile_transform,
            runtime_cfg=runtime_cfg,
        )

        write_slide_outputs(
            output_dir=output_dir,
            outputs=outputs,
            output_target_mpp=runtime_cfg.output_target_mpp,
            save_pt=runtime_cfg.save_pt,
            save_argmax_png=runtime_cfg.save_argmax_png,
            save_safe_png=runtime_cfg.save_safe_png,
        )

        print(
            f"[OK] {outputs.wsi_id}: "
            f"{outputs.timing_row.elapsed_s:.2f}s "
            f"({outputs.timing_row.tiles_per_s:.2f} tiles/s)"
        )


if __name__ == "__main__":
    main()
