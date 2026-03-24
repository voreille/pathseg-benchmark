from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Callable

import pandas as pd
import torch
from PIL import Image
from torchvision import tv_tensors

_TILE_COLUMNS = {"origin_x", "origin_y", "w", "h"}


def _build_crop_from_row(r: pd.Series) -> tuple[int, int, int, int]:
    if not _TILE_COLUMNS.issubset(r.index):
        raise ValueError(
            f"Row is missing tile columns: expected {_TILE_COLUMNS}, got {set(r.index)}"
        )
    x = int(r["origin_x"])
    y = int(r["origin_y"])
    w = int(r["w"])
    h = int(r["h"])
    return (x, y, x + w, y + h)


class SupportDataset(torch.utils.data.Dataset):
    """
    Returns one support set per __getitem__:
      images:  list[Tensor[C,H,W]]
      targets: list[dict]

    Modes
    -----
    - random mode:
        sample `shots_per_class` rows independently for each class
    - fixed mode:
        use rows from `fixed_df` exactly as provided
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        root_dir: Path,
        shots_per_class: int,
        transform: Callable | None = None,
        *,
        class_ids: Optional[Sequence[int]] = None,
        fixed_df: Optional[pd.DataFrame] = None,
        ignore_idx: int = 255,
        return_background: bool = True,
        norm_01: bool = True,
    ):
        self.df = index_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.shots_per_class = int(shots_per_class)
        self.transform = transform
        self.fixed_df = None if fixed_df is None else fixed_df.reset_index(drop=True)
        self.ignore_idx = int(ignore_idx)
        self.return_background = bool(return_background)
        self.norm_01 = bool(norm_01)

        if class_ids is None:
            self.class_ids = sorted(int(x) for x in self.df["dataset_class_id"].unique())
        else:
            self.class_ids = [int(x) for x in class_ids]

        if self.fixed_df is None:
            self.rows_by_class = {
                cls: df_cls.reset_index(drop=True)
                for cls, df_cls in self.df.groupby("dataset_class_id", sort=True)
            }
            missing = [cls for cls in self.class_ids if cls not in self.rows_by_class]
            if missing:
                raise ValueError(f"No support rows found for class_ids={missing}")
        else:
            self.rows_by_class = None

    def __len__(self) -> int:
        return 1 if self.fixed_df is not None else 1_000_000

    def _target_from_mask(self, mask: torch.Tensor) -> Dict[str, Any]:
        mask = torch.squeeze(mask, 0)

        unique_labels = torch.unique(mask)
        masks, labels = [], []

        for label_id in unique_labels:
            class_id = int(label_id.item())
            if class_id != self.ignore_idx and (self.return_background or class_id != 0):
                masks.append(mask == label_id)
                labels.append(class_id)

        if masks:
            return {
                "masks": tv_tensors.Mask(torch.stack(masks)),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

        return {
            "masks": tv_tensors.Mask(torch.zeros((0, *mask.shape), dtype=torch.bool)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    def _load_one(self, row: pd.Series):
        img_path = self.root_dir / row["image_relpath"]
        mask_path = self.root_dir / row["mask_relpath"]

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        mask = tv_tensors.Mask(Image.open(mask_path).convert("L"))
        mask = torch.squeeze(mask, 0)

        x1, y1, x2, y2 = _build_crop_from_row(row)
        img = img[:, y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        target = self._target_from_mask(mask)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def _sample_rows(self) -> pd.DataFrame:
        if self.fixed_df is not None:
            return self.fixed_df

        sampled_parts = []
        for cls in self.class_ids:
            df_cls = self.rows_by_class[cls]
            n = min(self.shots_per_class, len(df_cls))
            sampled_parts.append(df_cls.sample(n=n))

        return pd.concat(sampled_parts, axis=0, ignore_index=True)

    def __getitem__(self, idx):
        sampled = self._sample_rows()

        images = []
        targets = []

        for _, row in sampled.iterrows():
            img, target = self._load_one(row)
            if self.norm_01:
                img = img / 255.0
            images.append(img)
            targets.append(target)

        return images, targets