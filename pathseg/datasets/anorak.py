from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from pathseg.datasets.dataset import Dataset
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.datasets.transforms import CustomTransforms
from pathseg.datasets.utils import RepeatDataset

# -----------------------------
# Deterministic crop view for Mask2Former-style targets
# Base dataset must return: (image, target, image_id)
# target = {"masks": (N,H,W) bool, "labels": (N,) int64}
# -----------------------------


@dataclass(frozen=True)
class CropMeta:
    y0: int
    x0: int
    y1: int
    x1: int
    crop_idx: int
    orig_h: int
    orig_w: int


def _crop_box_center(h: int, w: int, ch: int, cw: int) -> Tuple[int, int, int, int]:
    ch = min(ch, h)
    cw = min(cw, w)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return y0, x0, y0 + ch, x0 + cw


def _crop_box_grid(
    h: int, w: int, ch: int, cw: int, crop_idx: int, grid_n: int
) -> Tuple[int, int, int, int]:
    ch = min(ch, h)
    cw = min(cw, w)
    gy = crop_idx // grid_n
    gx = crop_idx % grid_n
    max_y0 = max(0, h - ch)
    max_x0 = max(0, w - cw)
    y0 = int(round(gy * (max_y0 / max(1, grid_n - 1))))
    x0 = int(round(gx * (max_x0 / max(1, grid_n - 1))))
    return y0, x0, y0 + ch, x0 + cw


class Mask2FormerCropView(TorchDataset):
    """
    Wraps a base dataset returning (image, target, image_id) into fixed-size crops:
      -> (crop_image, crop_target, image_id, CropMeta)

    Crops are generated lazily and deterministically.
    """

    def __init__(
        self,
        base: TorchDataset,
        crop_size: Tuple[int, int],
        *,
        mode: Literal["center", "grid"] = "center",
        grid_n: int = 2,
        drop_empty: bool = True,
    ):
        self.base = base
        self.crop_h, self.crop_w = crop_size
        self.mode = mode
        self.grid_n = grid_n
        self.drop_empty = drop_empty

        self.crops_per_image = 1 if mode == "center" else grid_n * grid_n

    @staticmethod
    def _center_pad_to_min_size(
        image: torch.Tensor,
        masks: torch.Tensor,
        min_h: int,
        min_w: int,
    ):
        """
        image: (C,H,W) or (...,H,W) tensor-like
        masks: (N,H,W) bool/uint8 tensor-like
        Pads to at least (min_h, min_w) with:
          - replicate for image
          - constant 0 for masks
        Padding is centered (split top/bottom and left/right).
        """
        h, w = int(image.shape[-2]), int(image.shape[-1])
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)

        if pad_h == 0 and pad_w == 0:
            return image, masks

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects pads as (left, right, top, bottom)
        image_p = F.pad(
            image, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
        )

        # masks: constant 0 padding
        # ensure masks is a tensor (tv_tensors.Mask is tensor-like; F.pad should still work)
        masks_p = F.pad(
            masks, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )

        return image_p, masks_p

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_image

    def __getitem__(self, idx: int):
        img_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image

        image, target, image_id = self.base[img_idx]

        # extract masks/labels
        masks = target["masks"]
        labels = target["labels"]

        image, masks = self._center_pad_to_min_size(
            image=image,
            masks=masks,
            min_h=self.crop_h,
            min_w=self.crop_w,
        )

        h, w = int(image.shape[-2]), int(image.shape[-1])

        if self.mode == "center":
            y0, x0, y1, x1 = _crop_box_center(h, w, self.crop_h, self.crop_w)
        else:
            y0, x0, y1, x1 = _crop_box_grid(
                h, w, self.crop_h, self.crop_w, crop_idx, self.grid_n
            )

        img_crop = image[..., y0:y1, x0:x1]
        masks = masks[..., y0:y1, x0:x1]

        if self.drop_empty and masks.numel() > 0:
            keep = masks.flatten(1).any(dim=1)
            masks = masks[keep]
            labels = labels[keep]

        target_crop = {"masks": masks, "labels": labels}
        meta = CropMeta(
            y0=y0, x0=x0, y1=y1, x1=x1, crop_idx=crop_idx, orig_h=h, orig_w=w
        )

        return img_crop, target_crop, image_id, meta


# -----------------------------
# ANORAK DataModule
# -----------------------------


class ANORAK(LightningDataModule):
    def __init__(
        self,
        root,
        num_workers: int = 0,
        fold: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_classes: int = 7,
        num_metrics: int = 1,
        scale_range=(0.8, 1.2),
        ignore_idx: int = 255,
        prefetch_factor: int = 2,
        transforms: Optional[nn.Module] = None,
        epoch_repeat: int = 1,
        # Predict controls (kept minimal)
        predict_split: Literal["val", "test", "val+test"] = "test",
        predict_crop_size: Optional[tuple[int, int]] = None,  # fixed-size crops for viz
        predict_grid_n: int = 2,  # default grid crops per image = 4
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
        )

        root_dir = Path(root)
        self.fold = fold
        rank_zero_info(f"[ANORAK] fold={self.fold}, batch_size={batch_size}")

        self.images_dir = root_dir / "images"
        self.masks_dir = root_dir / "masks_semantic"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]

        self.epoch_repeat = epoch_repeat

        self.predict_split = predict_split
        self.predict_crop_size = predict_crop_size or img_size
        self.predict_grid_n = predict_grid_n

        self.save_hyperparameters(ignore=["transforms"])
        self.transforms = transforms or CustomTransforms(
            img_size=img_size, scale_range=scale_range
        )

    def _get_split_ids(self):
        return (
            self.split_df[self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]["image_id"].unique().tolist(),
        )

    def compute_class_weights(self):
        from pathseg.datasets.stats import compute_class_weights_from_ids

        train_ids, _, _ = self._get_split_ids()
        return compute_class_weights_from_ids(
            train_ids,
            self.masks_dir,
            self.num_classes,
            ignore_idx=self.ignore_idx,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        train_ids, val_ids, test_ids = self._get_split_ids()

        if stage in ("fit", "validate", None):
            self.train_dataset = Dataset(
                train_ids, self.images_dir, self.masks_dir, transforms=self.transforms
            )
            # deterministic val: no random-scale/crop transforms
            self.val_dataset = Dataset(val_ids, self.images_dir, self.masks_dir)
            self.class_weights = self.compute_class_weights()

        if stage in ("test", None):
            self.test_dataset = Dataset(test_ids, self.images_dir, self.masks_dir)

        if stage in ("predict", None):
            ids = []
            if self.predict_split in ("val", "val+test"):
                ids += val_ids
            if self.predict_split in ("test", "val+test"):
                ids += test_ids

            # base predict dataset: full image + GT + image_id, no transforms (clean viz)
            self.predict_base = Dataset(
                ids,
                self.images_dir,
                self.masks_dir,
                transforms=None,
                return_image_id=True,
            )

        return self

    # ---- train/val/test loaders ----

    def train_dataloader(self):
        dataset = self.train_dataset
        if getattr(self, "epoch_repeat", 1) > 1:
            dataset = RepeatDataset(dataset, repeats=self.epoch_repeat)
        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    # ---- predict loaders (three variants) ----

    def predict_full_dataloader(self):
        """Full images (varying sizes). Intended for model window+stitch in predict_step."""
        if not hasattr(self, "predict_base"):
            self.setup("predict")
        return DataLoader(
            self.predict_base,
            collate_fn=self.eval_collate,  # keeps variable sizes as tuples
            **self.dataloader_kwargs,
        )

    def predict_center_crop_dataloader(self):
        """One deterministic center crop per image (fixed size). Great for clean viz."""
        if not hasattr(self, "predict_base"):
            self.setup("predict")
        ds = Mask2FormerCropView(
            base=self.predict_base,
            crop_size=self.predict_crop_size,
            mode="center",
            drop_empty=True,
        )
        return DataLoader(
            ds,
            collate_fn=self.predict_crop_collate,  # stacks fixed crops nicely
            **self.dataloader_kwargs,
        )

    def predict_grid_crop_dataloader(self, grid_n: Optional[int] = None):
        """grid_n x grid_n deterministic crops per image (fixed size)."""
        if not hasattr(self, "predict_base"):
            self.setup("predict")
        n = grid_n or self.predict_grid_n
        ds = Mask2FormerCropView(
            base=self.predict_base,
            crop_size=self.predict_crop_size,
            mode="grid",
            grid_n=n,
            drop_empty=True,
        )
        return DataLoader(
            ds,
            collate_fn=self.predict_crop_collate,
            **self.dataloader_kwargs,
        )

    # Lightning hook: choose a default predict loader
    def predict_dataloader(self):
        return self.predict_full_dataloader()

    @staticmethod
    def predict_crop_collate(batch):
        """
        For crop-view predict samples:
          (img, target, image_id, meta)
        Returns:
          imgs: Tensor[B,C,H,W]
          targets: list[dict]
          image_ids: list[str]
          metas: list[CropMeta]
        """
        imgs, targets, image_ids, metas = [], [], [], []
        for img, target, image_id, meta in batch:
            imgs.append(img)
            targets.append(target)
            image_ids.append(image_id)
            metas.append(meta)
        return torch.stack(imgs), targets, image_ids, metas
