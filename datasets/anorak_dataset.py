from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and masks.
    for the ANORAK data found here https://zenodo.org/records/10016027
    the dataset was preprocessed so to have mask as one int ranging from 0 to 6,
    reprensenting the per-pixel classes.
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        masks_directory,
        ignore_idx=-1,
        return_background=True,
        transforms=None,
        class_mapping=None,
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transforms = transforms
        self.ignore_idx = ignore_idx
        self.data_df = self._get_data_df()
        self.return_background = return_background
        self.class_mapping = class_mapping

    def _get_data_df(self):
        df = pd.DataFrame(columns=["image_id", "image_path", "mask_path"])
        data_list = []
        for image_id in self.image_ids:
            image_path = list(self.images_directory.glob(f"{image_id}.*"))[0]
            mask_path = self.masks_directory / f"{image_id}.png"
            data_list.append(
                {"image_id": image_id, "image_path": image_path, "mask_path": mask_path}
            )
        df = pd.DataFrame(data_list)
        return df.set_index("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.data_df.index[idx]
        image = tv_tensors.Image(
            Image.open(self.data_df.loc[image_id, "image_path"]).convert("RGB")
        )
        mask = tv_tensors.Mask(
            Image.open(self.data_df.loc[image_id, "mask_path"]).convert("L")
        )

        mask = torch.squeeze(mask, 0)

        unique_labels = torch.unique(mask)
        masks, labels = [], []
        for label_id in unique_labels:
            class_id = label_id.item()

            if class_id != self.ignore_idx and (
                self.return_background or class_id != 0
            ):
                masks.append(mask == label_id)
                labels.append(torch.tensor([class_id]))

        target = {}

        if len(masks) > 0:
            target["masks"] = tv_tensors.Mask(torch.stack(masks))
            target["labels"] = torch.cat(labels)
        else:
            # image has only background or ignore regions
            target["masks"] = tv_tensors.Mask(
                torch.zeros((0, *mask.shape), dtype=torch.bool)
            )
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class PredictDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and masks.
    for the ANORAK data found here https://zenodo.org/records/10016027
    the dataset was preprocessed so to have mask as one int ranging from 0 to 6,
    reprensenting the per-pixel classes.
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        masks_directory,
        ignore_idx=-1,
        transforms=None,
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transforms = transforms
        self.ignore_idx = ignore_idx
        self.data_df = self._get_data_df()

    def _get_data_df(self):
        df = pd.DataFrame(columns=["image_id", "image_path", "mask_path"])
        data_list = []
        for image_id in self.image_ids:
            image_path = list(self.images_directory.glob(f"{image_id}.*"))[0]
            mask_path = self.masks_directory / f"{image_id}.png"
            data_list.append(
                {"image_id": image_id, "image_path": image_path, "mask_path": mask_path}
            )
        df = pd.DataFrame(data_list)
        return df.set_index("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.data_df.index[idx]
        image = tv_tensors.Image(
            Image.open(self.data_df.loc[image_id, "image_path"]).convert("RGB")
        )
        mask = tv_tensors.Mask(
            Image.open(self.data_df.loc[image_id, "mask_path"]).convert("L")
        )

        mask = torch.squeeze(mask, 0)

        unique_labels = torch.unique(mask)
        masks, labels = [], []
        for label_id in unique_labels:
            class_id = label_id.item()

            if class_id != self.ignore_idx:
                masks.append(mask == label_id)
                labels.append(torch.tensor([class_id]))

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.cat(labels),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_id


class FewShotSupportDataset(TorchDataset):
    """
    Few-shot dataset that:
      - uses a slice of fewshot_plan.csv (one row per support image),
      - crops each image/mask to the stored bbox (x0, y0, w, h),
      - if the bbox goes out of bounds:
          * image is padded by replicating border pixels (edge mode)
          * mask is padded with ignore_idx
      - then builds instance-like binary masks and labels (Mask2Former-style).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_directory: str | Path,
        masks_directory: str | Path,
        ignore_idx: int = -1,
        return_background: bool = True,
        transforms: Optional[Callable[[Any, dict], tuple[Any, dict]]] = None,
        class_mapping: Optional[dict[int, int]] = None,  # kept for compatibility, unused
    ):
        """
        df: slice of fewshot_plan.csv where support_set>=0 and 1<=k_shot<=K.
            Must contain columns: image_id, x0, y0, w, h.
        """
        self.meta_df = df.reset_index(drop=True).copy()
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.ignore_idx = ignore_idx
        self.return_background = return_background
        self.transforms = transforms
        self.class_mapping = class_mapping

        self.data_df = self._build_index(self.meta_df)
        self.image_ids = self.data_df.index.to_numpy()

    # -------------------------------------------------------------------------
    # Index building
    # -------------------------------------------------------------------------
    def _build_index(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve image/mask paths and store bbox information."""
        records = []
        for _, row in meta_df.iterrows():
            image_id = row["image_id"]

            # same glob logic as your original Dataset
            image_path = list(self.images_directory.glob(f"{image_id}.*"))[0]
            mask_path = self.masks_directory / f"{image_id}.png"

            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "x0": int(row["x0"]),
                    "y0": int(row["y0"]),
                    "w": int(row["w"]),
                    "h": int(row["h"]),
                }
            )

        return pd.DataFrame(records).set_index("image_id")

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        row = self.data_df.loc[image_id]

        image_path: Path = row["image_path"]
        mask_path: Path = row["mask_path"]
        x0, y0, w, h = int(row["x0"]), int(row["y0"]), int(row["w"]), int(row["h"])

        pil_img, pil_mask = self._load_pil(image_path, mask_path)
        image, mask = self._pil_to_tensors(pil_img, pil_mask)

        # pad if bbox goes out of bounds, then crop to bbox
        image, mask = self._pad_and_crop_to_bbox(image, mask, x0, y0, w, h)

        # build Mask2Former-style target (binary instance masks + labels)
        target = self._build_target_from_mask(mask)

        # Apply transforms if any (must expect (image, target))
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    # -------------------------------------------------------------------------
    # Loading helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_pil(image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image]:
        """Load image and mask as PIL images."""
        pil_img = Image.open(image_path).convert("RGB")
        pil_mask = Image.open(mask_path).convert("L")
        return pil_img, pil_mask

    @staticmethod
    def _pil_to_tensors(
        pil_img: Image.Image,
        pil_mask: Image.Image,
    ) -> tuple[tv_tensors.Image, torch.Tensor]:
        """
        Convert PIL image/mask to:
        - image: tv_tensors.Image, CHW float tensor (for v2 transforms)
        - mask: plain HW int64 tensor with class IDs
        """
        image = tv_tensors.Image(pil_img)  # v2-friendly image

        mask_np = np.array(pil_mask, dtype=np.int64)
        mask = torch.from_numpy(mask_np)  # (H, W), int64

        return image, mask

    # -------------------------------------------------------------------------
    # Padding / cropping helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_padding(
        x0: int, y0: int, w: int, h: int, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """
        Compute padding (left, top, right, bottom) so that the bbox (x0, y0, w, h)
        lies fully inside the padded image.
        """
        x1 = x0 + w
        y1 = y0 + h

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - width)
        pad_bottom = max(0, y1 - height)

        return pad_left, pad_top, pad_right, pad_bottom

    def _pad_and_crop_to_bbox(
        self,
        image: tv_tensors.Image,
        mask: torch.Tensor,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> tuple[tv_tensors.Image, torch.Tensor]:
        """
        If the bbox goes out of bounds, pad image & mask, then crop.
        - image: padded with border replication (mode 'edge' in v2)
        - mask: padded with ignore_idx (via torch.nn.functional.pad)
        """
        # mask is HW, image is CHW (internally)
        H, W = mask.shape[-2:]

        pad_left, pad_top, pad_right, pad_bottom = self._compute_padding(
            x0, y0, w, h, width=W, height=H
        )

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padding = [pad_left, pad_top, pad_right, pad_bottom]  # L, T, R, B

            # 1) pad image by replicating border pixels (v2 name: "edge")
            image = F.pad(image, padding, padding_mode="edge")

            # 2) pad mask with ignore_idx (2D tensor, use torch.nn.functional.pad)
            #    F_torch.pad for 2D expects (pad_left, pad_right, pad_top, pad_bottom)
            mask = F_torch.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=self.ignore_idx,
            )

            # shift bbox into padded coordinates
            x0 += pad_left
            y0 += pad_top

        # crop image using torchvision (keeps tv_tensors.Image)
        image = F.crop(image, top=y0, left=x0, height=h, width=w)

        # crop mask by slicing (H, W)
        mask = mask[y0 : y0 + h, x0 : x0 + w]

        # ensure integer labels
        mask = mask.to(torch.int64)

        return image, mask

    # -------------------------------------------------------------------------
    # Target construction (Mask2Former-style)
    # -------------------------------------------------------------------------
    def _build_target_from_mask(self, mask: torch.Tensor) -> dict:
        """
        Build:
          - target["masks"]: stacked binary masks (N_inst, H, W)
          - target["labels"]: class ids (N_inst,)
        Exactly like your original dataset for Mask2Former-style benchmarks.
        """
        unique_labels = torch.unique(mask)

        masks_list = []
        labels_list = []

        for label_id in unique_labels:
            class_id = int(label_id.item())

            if class_id == self.ignore_idx:
                continue
            if not self.return_background and class_id == 0:
                continue

            # keep per-class binary mask
            binary_mask = (mask == label_id)
            masks_list.append(binary_mask)
            labels_list.append(torch.tensor([class_id], dtype=torch.int64))

        target: dict[str, torch.Tensor | tv_tensors.Mask] = {}

        if len(masks_list) > 0:
            stacked_masks = torch.stack(masks_list)  # (N_inst, H, W)
            labels = torch.cat(labels_list)          # (N_inst,)

            target["masks"] = tv_tensors.Mask(stacked_masks)
            target["labels"] = labels
        else:
            # only background or ignore regions
            H, W = mask.shape[-2:]
            target["masks"] = tv_tensors.Mask(
                torch.zeros((0, H, W), dtype=torch.bool)
            )
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return target
