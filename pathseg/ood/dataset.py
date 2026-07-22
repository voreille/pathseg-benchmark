from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import tv_tensors


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for ANORAK images and masks (Mask2Former-style targets).
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        masks_directory,
        ignore_idx=255,
        return_background=True,
        transforms=None,
        class_mapping=None,
        return_image_id: bool = False,  # NEW
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transforms = transforms
        self.ignore_idx = ignore_idx
        self.return_background = return_background
        self.class_mapping = class_mapping
        self.return_image_id = return_image_id  # NEW

        self.data_df = self._get_data_df()

    def _get_data_df(self):
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

        if len(masks) > 0:
            target = {
                "masks": tv_tensors.Mask(torch.stack(masks)),
                "labels": torch.cat(labels),
            }
        else:
            target = {
                "masks": tv_tensors.Mask(
                    torch.zeros((0, *mask.shape), dtype=torch.bool)
                ),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.return_image_id:
            return image, target, image_id

        return image, target


class InstanceDataset(torch.utils.data.Dataset):
    """
    Dataset that loads images + precomputed instance targets (Mask2Former-style).

    Expected target formats:

    NPZ:
      masks: uint8 array (N, H, W) with values {0,1}
      labels: int64 array (N,)

    PT:
      torch dict {"masks": BoolTensor (N,H,W), "labels": LongTensor (N,)}
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        instances_directory,
        transforms=None,
        return_image_id: bool = False,
        extension: str = "pt",  # "npz" or "pt"
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.instances_directory = Path(instances_directory)
        self.transforms = transforms
        self.return_image_id = return_image_id
        self.extension = extension.lower()

        if self.extension not in {"npz", "pt"}:
            raise ValueError(f"extension must be 'npz' or 'pt', got: {extension}")

        self.data_df = self._get_data_df()

    def _get_data_df(self) -> pd.DataFrame:
        data_list = []
        for image_id in self.image_ids:
            matches = list(self.images_directory.glob(f"{image_id}.*"))
            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No image found for id={image_id} in {self.images_directory}"
                )
            if len(matches) > 1:
                # If you have multiple extensions per id, pick the first deterministically.
                matches = sorted(matches)
            image_path = matches[0]

            instance_path = self.instances_directory / f"{image_id}.{self.extension}"
            if not instance_path.exists():
                raise FileNotFoundError(
                    f"Missing instance target for id={image_id}: {instance_path}"
                )

            data_list.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "instance_path": instance_path,
                }
            )
        df = pd.DataFrame(data_list)
        return df.set_index("image_id")

    def __len__(self):
        return len(self.image_ids)

    def _load_target(self, instance_path: Path) -> Dict[str, Any]:
        if self.extension == "npz":
            z = np.load(instance_path)
            masks = z["masks"]  # (N,H,W) uint8
            labels = z["labels"]  # (N,) int64

            if masks.ndim != 3:
                raise ValueError(
                    f"{instance_path}: masks must have shape (N,H,W), got {masks.shape}"
                )
            if labels.ndim != 1:
                raise ValueError(
                    f"{instance_path}: labels must have shape (N,), got {labels.shape}"
                )
            if masks.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{instance_path}: mismatch N between masks and labels: "
                    f"{masks.shape[0]} vs {labels.shape[0]}"
                )

            masks_t = torch.from_numpy(masks.astype(np.bool_))  # BoolTensor
            labels_t = torch.from_numpy(labels.astype(np.int64))  # LongTensor

        else:  # pt
            d = torch.load(instance_path, map_location="cpu")
            if not isinstance(d, dict) or "masks" not in d or "labels" not in d:
                raise ValueError(
                    f"{instance_path}: expected dict with keys 'masks' and 'labels'"
                )

            masks_t = d["masks"].to(torch.bool)
            labels_t = d["labels"].to(torch.int64)

            if masks_t.ndim != 3:
                raise ValueError(
                    f"{instance_path}: masks must have shape (N,H,W), got {tuple(masks_t.shape)}"
                )
            if labels_t.ndim != 1:
                raise ValueError(
                    f"{instance_path}: labels must have shape (N,), got {tuple(labels_t.shape)}"
                )
            if masks_t.shape[0] != labels_t.shape[0]:
                raise ValueError(
                    f"{instance_path}: mismatch N between masks and labels: "
                    f"{masks_t.shape[0]} vs {labels_t.shape[0]}"
                )

        target = {
            "masks": tv_tensors.Mask(masks_t),
            "labels": labels_t,
        }
        return target

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        row = self.data_df.loc[image_id]

        image = tv_tensors.Image(Image.open(row["image_path"]).convert("RGB"))
        target = self._load_target(row["instance_path"])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.return_image_id:
            return image, target, image_id

        return image, target
