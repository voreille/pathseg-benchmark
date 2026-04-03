from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from pathseg.datasets.lightning_data_module import LightningDataModule


class CenterPad(torch.nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
    ):
        super().__init__()

        self.img_size = img_size
        self.center_crop = T.CenterCrop(img_size)

    def pad(self, img):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding, padding_mode="edge")

        return img

    def forward(self, img):
        img = self.pad(img)
        return self.center_crop(img)


class LungHist700ROIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        transform=None,
        return_image_id: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.transform = transform
        self.return_image_id = bool(return_image_id)

        required_cols = {"sample_id", "image_relpath", "label_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_id = str(row["sample_id"])
        image_path = self.root / row["image_relpath"]
        label = int(row["label_id"])

        image = tv_tensors.Image(Image.open(image_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)

        if self.return_image_id:
            return image, label, image_id
        return image, label


class LungHist700SupportDataset(Dataset):
    """
    Returns one episodic support batch:
        images: list[Tensor]
        labels: list[int]
        image_ids: list[str]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        shots_per_class: int,
        transform=None,
        fixed_seed: int = 0,
        fixed: bool = False,
        norm_01: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.shots_per_class = int(shots_per_class)
        self.transform = transform
        self.fixed_seed = int(fixed_seed)
        self.fixed = bool(fixed)
        self.norm_01 = bool(norm_01)

        required_cols = {"sample_id", "image_relpath", "label_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

        self.class_ids = sorted(self.df["label_id"].unique().tolist())

        if self.fixed:
            self.fixed_df = self._sample_support_df(random_state=self.fixed_seed)
        else:
            self.fixed_df = None

    def _sample_support_df(self, random_state: int | None = None) -> pd.DataFrame:
        chunks = []
        for class_id in self.class_ids:
            df_c = self.df[self.df["label_id"] == class_id]
            if len(df_c) == 0:
                continue
            n = min(self.shots_per_class, len(df_c))
            chunks.append(df_c.sample(n=n, random_state=random_state))
        if not chunks:
            raise RuntimeError("Could not sample support set: no data available.")
        return pd.concat(chunks, axis=0).reset_index(drop=True)

    def __len__(self) -> int:
        # One episode per iteration
        return 10**9 if not self.fixed else 1

    def __getitem__(self, idx: int):
        if self.fixed:
            df_support = self.fixed_df
        else:
            df_support = self._sample_support_df(random_state=None)

        images: list[torch.Tensor] = []
        labels: list[int] = []
        image_ids: list[str] = []

        for _, row in df_support.iterrows():
            image_id = str(row["sample_id"])
            image_path = self.root / row["image_relpath"]
            label = int(row["label_id"])

            image = tv_tensors.Image(Image.open(image_path).convert("RGB"))
            if self.transform is not None:
                image = self.transform(image)

            if self.norm_01:
                image = image / 255.0

            images.append(image)
            labels.append(label)
            image_ids.append(image_id)

        return images, labels, image_ids


class LungHist700PrototypeDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        metadata_csv: str | Path,
        batch_size: int = 8,
        num_workers: int = 0,
        img_size: tuple[int, int] = (896, 896),
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        transforms=None,
        val_transforms=None,
        predict_split: str = "test",
        support_split: str = "train",
        shots_per_class: int = 5,
        support_fixed_seed: int = 0,
        support_fixed_for_predict: bool = True,
        exclude_support_from_query: bool = True,
        ignore_idx: int = 255,
        superclass: str | None = None,
    ) -> None:
        super().__init__(
            root=str(root),
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=0,
            num_metrics=0,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
        )

        self.root = Path(root)
        self.metadata_csv = Path(metadata_csv)

        self.transforms = transforms
        self.val_transforms = (
            val_transforms if val_transforms is not None else transforms
        )

        self.predict_split = str(predict_split)
        self.support_split = str(support_split)
        self.shots_per_class = int(shots_per_class)
        self.support_fixed_seed = int(support_fixed_seed)
        self.support_fixed_for_predict = bool(support_fixed_for_predict)
        self.exclude_support_from_query = bool(exclude_support_from_query)
        self.superclass = str(superclass) if superclass is not None else None

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        }
        self.support_dataloader_kwargs = {
            "batch_size": 1,
            "num_workers": min(2, num_workers) if num_workers > 0 else 0,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        }

        self.save_hyperparameters(ignore=["transforms", "val_transforms"])

    @staticmethod
    def predict_collate(batch):
        images, labels, image_ids = zip(*batch)
        return images, labels, image_ids

    @staticmethod
    def support_collate(batch):
        images, labels, image_ids = batch[0]
        return images, labels, image_ids

    def _read_metadata(self) -> pd.DataFrame:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"metadata_csv not found: {self.metadata_csv}")
        df = pd.read_csv(self.metadata_csv)

        required_cols = {"sample_id", "image_relpath", "label_id", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

        return df

    def setup(self, stage: str | None = None):
        df = self._read_metadata()

        if self.superclass is not None:
            if "superclass" not in df.columns:
                raise ValueError(
                    f"superclass column not found in metadata, required for superclass={self.superclass}"
                )
            df = df[df["superclass"] == self.superclass].reset_index(drop=True)

        support_df = df[df["split"] == self.support_split].reset_index(drop=True)
        query_df = df[df["split"] == self.predict_split].reset_index(drop=True)

        if len(support_df) == 0:
            raise RuntimeError(f"No rows found for support_split={self.support_split}")
        if len(query_df) == 0:
            raise RuntimeError(f"No rows found for predict_split={self.predict_split}")

        self.support_dataset = LungHist700SupportDataset(
            df=support_df,
            root=self.root,
            shots_per_class=self.shots_per_class,
            transform=CenterPad(self.img_size),
            fixed_seed=self.support_fixed_seed,
            fixed=self.support_fixed_for_predict,
        )

        # optionally remove fixed support IDs from query set if same split is used
        if self.exclude_support_from_query and self.support_split == self.predict_split:
            fixed_support_df = self.support_dataset.fixed_df
            if fixed_support_df is None:
                raise RuntimeError(
                    "exclude_support_from_query=True requires fixed support sampling."
                )
            support_ids = set(fixed_support_df["sample_id"].astype(str).tolist())
            query_df = query_df[
                ~query_df["sample_id"].astype(str).isin(support_ids)
            ].reset_index(drop=True)

        self.predict_dataset = LungHist700ROIDataset(
            df=query_df,
            root=self.root,
            transform=self.val_transforms,
            return_image_id=True,
        )

        rank_zero_info(
            f"[LungHist700PrototypeDataModule] support={len(support_df)} query={len(query_df)}"
        )

        return self

    def support_dataloader(self):
        return DataLoader(
            self.support_dataset,
            shuffle=False,
            collate_fn=self.support_collate,
            **self.support_dataloader_kwargs,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            collate_fn=self.predict_collate,
            **self.dataloader_kwargs,
        )

    def val_support_dataloader(self):
        return self.support_dataloader()

    def test_support_dataloader(self):
        return self.support_dataloader()

    def test_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            collate_fn=self.predict_collate,
            **self.dataloader_kwargs,
        )
