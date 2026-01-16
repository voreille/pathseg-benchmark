from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import DataLoader

from pathseg.datasets.dataset import (
    Dataset,
    PredictDataset,
)
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.datasets.transforms import CustomTransforms
from pathseg.datasets.utils import RepeatDataset


class IGNITE(LightningDataModule):
    def __init__(
        self,
        root,
        num_workers: int = 0,
        fold: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_classes: int = 16,
        num_metrics: int = 1,
        scale_range=(0.8, 1.2),
        ignore_idx: int = 255,
        overwrite_root: Optional[str] = None,
        prefetch_factor: int = 2,
        transforms: Optional[nn.Module] = None,
        epoch_repeat: int = 1,
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
        if overwrite_root:
            root = overwrite_root

        root_dir = Path(root)
        self.fold = fold
        rank_zero_info(f"[IGNITE] Initializing datamodule with fold = {self.fold}")
        rank_zero_info(
            f"[IGNITE] Initializing datamodule with batch_size = {batch_size}"
        )

        self.images_dir = root_dir / "images"
        self.masks_dir = root_dir / "masks_semantic"

        data_overview_csv = root_dir / "metadata.csv"

        if not data_overview_csv.exists():
            raise ValueError("data_overview.csv not found in the root directory")

        self.data_overview_df = pd.read_csv(data_overview_csv)

        self.epoch_repeat = epoch_repeat
        self.save_hyperparameters(ignore=["transforms"])

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = CustomTransforms(
                img_size=img_size,
                scale_range=scale_range,
            )

    def _get_split_ids(self):
        m_test = self.data_overview_df["split"] == "test"
        m_val = self.data_overview_df["validation_fold"] == "fold" + str(self.fold)
        m_train = (self.data_overview_df["split"] == "train") & (~m_val)

        return (
            self.data_overview_df[m_train]["sample_id"].tolist(),
            self.data_overview_df[m_val]["sample_id"].tolist(),
            self.data_overview_df[m_test]["sample_id"].tolist(),
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
            self.val_dataset = Dataset(val_ids, self.images_dir, self.masks_dir)

            # compute once per fold for training
            # self.class_weights = self.compute_class_weights()
            self.class_weights = torch.ones(self.num_classes, dtype=torch.float32)

        if stage in ("test", None):
            self.test_dataset = Dataset(test_ids, self.images_dir, self.masks_dir)

        if stage in ("predict", None):
            # self.val_dataset = PredictDataset(val_ids, self.images_dir, self.masks_dir)
            self.test_dataset = PredictDataset(
                test_ids, self.images_dir, self.masks_dir
            )

        return self

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

    def predict_dataloader(self):
        # make sure datasets exist
        if not hasattr(self, "val_dataset") or not hasattr(self, "test_dataset"):
            self.setup(stage="predict")

        loaders, splits = [], []

        if getattr(self, "val_dataset", None) is not None:
            loaders.append(self.val_dataloader())
            splits.append("val")

        if getattr(self, "test_dataset", None) is not None:
            loaders.append(self.test_dataloader())
            splits.append("test")

        # store the mapping in the datamodule
        self.predict_splits = splits
        return loaders
