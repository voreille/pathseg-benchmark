from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset as TorchDataset,
    WeightedRandomSampler,
)

from pathseg.datasets.dataset import Dataset as BaseDataset
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.datasets.transforms import CustomTransforms


class WrapWithSource(TorchDataset):
    """
    Wrap a base dataset and add a per-sample source_id.

    Expected base dataset outputs:
      - (image, target) OR
      - (image, target, image_id)

    Wrapper output:
      - (image, target, source_id, image_id)
    """

    def __init__(self, base: TorchDataset, source_id: int):
        self.base = base
        self.source_id = int(source_id)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]

        if isinstance(item, (tuple, list)) and len(item) == 2:
            img, target = item
            image_id = str(idx)
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            img, target, image_id = item
            image_id = str(image_id)
        else:
            raise TypeError(
                "Base dataset must return (image, target) or (image, target, image_id)."
            )

        return img, target, self.source_id, image_id


class MultiTaskConcatDataModule(LightningDataModule):
    """
    Multi-dataset datamodule that mixes datasets within a batch using ConcatDataset.

    Training uses a WeightedRandomSampler so epoch length is controlled by
    num_iterations_per_epoch rather than dataset repetition.
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        num_workers: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        val_batch_size: int = 1,
        scale_range: tuple[float, float] = (0.8, 1.2),
        ignore_idx: int = 255,
        prefetch_factor: int = 2,
        transforms: Optional[nn.Module] = None,
        val_transforms: Optional[nn.Module] = None,
        return_background_mask: bool = True,
        num_iterations_per_epoch: int = 1500,
    ) -> None:
        super().__init__(
            root="",
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=0,
            num_metrics=0,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
        )

        if not datasets or len(datasets) < 2:
            raise ValueError("datasets must be a list with at least two dataset dicts.")

        self.datasets_cfg = datasets
        self.return_background_mask = bool(return_background_mask)

        self.num_iterations_per_epoch = int(num_iterations_per_epoch)
        if self.num_iterations_per_epoch <= 0:
            raise ValueError("num_iterations_per_epoch must be > 0")

        self.num_samples_per_epoch = self.num_iterations_per_epoch * batch_size

        self.val_dataloader_kwargs = self.dataloader_kwargs.copy()
        self.val_dataloader_kwargs["batch_size"] = val_batch_size

        self.save_hyperparameters(ignore=["transforms", "val_transforms"])

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = CustomTransforms(
                img_size=img_size, scale_range=scale_range
            )

        self.val_transforms = val_transforms

        rank_zero_info(f"[MultiTaskConcatDataModule] batch_size={batch_size}")
        rank_zero_info(
            f"[MultiTaskConcatDataModule] num_iterations_per_epoch={self.num_iterations_per_epoch}"
        )
        for d in self.datasets_cfg:
            rank_zero_info(
                f"[MultiTaskConcatDataModule] dataset={d.get('name', '?')} "
                f"root={d['root']} source_id={d.get('source_id')} "
                f"sampling_weight={d.get('sampling_weight', 1.0)}"
            )

    @staticmethod
    def _read_split_csv(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise ValueError(f".csv for split not found: {csv_path}")
        return pd.read_csv(csv_path)

    def _get_split_ids(
        self,
        df: pd.DataFrame,
        fold: int = 0,
    ) -> Tuple[List[str], List[str], List[str]]:
        m_test = df["split"] == "test"
        m_val = df["validation_fold"] == "fold" + str(fold)
        m_train = (df["split"] == "train") & (~m_val)

        return (
            df[m_train]["sample_id"].tolist(),
            df[m_val]["sample_id"].tolist(),
            df[m_test]["sample_id"].tolist(),
        )

    def _make_base_dataset(
        self,
        *,
        ids: Sequence[str],
        images_dir: Path,
        masks_dir: Path,
        stage: str,
    ) -> TorchDataset:
        if stage in {"fit", "validate", "test"}:
            return BaseDataset(
                ids,
                images_dir,
                masks_dir,
                transforms=(self.transforms if stage == "fit" else self.val_transforms),
                ignore_idx=self.ignore_idx,
                return_background=self.return_background_mask,
                return_image_id=True,
            )

        if stage == "predict":
            return BaseDataset(
                ids,
                images_dir,
                masks_dir,
                ignore_idx=self.ignore_idx,
                return_image_id=True,
            )

        raise ValueError(f"Unknown stage: {stage}")

    def _build_train_sampler(self) -> WeightedRandomSampler:
        if not isinstance(self.train_dataset, ConcatDataset):
            raise TypeError("train_dataset must be a ConcatDataset.")

        sample_weights: list[float] = []

        for dcfg, subdataset in zip(self.datasets_cfg, self.train_dataset.datasets):
            ds_weight = float(dcfg.get("sampling_weight", 1.0))
            if ds_weight <= 0:
                raise ValueError(
                    f"sampling_weight must be > 0 for dataset {dcfg.get('name', '?')}"
                )

            n = len(subdataset)
            if n == 0:
                raise ValueError(
                    f"Dataset {dcfg.get('name', '?')} has zero training samples."
                )

            # Equal total probability mass per dataset if all sampling_weight=1.0
            per_sample_weight = ds_weight / n
            sample_weights.extend([per_sample_weight] * n)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.num_samples_per_epoch,
            replacement=True,
        )

    @staticmethod
    def train_collate(batch):
        imgs, targets, source_ids, image_ids = [], [], [], []

        for img, target, source_id, image_id in batch:
            imgs.append(img)
            targets.append(target)
            source_ids.append(int(source_id))
            image_ids.append(str(image_id))

        return (
            torch.stack(imgs),
            targets,
            torch.tensor(source_ids, dtype=torch.long),
            image_ids,
        )

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))

    def setup(self, stage: Union[str, None] = None) -> "MultiTaskConcatDataModule":
        train_wrapped = []
        self.val_wrapped = []
        self.test_wrapped = []
        self.predict_wrapped = []
        self.predict_splits = []

        for dcfg in self.datasets_cfg:
            name = dcfg.get("name", f"ds{dcfg.get('source_id', 0)}")
            root = Path(dcfg["root"])
            images_dir = root / dcfg.get("images_subdir", "images")
            masks_dir = root / dcfg.get("masks_subdir", "masks")
            split_csv = root / dcfg.get("split_csv", "split.csv")
            source_id = int(dcfg.get("source_id", 0))
            fold = int(dcfg.get("fold", 0))

            df = self._read_split_csv(split_csv)
            train_ids, val_ids, test_ids = self._get_split_ids(df, fold=fold)

            if stage in ("fit", "validate", None):
                base_train = self._make_base_dataset(
                    ids=train_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="fit",
                )
                train_wrapped.append(WrapWithSource(base_train, source_id=source_id))

                base_val = self._make_base_dataset(
                    ids=val_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="validate",
                )
                self.val_wrapped.append(
                    (name, WrapWithSource(base_val, source_id=source_id))
                )

            if stage in ("test", None):
                base_test = self._make_base_dataset(
                    ids=test_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="test",
                )
                self.test_wrapped.append(
                    (name, WrapWithSource(base_test, source_id=source_id))
                )

            if stage in ("predict", None):
                base_val_p = self._make_base_dataset(
                    ids=val_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="predict",
                )
                self.predict_wrapped.append(
                    (f"{name}_val", WrapWithSource(base_val_p, source_id=source_id))
                )

        if stage in ("fit", "validate", None):
            self.train_dataset = ConcatDataset(train_wrapped)

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            sampler=self._build_train_sampler(),
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return [
            DataLoader(ds, collate_fn=self.eval_collate, **self.val_dataloader_kwargs)
            for _, ds in self.val_wrapped
        ]

    def test_dataloader(self):
        return [
            DataLoader(ds, collate_fn=self.eval_collate, **self.val_dataloader_kwargs)
            for _, ds in self.test_wrapped
        ]

    def predict_dataloader(self):
        if not getattr(self, "predict_wrapped", None):
            self.setup(stage="predict")

        loaders, splits = [], []
        for split_name, ds in self.predict_wrapped:
            loaders.append(
                DataLoader(
                    ds, collate_fn=self.eval_collate, **self.val_dataloader_kwargs
                )
            )
            splits.append(split_name)

        self.predict_splits = splits
        return loaders