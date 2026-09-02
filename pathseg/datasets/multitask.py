from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
)
from torch.utils.data import (
    Dataset as TorchDataset,
)

from pathseg.datasets.dataset import Dataset as BaseDataset
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.datasets.transforms import CustomTransforms


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    task_name: str
    root: Path
    fold: int 
    images_subdir: str = "images"
    masks_subdir: str = "masks_semantic"
    split_csv: str = "split.csv"
    sampling_weight: float = 1.0
    include_in_train: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

        if not isinstance(self.name, str):
            raise TypeError("Dataset name must be a string.")
        if not self.name.strip():
            raise ValueError("Dataset name cannot be empty.")

        if not isinstance(self.task_name, str):
            raise TypeError(f"Dataset {self.name!r}: task_name must be a string.")
        if not self.task_name.strip():
            raise ValueError(f"Dataset {self.name!r} has an empty task_name.")

        if self.sampling_weight <= 0:
            raise ValueError(
                f"Dataset {self.name!r}: sampling_weight must be positive."
            )
        if self.fold is not None and self.fold < 0:
            raise ValueError(f"Dataset {self.name!r}: fold cannot be negative.")

        for field_name in ("images_subdir", "masks_subdir", "split_csv"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"Dataset {self.name!r}: {field_name} cannot be empty."
                )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> DatasetConfig:
        if not isinstance(values, Mapping):
            raise TypeError(
                "Each dataset configuration must be a mapping, got "
                f"{type(values).__name__}."
            )
        values = dict(values)
        if "sampling_weight" in values:
            values["sampling_weight"] = float(values["sampling_weight"])
        if values.get("fold") is not None:
            values["fold"] = int(values["fold"])
        try:
            return cls(**values)
        except TypeError as error:
            name = values.get("name", "<unnamed>")
            raise ValueError(
                f"Invalid configuration for dataset {name!r}: {error}"
            ) from error


class WrapWithTaskName(TorchDataset):
    """Wrap a dataset and attach its semantic task name to every sample.

    Expected base dataset outputs:
      - (image, target) OR
      - (image, target, image_id)

    Wrapper output:
      - (image, target, task_name, image_id)
    """

    def __init__(self, base: TorchDataset, task_name: str):
        if not task_name:
            raise ValueError("task_name cannot be empty.")
        self.base = base
        self.task_name = task_name

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

        return img, target, self.task_name, image_id


class MultiTaskConcatDataModule(LightningDataModule):
    """Mix datasets and route every sample to a named semantic task.

    Training uses a WeightedRandomSampler so epoch length is controlled by
    ``num_iterations_per_epoch`` rather than dataset repetition. Dataset names
    identify data sources; task names identify decoder heads and losses. Several
    datasets may therefore share one task name.
    """

    def __init__(
        self,
        datasets: list[dict[str, Any]],
        num_workers: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        val_batch_size: int = 1,
        scale_range: tuple[float, float] = (0.8, 1.2),
        ignore_idx: int = 255,
        prefetch_factor: int = 2,
        transforms: nn.Module | None = None,
        val_transforms: nn.Module | None = None,
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

        if not datasets:
            raise ValueError("datasets must contain at least one dataset.")

        self.datasets_cfg = tuple(
            DatasetConfig.from_mapping(config) for config in datasets
        )
        dataset_names = [config.name for config in self.datasets_cfg]

        if len(dataset_names) != len(set(dataset_names)):
            duplicates = {
                name for name in dataset_names if dataset_names.count(name) > 1
            }
            raise ValueError(f"Duplicate dataset names: {sorted(duplicates)}")

        if not any(config.include_in_train for config in self.datasets_cfg):
            raise ValueError("At least one dataset must have include_in_train=true.")

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
            "[MultiTaskConcatDataModule] "
            f"num_iterations_per_epoch={self.num_iterations_per_epoch}"
        )
        for config in self.datasets_cfg:
            rank_zero_info(
                f"[MultiTaskConcatDataModule] dataset={config.name} "
                f"task={config.task_name} root={config.root} "
                f"sampling_weight={config.sampling_weight} "
                f"include_in_train={config.include_in_train}"
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
    ) -> tuple[list[str], list[str], list[str]]:
        required_columns = {"sample_id", "split", "validation_fold"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Split CSV is missing columns: {sorted(missing_columns)}."
            )

        m_test = df["split"] == "test"
        m_val = (df["split"] == "train") & (df["validation_fold"] == f"fold{fold}")
        m_train = (df["split"] == "train") & (~m_val)

        return (
            df.loc[m_train, "sample_id"].astype(str).tolist(),
            df.loc[m_val, "sample_id"].astype(str).tolist(),
            df.loc[m_test, "sample_id"].astype(str).tolist(),
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

        for config, subdataset in zip(
            self.train_configs,
            self.train_dataset.datasets,
            strict=True,
        ):
            n = len(subdataset)
            if n == 0:
                raise ValueError(f"Dataset {config.name!r} has zero training samples.")

            # Equal total probability mass per dataset if all sampling_weight=1.0
            per_sample_weight = config.sampling_weight / n
            sample_weights.extend([per_sample_weight] * n)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.num_samples_per_epoch,
            replacement=True,
        )

    @staticmethod
    def train_collate(batch):
        imgs, targets, task_names, image_ids = [], [], [], []

        for img, target, task_name, image_id in batch:
            imgs.append(img)
            targets.append(target)
            task_names.append(str(task_name))
            image_ids.append(str(image_id))

        return (
            torch.stack(imgs),
            targets,
            tuple(task_names),
            image_ids,
        )

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))

    def setup(self, stage: str | None = None) -> MultiTaskConcatDataModule:
        train_wrapped: list[TorchDataset] = []
        self.train_configs: list[DatasetConfig] = []
        self.val_wrapped = []
        self.test_wrapped = []
        self.predict_wrapped = []
        self.predict_splits = []

        for config in self.datasets_cfg:
            root = config.root
            images_dir = root / config.images_subdir
            masks_dir = root / config.masks_subdir
            split_csv = root / config.split_csv
            fold = config.fold

            df = self._read_split_csv(split_csv)
            train_ids, val_ids, test_ids = self._get_split_ids(
                df,
                fold=fold,
            )

            if stage in ("fit", None) and config.include_in_train:
                base_train = self._make_base_dataset(
                    ids=train_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="fit",
                )
                train_wrapped.append(
                    WrapWithTaskName(base_train, task_name=config.task_name)
                )
                self.train_configs.append(config)

            if stage in ("fit", "validate", None):
                base_val = self._make_base_dataset(
                    ids=val_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="validate",
                )
                self.val_wrapped.append(
                    (
                        config.name,
                        WrapWithTaskName(base_val, task_name=config.task_name),
                    )
                )

            if stage in ("test", None):
                base_test = self._make_base_dataset(
                    ids=test_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="test",
                )
                self.test_wrapped.append(
                    (
                        config.name,
                        WrapWithTaskName(base_test, task_name=config.task_name),
                    )
                )

            if stage in ("predict", None):
                base_val_p = self._make_base_dataset(
                    ids=val_ids,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    stage="predict",
                )
                self.predict_wrapped.append(
                    (
                        f"{config.name}_val",
                        WrapWithTaskName(
                            base_val_p,
                            task_name=config.task_name,
                        ),
                    )
                )

        if stage in ("fit", None):
            if not train_wrapped:
                raise ValueError("No datasets are enabled for training.")
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
