from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset as TorchDataset

from pathseg.datasets.dataset import Dataset as BaseDataset
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.datasets.support_dataset import SupportDataset
from pathseg.datasets.transforms import CenterPad, CustomTransforms


class WrapWithSource(TorchDataset):
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
    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        num_workers: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        scale_range: tuple[float, float] = (0.8, 1.2),
        ignore_idx: int = 255,
        prefetch_factor: int = 2,
        transforms: Optional[nn.Module] = None,
        val_transforms: Optional[nn.Module] = None,
        return_background_mask: bool = True,
        num_iterations_per_epoch: int = 1500,
        support_parquet: str | Path | None = None,
        support_source_id: int | None = None,
        support_shots_per_class: int = 1,
        support_random_seed: int = 0,
        support_min_valid_frac: float = 0.85,
        support_min_class_ratio: float = 0.25,
        support_max_class_ratio: float = 0.75,
        pin_memory: bool = True,
        persistent_workers: bool = True,
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
        self.support_dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": 2,
            "pin_memory": pin_memory,
            "batch_size": 1,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        }

        if not datasets or len(datasets) < 2:
            raise ValueError("datasets must be a list with at least two dataset dicts.")

        self.datasets_cfg = datasets
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_samples_per_epoch = num_iterations_per_epoch * batch_size
        self.return_background_mask = bool(return_background_mask)

        self.support_parquet = (
            None if support_parquet is None else Path(support_parquet)
        )
        self.support_source_id = support_source_id
        self.support_shots_per_class = int(support_shots_per_class)
        self.support_random_seed = int(support_random_seed)

        self.support_min_valid_frac = float(support_min_valid_frac)
        self.support_min_class_ratio = float(support_min_class_ratio)
        self.support_max_class_ratio = float(support_max_class_ratio)

        self.save_hyperparameters(ignore=["transforms", "val_transforms"])

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = CustomTransforms(
                img_size=img_size, scale_range=scale_range
            )

        self.val_transforms = val_transforms

        rank_zero_info(f"[MultiTaskConcatDataModule] batch_size={batch_size}")
        for d in self.datasets_cfg:
            rank_zero_info(
                f"[MultiTaskConcatDataModule] dataset={d.get('name', '?')} root={d['root']} source_id={d.get('source_id')}"
            )

    def _build_train_sampler(self):
        if not isinstance(self.train_dataset, ConcatDataset):
            raise TypeError("train_dataset must be a ConcatDataset.")

        sample_weights = []
        total_len = 0

        for dcfg, subdataset in zip(self.datasets_cfg, self.train_dataset.datasets):
            ds_weight = float(dcfg.get("sampling_weight", 1.0))
            if ds_weight <= 0:
                raise ValueError(
                    f"sampling_weight must be > 0 for dataset {dcfg.get('name', '?')}"
                )

            n = len(subdataset)
            total_len += n
            if n == 0:
                raise ValueError(
                    f"Dataset {dcfg.get('name', '?')} has zero training samples."
                )

            per_sample_weight = ds_weight / n
            sample_weights.extend([per_sample_weight] * n)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.num_samples_per_epoch,
            replacement=True,
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

    @staticmethod
    def support_collate(batch):
        # support dataset returns a single item:
        #   (images: list[Tensor], targets: list[target])
        # DataLoader(batch_size=1) -> [ (images, targets) ]
        images, targets = batch[0]
        return images, targets

    def _get_support_cfg(self) -> Dict[str, Any]:
        if self.support_source_id is None:
            raise ValueError(
                "support_source_id must be provided to build support dataloaders."
            )

        matches = [
            d
            for d in self.datasets_cfg
            if int(d.get("source_id", -1)) == int(self.support_source_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one dataset config with source_id={self.support_source_id}, "
                f"got {len(matches)}."
            )
        return matches[0]

    def _build_fixed_support_df(
        self,
        df: pd.DataFrame,
        shots_per_class: int,
        seed: int,
    ) -> pd.DataFrame:
        rows = []
        for cls in sorted(df["dataset_class_id"].unique()):
            df_cls = df[df["dataset_class_id"] == cls]
            if len(df_cls) == 0:
                continue
            n = min(shots_per_class, len(df_cls))
            rows.append(df_cls.sample(n=n, random_state=seed))
        if len(rows) == 0:
            raise RuntimeError("No support rows found after filtering.")
        return pd.concat(rows, axis=0).reset_index(drop=True)

    def _filter_index(self, df):
        df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])
        if "frac_valid" not in df.columns:
            print("Computing frac_valid column")
            df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

        df = df[df["frac_valid"] >= self.support_min_valid_frac]

        df = df[df["class_frac_valid"] >= self.support_min_class_ratio]
        df = df[df["class_frac_valid"] <= self.support_max_class_ratio]
        return df

    def setup(self, stage: Union[str, None] = None) -> "MultiTaskConcatDataModule":
        train_wrapped = []
        self.val_wrapped = []
        self.test_wrapped = []
        self.predict_wrapped = []
        self.predict_splits = []

        split_info_by_source = {}

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

            split_info_by_source[source_id] = {
                "cfg": dcfg,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            }

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

        # ---------------------------------------------------------
        # Support datasets: always built from TRAIN split of dataset B
        # ---------------------------------------------------------
        if self.support_parquet is not None and stage in (
            "fit",
            "validate",
            "test",
            None,
        ):
            support_cfg = self._get_support_cfg()
            support_source_id = int(support_cfg["source_id"])
            support_root = Path(support_cfg["root"])

            train_ids_support = set(
                split_info_by_source[support_source_id]["train_ids"]
            )

            support_df = pd.read_parquet(self.support_parquet)

            if "sample_id" not in support_df.columns:
                raise ValueError("support_parquet must contain a 'sample_id' column.")
            if "dataset_class_id" not in support_df.columns:
                raise ValueError(
                    "support_parquet must contain a 'dataset_class_id' column."
                )
            if (
                "image_relpath" not in support_df.columns
                or "mask_relpath" not in support_df.columns
            ):
                raise ValueError(
                    "support_parquet must contain 'image_relpath' and 'mask_relpath' columns."
                )

            support_df = support_df[
                support_df["sample_id"].isin(train_ids_support)
            ].reset_index(drop=True)

            if len(support_df) == 0:
                raise RuntimeError(
                    "Support parquet is empty after filtering to train split."
                )

            self.support_train_dataset = SupportDataset(
                index_df=support_df,
                root_dir=support_root,
                shots_per_class=self.support_shots_per_class,
                transform=self.transforms,
                ignore_idx=self.ignore_idx,
                return_background=self.return_background_mask,
            )

            fixed_support_df = self._build_fixed_support_df(
                support_df,
                shots_per_class=self.support_shots_per_class,
                seed=self.support_random_seed,
            )

            self.support_val_dataset = SupportDataset(
                index_df=support_df,
                fixed_df=fixed_support_df,
                root_dir=support_root,
                shots_per_class=self.support_shots_per_class,
                transform=CenterPad(self.img_size),
                ignore_idx=self.ignore_idx,
                return_background=self.return_background_mask,
            )

            # no real test split needed, reuse fixed val support
            self.support_test_dataset = self.support_val_dataset

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

    def train_support_dataloader(self):
        if not hasattr(self, "support_train_dataset"):
            raise AttributeError(
                "support_train_dataset is not initialized. Did you set support_parquet?"
            )
        return DataLoader(
            self.support_train_dataset,
            shuffle=True,
            collate_fn=self.support_collate,
            **self.support_dataloader_kwargs,
        )

    def val_support_dataloader(self):
        if not hasattr(self, "support_val_dataset"):
            raise AttributeError(
                "support_val_dataset is not initialized. Did you set support_parquet?"
            )
        return DataLoader(
            self.support_val_dataset,
            shuffle=False,
            collate_fn=self.support_collate,
            **self.support_dataloader_kwargs,
        )

    def test_support_dataloader(self):
        if not hasattr(self, "support_test_dataset"):
            raise AttributeError(
                "support_test_dataset is not initialized. Did you set support_parquet?"
            )
        return DataLoader(
            self.support_test_dataset,
            shuffle=False,
            collate_fn=self.support_collate,
            **self.support_dataloader_kwargs,
        )

    def val_dataloader(self):
        return [
            DataLoader(ds, collate_fn=self.eval_collate, **self.dataloader_kwargs)
            for _, ds in self.val_wrapped
        ]

    def test_dataloader(self):
        return [
            DataLoader(ds, collate_fn=self.eval_collate, **self.dataloader_kwargs)
            for _, ds in self.test_wrapped
        ]

    def predict_dataloader(self):
        if not getattr(self, "predict_wrapped", None):
            self.setup(stage="predict")

        loaders, splits = [], []
        for split_name, ds in self.predict_wrapped:
            loaders.append(
                DataLoader(ds, collate_fn=self.eval_collate, **self.dataloader_kwargs)
            )
            splits.append(split_name)

        self.predict_splits = splits
        return loaders
