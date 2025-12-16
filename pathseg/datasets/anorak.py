from pathlib import Path
from typing import Optional, Union
import math

import pandas as pd
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import DataLoader

from datasets.anorak_dataset import Dataset, PredictDataset, FewShotSupportDataset
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import CustomTransforms, CustomTransformsVaryingSize
from datasets.utils import RepeatDataset


class ANORAK(LightningDataModule):
    def __init__(
        self,
        root,
        devices,
        num_workers: int = 0,
        fold: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_classes: int = 7,
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
            devices=devices,
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
        rank_zero_info(f"[ANORAK] Initializing datamodule with fold = {self.fold}")
        rank_zero_info(
            f"[ANORAK] Initializing datamodule with batch_size = {batch_size}"
        )

        self.images_dir = root_dir / "image"
        self.masks_dir = root_dir / "mask"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]

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
        return (
            self.split_df[self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]["image_id"].unique().tolist(),
        )

    def compute_class_weights(self):
        from datasets.stats import compute_class_weights_from_ids

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
            self.class_weights = self.compute_class_weights()

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


class ANORAKFewShotOld(LightningDataModule):
    def __init__(
        self,
        root,
        devices,
        num_workers: int = 0,
        fold: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_classes: int = 7,
        num_metrics: int = 1,
        ignore_idx: int = 255,
        overwrite_root: Optional[str] = None,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
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

        self.images_dir = root_dir / "image"
        self.masks_dir = root_dir / "mask"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]

        self.save_hyperparameters()

    def _get_split_ids(self):
        return (
            self.split_df[self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]["image_id"].unique().tolist(),
        )

    def compute_class_weights(self):
        from datasets.stats import compute_class_weights_from_ids

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
            self.train_dataset = Dataset(train_ids, self.images_dir, self.masks_dir)

            self.val_dataset = Dataset(val_ids, self.images_dir, self.masks_dir)

            # compute once per fold for training
            self.class_weights = self.compute_class_weights()

        if stage in ("test", None):
            self.test_dataset = Dataset(test_ids, self.images_dir, self.masks_dir)

        if stage in ("predict", None):
            # self.val_dataset = PredictDataset(val_ids, self.images_dir, self.masks_dir)
            self.test_dataset = PredictDataset(
                test_ids, self.images_dir, self.masks_dir
            )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.eval_collate,
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


class ANORAKFewShot(LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        devices,
        fewshot_csv: str | Path,
        support_set: int = 0,  # 0, 1, or 2
        n_shot: int = 10,  # k: 1..10
        img_size: tuple[int, int] = (448, 448),
        scale_range: tuple[float, float] = (0.7, 1.3),
        num_classes: int = 7,
        num_metrics: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        num_iter_per_epoch: int = 1500,
        ignore_idx: int = 255,
        train_transforms: Optional[nn.Module] = None,
        test_transforms: Optional[nn.Module] = None,
        overwrite_root: Optional[str] = None,
        drop_last: bool = True,
        no_train_augmentation: bool = False,
    ):
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
        )

        # from datasets.transforms_extended import CustomTransformsExtended

        if overwrite_root:
            root = overwrite_root
        self.root = Path(root)
        self.devices = devices
        self.fewshot_csv = Path(fewshot_csv)
        self.support_set = support_set
        self.n_shot = n_shot
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        self.num_iter_per_epoch = num_iter_per_epoch
        self.epoch_repeat = 1
        self.ignore_idx = ignore_idx

        self.images_dir = self.root / "image"
        self.masks_dir = self.root / "mask"

        rank_zero_info(
            f"[ANORAKFewShot] support_set={support_set}, n_shot={n_shot}, csv={self.fewshot_csv}"
        )


        if not no_train_augmentation and train_transforms is None:
            self.train_transforms = CustomTransforms(
                img_size=img_size,
                scale_range=scale_range,
            )
        elif not no_train_augmentation and train_transforms is not None:
            self.train_transforms = train_transforms
        else:
            self.train_transforms = None  # or an eval transform

        if test_transforms is not None:
            self.test_transforms = test_transforms
        else:
            self.test_transforms = None  # or an eval transform

    def setup(self, stage: Union[str, None] = None):
        df = pd.read_csv(self.fewshot_csv)

        # TRAIN = support_set == chosen, is_test == False, 1 <= k_shot <= n_shot
        train_mask = (
            (df["is_test"] == False)
            & (df["support_set"] == self.support_set)
            & (df["k_shot"] >= 1)
            & (df["k_shot"] <= self.n_shot)
        )
        train_df = df[train_mask].copy()

        # TEST = all is_test == True
        test_ids = df[df["is_test"] == True]["image_id"].unique().tolist()

        N = len(train_df)
        base_iters = math.ceil(N / self.batch_size)
        if self.num_iter_per_epoch > 1:
            self.epoch_repeat = math.ceil(self.num_iter_per_epoch / base_iters)
        else:
            self.epoch_repeat = 1

        rank_zero_info(
            f"[ANORAKFewShot] N={N}, base_iters={base_iters}, "
            f"num_iter_per_epoch={self.num_iter_per_epoch} -> epoch_repeat={self.epoch_repeat}"
        )

        rank_zero_info(
            f"[ANORAKFewShot] train_images={train_df['image_id'].nunique()}, "
            f"train_rows={len(train_df)}, test_images={len(test_ids)}"
        )

        if stage in ("fit", "validate", None):
            self.train_dataset = FewShotSupportDataset(
                df=train_df,
                images_directory=self.images_dir,
                masks_directory=self.masks_dir,
                ignore_idx=self.ignore_idx,
                transforms=self.train_transforms,
            )
 
            self.test_dataset = Dataset(
                test_ids,
                self.images_dir,
                self.masks_dir,
                transforms=self.test_transforms,
            )

        if stage in ("test", None):
            self.test_dataset = Dataset(
                test_ids,
                self.images_dir,
                self.masks_dir,
                transforms=self.test_transforms,
            )

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
            drop_last=self.drop_last,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
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
