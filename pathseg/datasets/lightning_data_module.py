from typing import Optional
import torch
import lightning


class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        num_metrics: int,
        ignore_idx: int = 255,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()

        self.root = root
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_metrics = num_metrics

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        }

    @staticmethod
    def train_collate(batch):
        imgs, targets = [], []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs), targets

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))
