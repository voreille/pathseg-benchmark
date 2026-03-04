from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class LinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        tiler: Optional[Tiler] = None,
        class_weights: Optional[list] = None,
        interpolate_logits: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            tiler=tiler,
        )

        self.interpolate_logits = interpolate_logits
        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        weights = None
        if class_weights is not None:
            weights = class_weights

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_idx,
            weight=torch.tensor(weights) if weights is not None else None,
        )

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        logits = self(imgs)
        if self.interpolate_logits:
            logits = F.interpolate(logits, self.img_size, mode="bilinear")

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        loss_total = self.criterion(logits, targets)
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, targets = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops)
        if self.interpolate_logits:
            crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        if is_notebook:
            return logits

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        self.update_metrics(logits, targets, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(
                imgs[0],
                targets[0],
                logits=logits[0],
            )
            # single clean call, no logger assumptions here
            self.log_wandb_image(name, plot, commit=False)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, targets, img_ids = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops)
        if self.interpolate_logits:
            crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        outs = []
        for i, logit in enumerate(logits):
            pred = torch.argmax(logit, dim=0)

            outs.append(
                {
                    "img_id": img_ids[i],
                    "img": imgs[i].detach().cpu(),  # optional but useful for viz
                    "logits": logit.detach().cpu(),
                    "pred": pred.detach().cpu(),
                    "tgt": targets_pp[i].detach().cpu(),  # you have GT, so keep it
                }
            )

        return outs
