from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.training.histo_loss import CrossEntropyDiceLoss
from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class MetaLinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,  # put an hist encoder
        metahead: nn.Module,
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

        self.save_hyperparameters(ignore=["network"])
        self.metahead = metahead

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        if class_weights is not None:
            weights = class_weights
        else:
            # pull from datamodule in setup()
            weights = None

        # self._init_class_weights = weights
        # self.criterion = None

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_idx,
            weight=torch.tensor(weights) if weights is not None else None,
        )
        patch_size = self.network.patch_size
        self.label_downsampler = nn.AvgPool2d(patch_size, patch_size)

    def training_step(self, batch, batch_idx):
        imgs_s, targets_s, imgs_q, targets_q = batch # target (B, N, H, W)
        n_classes = targets_s.shape[1]

        fmaps_s = self(
            imgs_s
        )  # (K*N, E, Hp, Wp) K=shots, N=1, E=embed_dim, Hp,Wp=spatial dims divided by encoder stride

        fmaps_q = self(imgs_q)  # (Q, E, Hp, Wp) Q=queries
        ws = []
        bs = []
        for c in range(n_classes):
            support_c_index = 
            w, b = self.metahead(fmaps_s, targets_s.float())  # w: (B,E), b: (B,1)
            ws.append(w)
            bs.append(b)

        logits = F.interpolate(logits, self.img_size, mode="bilinear")

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        loss_total = self.criterion(logits, targets)
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total

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
