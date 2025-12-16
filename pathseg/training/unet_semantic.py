from typing import Optional

import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from training.lightning_module import LightningModule
from training.tiler import Tiler
from training.histo_loss import CrossEntropyDiceLoss


class UnetSemantic(LightningModule):

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

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        # just remember the init value; we’ll materialize criterion in setup()
        self._init_class_weights = class_weights
        self.criterion = None  # set in setup()
        if class_weights is not None:
            weights = class_weights
        else:
            # pull from datamodule in setup()
            weights = None
        self._init_class_weights = weights
        self.criterion = None

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            w = (torch.tensor(self._init_class_weights,
                              dtype=torch.float32,
                              device=self.device)
                 if self._init_class_weights is not None else getattr(
                     self.trainer.datamodule, "class_weights", None))
            if w is not None and not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float32, device=self.device)
            print(f"Using class weights: {w} for semantic loss")
            self.criterion = CrossEntropyDiceLoss(class_weights=w,
                                                  ignore_index=self.ignore_idx,
                                                  ce_weight=0.5,
                                                  dice_weight=0.5)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        logits = self(imgs)
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
        crop_logits = F.interpolate(crop_logits,
                                    self.img_size,
                                    mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins,
                                                    img_sizes)

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
            if hasattr(self.trainer.logger.experiment, "log"):
                self.trainer.logger.experiment.log({name: [wandb.Image(plot)]
                                                    })  # type: ignore

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler":
            PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval":
            "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
