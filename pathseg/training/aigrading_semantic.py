from typing import Optional

import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from training.lightning_module import LightningModule
from training.tiler import Tiler


class AIGradingSemantic(LightningModule):
    """
    Lightning module wrapping the SelfCrossPSP network.

    Differences from FPNSemantic:
    - No deep supervision (only "main" prediction).
    - Simple piecewise LR schedule (as in the original Keras script).
    - Expects per-pixel integer targets (B, H, W) or (B, 1, H, W).
    """

    def __init__(
            self,
            network: nn.Module,
            num_metrics: int,
            num_classes: int,
            ignore_idx: int,
            img_size: tuple[int, int],
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            freeze_encoder: bool = False,  # kept for interface compatibility
            tiler: Optional[Tiler] = None,  # not used by default
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=1.0,
            tiler=tiler,
        )

        self.ignore_idx = ignore_idx

        # loss ~ sparse_categorical_crossentropy
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

        # Don’t save model object in hparams
        self.save_hyperparameters(ignore=["network"])

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        output = self(imgs)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        loss_total = self.criterion(output, targets)

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


        def lr_lambda(global_step: int) -> float:
            if global_step < 4000:
                return 1.0
            elif global_step < 20000:
                return 0.1
            else:
                return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
