from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.training.histo_loss import CrossEntropyDiceLoss
from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class FPNSemantic(LightningModule):
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
        deep_supervision: bool = True,
        deep_supervision_weights: Optional[dict[str, float]] = None,
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
        self.deep_supervision = deep_supervision
        if deep_supervision_weights is None:
            deep_supervision_weights = {"main": 1.0, "aux8": 0.4, "aux16": 0.2}
        self.deep_supervision_weights = deep_supervision_weights

        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        # just remember the init value; we’ll materialize criterion in setup()
        self._init_class_weights = class_weights
        if class_weights is not None:
            weights = class_weights
        else:
            weights = None

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_idx,
            weight=torch.tensor(weights) if weights is not None else None,
        )
        # self.criterion = CrossEntropyDiceLoss(class_weights=weights,
        #                                       ignore_index=self.ignore_idx,
        #                                       ce_weight=0.5,
        #                                       dice_weight=0.5)

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        output = self(imgs)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        if self.deep_supervision:
            loss_total = (
                self.criterion(
                    output["main"],
                    targets,
                )
                * self.deep_supervision_weights.get("main", 1.0)
                + self.criterion(
                    output.get("aux8"),
                    targets,
                )
                * self.deep_supervision_weights.get("aux8", 0.0)
                + self.criterion(
                    output.get("aux16"),
                    targets,
                )
                * self.deep_supervision_weights.get("aux16", 0.0)
            )
        else:
            loss_total = self.criterion(
                output["main"],
                targets,
            )

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
        crop_logits = self(crops)["main"]
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
