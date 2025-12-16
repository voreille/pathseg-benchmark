from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.lightning_module import LightningModule
from training.tiler import Tiler


class ProtoNetSemantic(LightningModule):
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
        # prototypes_path REMOVED – handled by `network` (ProtoNetDecoder) itself
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        split = self.trainer.datamodule.predict_splits[dataloader_idx]
        imgs, targets, img_ids = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops)  # (N,C,h,w)
        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")

        logits = self.revert_window_logits_semantic(
            crop_logits, origins, img_sizes
        )  # list of (C,H,W)

        # per-pixel targets
        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        outs = []
        for i, logit in enumerate(logits):
            pred = torch.argmax(logit, dim=0)  # (H,W)
            tgt = targets_pp[i].long().to(pred.device)  # (H,W)

            # mIoU (ignoring ignore_idx)
            vals = []
            for c in range(self.hparams.num_classes):
                if c == self.ignore_idx:
                    continue
                inter = ((pred == c) & (tgt == c)).sum().item()
                union = ((pred == c) | (tgt == c)).sum().item()
                if union > 0:
                    vals.append(inter / union)
            miou = float(np.mean(vals)) if vals else float("nan")

            fig_img = self.plot_semantic(imgs[i], tgt, logits=logit)
            outs.append(
                {
                    "pil": fig_img,
                    "miou": miou,
                    "split": split,
                    "img_id": img_ids[i],
                }
            )
        return outs
