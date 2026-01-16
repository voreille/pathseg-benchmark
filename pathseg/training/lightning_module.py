import io
import math
from typing import Optional

import lightning
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from matplotlib.lines import Line2D
from PIL import Image
from torch.nn.functional import interpolate
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torchvision.transforms.v2.functional import resize

from pathseg.training.tiler import Tiler


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        img_size: tuple[int, int],
        freeze_encoder: bool,
        network: nn.Module,
        weight_decay: float,
        lr: float,
        lr_multiplier_encoder: float,
        tiler: Optional[Tiler] = None,
    ):
        super().__init__()

        self.img_size = img_size
        self.network = network
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_multiplier_encoder = lr_multiplier_encoder

        if hasattr(self.network, "encoder") and freeze_encoder:
            for param in self.network.encoder.parameters():
                param.requires_grad = False

            self.network.encoder.eval()
        elif freeze_encoder:
            raise ValueError(
                "Cannot freeze encoder: the network has no attribute 'encoder'."
            )

        self.log = torch.compiler.disable(self.log)  # type: ignore
        self.tiler = tiler

    def init_metrics_semantic(self, num_classes, ignore_idx, num_metrics):
        self.iou_metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,  # per-class
                )
                for _ in range(num_metrics)
            ]
        )

        self.f1_metrics = nn.ModuleList(
            [
                MulticlassF1Score(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,  # per-class
                )
                for _ in range(num_metrics)
            ]
        )

    @torch.compiler.disable
    def update_metrics(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor], dataloader_idx
    ):
        for i in range(len(preds)):
            p = preds[i][None, ...]  # add batch dim
            t = targets[i][None, ...]
            self.iou_metrics[dataloader_idx].update(p, t)
            self.f1_metrics[dataloader_idx].update(p, t)

    def forward(self, imgs):
        x = imgs / 255.0

        output = self.network(x)

        if not self.training and isinstance(output, tuple):
            return (y[-1] for y in self.network(x))

        return output

    def on_train_batch_start(self, _, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(
                f"group_{i}_lr_{len(param_group['params'])}",
                param_group["lr"],
                on_step=True,
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def _on_eval_epoch_end_semantic(self, log_prefix):
        for metric_idx in range(len(self.iou_metrics)):
            iou_pc = self.iou_metrics[metric_idx].compute()  # (C,)
            f1_pc = self.f1_metrics[metric_idx].compute()  # (C,)

            self.iou_metrics[metric_idx].reset()
            self.f1_metrics[metric_idx].reset()

            # log per-class
            for c, v in enumerate(iou_pc):
                self.log(f"{log_prefix}_{metric_idx}_iou_{c}", v, sync_dist=True)
            for c, v in enumerate(f1_pc):
                self.log(f"{log_prefix}_{metric_idx}_f1_{c}", v, sync_dist=True)

            # log means (ignoring ignore_idx already handled by torchmetrics)
            self.log(
                f"{log_prefix}_{metric_idx}_miou", float(iou_pc.mean()), sync_dist=True
            )
            self.log(
                f"{log_prefix}_{metric_idx}_mf1", float(f1_pc.mean()), sync_dist=True
            )

    def configure_optimizers(self):
        if hasattr(self.network, "encoder"):
            enc_params = list(self.network.encoder.parameters())
        else:
            enc_params = []

        enc_ids = {id(p) for p in enc_params}

        base_params = [p for p in self.parameters() if id(p) not in enc_ids]

        return AdamW(
            [
                {"params": base_params, "lr": self.lr},
                {
                    "params": enc_params,
                    "lr": self.lr * self.lr_multiplier_encoder,
                },
            ],
            weight_decay=self.weight_decay,
        )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits=None,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = None
        if logits is not None:
            preds = torch.argmax(logits, dim=0).cpu().numpy()
            unique_classes = np.unique(
                np.concatenate((unique_classes, np.unique(preds)))
            )

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)

    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    def window_imgs_semantic(self, imgs):
        if self.tiler is not None:
            return self.tiler.window(imgs)

        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = [], []
        for i in range(len((imgs))):
            new_img = resize(
                imgs[i],
                self.scale_img_size_semantic(img_sizes[i]),
            )

            num_crops = math.ceil(max(new_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(new_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                crop = (
                    new_img[:, start:end, :]
                    if new_img.shape[-2] > new_img.shape[-1]
                    else new_img[:, :, start:end]
                )

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins, [img.shape[-2:] for img in imgs]

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        if self.tiler is not None:
            return self.tiler.stitch(crop_logits, origins, img_sizes)

        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate((sums / counts)[None, ...], img_sizes[i], mode="bilinear")[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        split = self.trainer.datamodule.predict_splits[dataloader_idx]
        imgs, targets, img_ids = batch
        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops)

        if isinstance(crop_logits, dict):
            crop_logits = crop_logits["main"]

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

            # reuse your plot function to get a PIL Image
            fig_img = self.plot_semantic(imgs[i], tgt, logits=logit)
            outs.append(
                {"pil": fig_img, "miou": miou, "split": split, "img_id": img_ids[i]}
            )
        return outs

    def get_wandb_experiment(self):
        """Return the wandb.Run object if available, else None."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None

        # Case 1 — single logger: trainer.logger
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        if exp is not None and hasattr(exp, "log"):
            return exp

        # Case 2 — multiple loggers: trainer.loggers (a Python list)
        loggers = getattr(trainer, "loggers", None)
        if isinstance(loggers, list):
            for lg in loggers:
                exp = getattr(lg, "experiment", None)
                if exp is not None and hasattr(exp, "log"):
                    return exp

        return None

    def log_wandb_image(self, name: str, image, commit: bool = False):
        exp = self.get_wandb_experiment()
        if exp is not None:
            exp.log({name: [wandb.Image(image)]}, commit=commit)
