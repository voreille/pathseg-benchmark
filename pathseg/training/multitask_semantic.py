from __future__ import annotations

import io
from typing import Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class TwoHeadSemantic(LightningModule):
    """
    Two-head training with mixed datasets in the same batch.

    Assumptions:
    - train batch: (imgs, targets, source_ids, image_ids)
      where targets is list[Any] untouched (e.g., Mask2Former style)
    - network(imgs) returns dict with keys {"logits_a","logits_b"}
    - source_ids indicates which dataset a sample came from:
        source_id_a supervises head A
        source_id_b supervises head B
    """

    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        source_id_a: int,
        source_id_b: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        tiler: Optional[Tiler] = None,
        class_weights_a: Optional[Sequence[float]] = None,
        class_weights_b: Optional[Sequence[float]] = None,
        loss_weight_a: float = 1.0,
        loss_weight_b: float = 1.0,
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

        self.ignore_idx = int(ignore_idx)
        self.poly_lr_decay_power = float(poly_lr_decay_power)

        self.source_id_a = int(source_id_a)
        self.source_id_b = int(source_id_b)

        self.loss_weight_a = float(loss_weight_a)
        self.loss_weight_b = float(loss_weight_b)

        self.num_classes_a = network.num_classes_a
        self.num_classes_b = network.num_classes_b
        self.init_metrics_semantic(
            [self.num_classes_a, self.num_classes_b], self.ignore_idx, num_metrics
        )

        w_a = (
            torch.tensor(class_weights_a, dtype=torch.float32)
            if class_weights_a is not None
            else None
        )
        w_b = (
            torch.tensor(class_weights_b, dtype=torch.float32)
            if class_weights_b is not None
            else None
        )

        self.criterion_a = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=w_a)
        self.criterion_b = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=w_b)

    def _select(self, x, idx: torch.Tensor):
        """
        Select batch elements by boolean mask idx (shape [B]).
        Works for tensors and python lists.
        """
        if torch.is_tensor(x):
            return x[idx]
        # list
        return [v for v, m in zip(x, idx.tolist()) if m]

    def _loss_on_subset(
        self,
        logits: torch.Tensor,  # [B, C, H, W]
        targets_list,  # list[Any]
        subset_mask: torch.Tensor,  # [B] bool
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Compute CE loss only for the subset of samples where subset_mask is True.
        targets_list is the raw target list; we convert only the subset.
        Returns a scalar tensor (0 if no samples).
        """
        if subset_mask.sum().item() == 0:
            # return a proper tensor on the right device for logging/backprop
            return torch.zeros((), device=logits.device)

        logits_s = logits[subset_mask]  # [b, C, H, W]
        targets_s = self._select(targets_list, subset_mask)

        # convert raw targets -> per-pixel semantic mask list[H,W]
        targets_pp = self.to_per_pixel_targets_semantic(targets_s, self.ignore_idx)
        targets_pp = torch.stack(targets_pp).long().to(logits.device)  # [b, H, W]

        return criterion(logits_s, targets_pp)

    def training_step(self, batch, batch_idx):
        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        out = self(imgs)  # your network forward
        # support either dict or tuple
        if isinstance(out, dict):
            logits_a = out["logits_a"]
            logits_b = out["logits_b"]
        else:
            logits_a, logits_b = out

        # bring both heads to same spatial size
        logits_a = F.interpolate(logits_a, self.img_size, mode="bilinear")
        logits_b = F.interpolate(logits_b, self.img_size, mode="bilinear")

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b

        loss_a = self._loss_on_subset(logits_a, targets, m_a, self.criterion_a)
        loss_b = self._loss_on_subset(logits_b, targets, m_b, self.criterion_b)

        loss_total = self.loss_weight_a * loss_a + self.loss_weight_b * loss_b

        # logging
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True, prog_bar=False)
        self.log("train_loss_b", loss_b, sync_dist=True, prog_bar=False)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True, prog_bar=False)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True, prog_bar=False)

        return loss_total

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, log_prefix="val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, log_prefix="test")

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
    ):
        imgs, targets, source_ids, image_ids = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops)
        if isinstance(crop_out, dict):
            crop_logits_a = crop_out["logits_a"]
            crop_logits_b = crop_out["logits_b"]
        else:
            crop_logits_a, crop_logits_b = crop_out

        sid0 = int(source_ids[0])
        crop_logits_a = F.interpolate(crop_logits_a, self.img_size, mode="bilinear")
        crop_logits_b = F.interpolate(crop_logits_b, self.img_size, mode="bilinear")
        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a, origins, img_sizes
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b, origins, img_sizes
        )

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        # update_metrics expects list logits? your current update_metrics takes (logits, targets)
        # In your old code you passed logits as list (from revert_window_logits_semantic), so keep it:
        self.update_metrics(
            logits_a_list if sid0 == self.source_id_a else logits_b_list,
            targets_pp,
            dataloader_idx,
        )

        if batch_idx == 0:
            img0 = imgs[0]
            tgt0 = targets_pp[0]

            plot = self.plot_two_head_semantic(
                img=img0,
                logits_a=logits_a_list[0],
                logits_b=None if logits_b_list is None else logits_b_list[0],
                target=tgt0,
                gt_head="a" if sid0 == self.source_id_a else "b",
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_two_head_{batch_idx}",
                plot,
                commit=False,
            )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_semantic("test")

    # -------------------------
    # Optim
    # -------------------------
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

    # -------------------------
    # Predict (for viz/export)
    # -------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, targets, source_ids, img_ids = batch  # from eval_collate
        imgs = torch.stack(list(imgs)).to(self.device)
        targets = list(targets)
        source_ids = list(source_ids)
        img_ids = list(img_ids)

        # infer which head from the loader
        sid0 = int(source_ids[0])

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops)
        if isinstance(crop_out, dict):
            crop_logits = (
                crop_out["logits_a"]
                if sid0 == self.source_id_a
                else crop_out["logits_b"]
            )
        else:
            crop_logits_a, crop_logits_b = crop_out
            crop_logits = crop_logits_a if sid0 == self.source_id_a else crop_logits_b

        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        outs = []
        for i, logit in enumerate(logits):
            pred = torch.argmax(logit, dim=0)
            H, W = int(pred.shape[-2]), int(pred.shape[-1])

            outs.append(
                {
                    "img_id": img_ids[i],
                    "source_id": sid0,
                    "img_hw": (H, W),
                    "img": imgs[i].detach().cpu(),
                    "logits": logit.detach().cpu(),
                    "pred": pred.detach().cpu(),
                    "tgt": targets_pp[i].detach().cpu(),
                }
            )
        return outs

    def init_metrics_semantic(
        self,
        num_classes: Union[int, Sequence[int]],
        ignore_idx: int,
        num_metrics: int | None = None,
    ) -> None:
        # Normalize num_classes -> list[int]
        if isinstance(num_classes, int):
            if num_metrics is None:
                num_metrics = 1
            num_classes_list = [int(num_classes)] * int(num_metrics)
        else:
            num_classes_list = [int(x) for x in num_classes]
            if num_metrics is not None and int(num_metrics) != len(num_classes_list):
                raise ValueError(
                    f"num_metrics ({num_metrics}) must match len(num_classes) "
                    f"({len(num_classes_list)}) when num_classes is a list."
                )

        self._metrics_num_classes = num_classes_list  # optional: useful for debugging

        self.iou_metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=nc,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,  # per-class
                )
                for nc in num_classes_list
            ]
        )

        self.f1_metrics = nn.ModuleList(
            [
                MulticlassF1Score(
                    num_classes=nc,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,  # per-class
                )
                for nc in num_classes_list
            ]
        )

    @torch.compiler.disable
    def update_metrics(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_idx: int,
    ) -> None:
        """
        preds:   list of (C,H,W) logits OR list of (H,W) class indices
        targets: list of (H,W) class indices

        Your current pipeline passes logits_list from revert_window_logits_semantic
        (each is (C,H,W)), so this keeps that behavior.

        If you ever pass class-index maps (H,W), this will also work.
        """
        if dataloader_idx < 0 or dataloader_idx >= len(self.iou_metrics):
            raise IndexError(
                f"dataloader_idx={dataloader_idx} out of range for "
                f"{len(self.iou_metrics)} metric streams."
            )

        for p, t in zip(preds, targets):
            # p: (C,H,W) logits OR (H,W) int preds
            if p.ndim == 3:
                p_in = p.unsqueeze(0)  # (1,C,H,W)
            elif p.ndim == 2:
                p_in = p.unsqueeze(0)  # (1,H,W)
            else:
                raise ValueError(f"Unexpected pred shape: {tuple(p.shape)}")

            if t.ndim == 2:
                t_in = t.unsqueeze(0)  # (1,H,W)
            else:
                raise ValueError(f"Unexpected target shape: {tuple(t.shape)}")

            self.iou_metrics[dataloader_idx].update(p_in, t_in)
            self.f1_metrics[dataloader_idx].update(p_in, t_in)

    @torch.compiler.disable
    def plot_two_head_semantic(
        self,
        img: torch.Tensor,
        logits_a: torch.Tensor | None = None,
        logits_b: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        gt_head: str | None = None,
        cmap: str = "tab20",
    ):
        panels = [("Image", "img")]
        if target is not None:
            panels.append((f"GT ({gt_head})" if gt_head is not None else "GT", "gt"))
        if logits_a is not None:
            panels.append(("Head A", "a"))
        if logits_b is not None:
            panels.append(("Head B", "b"))

        ncols = len(panels)
        fig, axes = plt.subplots(
            1, ncols, figsize=(5 * ncols, 5), sharex=True, sharey=True
        )
        if ncols == 1:
            axes = [axes]

        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)

        target_np = None if target is None else target.detach().cpu().numpy()

        pred_a = None
        if logits_a is not None:
            pred_a = torch.argmax(logits_a, dim=0).detach().cpu().numpy()

        pred_b = None
        if logits_b is not None:
            pred_b = torch.argmax(logits_b, dim=0).detach().cpu().numpy()

        uniq_parts = []
        if target_np is not None:
            uniq_parts.append(np.unique(target_np))
        if pred_a is not None:
            uniq_parts.append(np.unique(pred_a))
        if pred_b is not None:
            uniq_parts.append(np.unique(pred_b))

        if len(uniq_parts) == 0:
            unique_classes = np.array([0], dtype=np.int64)
        else:
            unique_classes = np.unique(np.concatenate(uniq_parts))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))

        if self.ignore_idx in unique_classes:
            ignore_mask = unique_classes == self.ignore_idx
            colors[ignore_mask] = [0, 0, 0, 1]

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)

        def encode_mask(x: np.ndarray) -> np.ndarray:
            return np.digitize(x, unique_classes, right=False) - 1

        for ax, (title, kind) in zip(axes, panels):
            ax.set_title(title)
            ax.axis("off")

            if kind == "img":
                ax.imshow(img_np)
            elif kind == "gt":
                ax.imshow(
                    encode_mask(target_np),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind == "a":
                ax.imshow(
                    encode_mask(pred_a),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind == "b":
                ax.imshow(
                    encode_mask(pred_b),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )

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
