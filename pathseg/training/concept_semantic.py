from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class ConceptQuerySemanticModule(LightningModule):
    """
    LightningModule for ConceptQuerySemanticDecoder.

    Supports:
      - Deep supervision (semantic CE on intermediate layers)
      - Anti-collapse regularization (usage entropy + optional class entropy)
      - Concept statistics logging
    """

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
        class_weights: Optional[Sequence[float]] = None,
        # ---- Deep supervision ----
        deep_supervision: bool = False,
        deep_sup_weights: Optional[Sequence[float]] = None,
        # ---- Anti-collapse ----
        anti_collapse: bool = True,
        lambda_usage: float = 1e-2,
        lambda_class_entropy: float = 0.0,
        active_area_threshold: float = 0.02,
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

        # Deep supervision
        self.deep_supervision = deep_supervision
        self.deep_sup_weights = deep_sup_weights
        if self.deep_supervision:
            self.network.enable_deep_supervision()

        # Anti-collapse
        self.anti_collapse = anti_collapse
        self.lambda_usage = lambda_usage
        self.lambda_class_entropy = lambda_class_entropy
        self.active_area_threshold = active_area_threshold

        # Metrics
        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

        # CE loss
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_idx,
            weight=weight_tensor,
        )

    # =========================================================
    # Deep Supervision CE
    # =========================================================
    def _compute_deep_supervised_ce(self, out: dict, targets: torch.Tensor):
        logits_list = []

        if self.deep_supervision and "semantic_logits_per_layer" in out:
            logits_list.extend(out["semantic_logits_per_layer"])

        logits_list.append(out["semantic_logits"])

        if len(logits_list) == 1:
            logits = F.interpolate(
                logits_list[0], self.img_size, mode="bilinear", align_corners=False
            )
            return self.criterion(logits, targets)

        # Default weights
        if self.deep_sup_weights is None:
            weights = [0.2] * (len(logits_list) - 1) + [1.0]
        else:
            weights = self.deep_sup_weights

        loss = 0.0
        for w, logits in zip(weights, logits_list):
            logits = F.interpolate(
                logits, self.img_size, mode="bilinear", align_corners=False
            )
            loss += w * self.criterion(logits, targets)

        return loss

    # =========================================================
    # Anti-collapse regularization
    # =========================================================
    def _compute_anti_collapse(self, out: dict, targets: torch.Tensor):
        if not self.anti_collapse:
            return torch.tensor(0.0, device=targets.device), {}

        mask_logits = out.get("concept_mask_logits", None)
        class_logits = out.get("concept_class_logits", None)

        if mask_logits is None or class_logits is None:
            return torch.tensor(0.0, device=targets.device), {}

        mask_logits = F.interpolate(
            mask_logits, self.img_size, mode="bilinear", align_corners=False
        )

        mask_probs = torch.sigmoid(mask_logits)  # (B,K,H,W)
        valid = (targets != self.ignore_idx).float()

        if valid.sum() == 0:
            return torch.tensor(0.0, device=targets.device), {}

        valid = valid[:, None, :, :]
        area = (mask_probs * valid).sum((-2, -1)) / (valid.sum((-2, -1)) + 1e-6)

        # ---- Usage entropy ----
        p = area / (area.sum(dim=1, keepdim=True) + 1e-6)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
        loss_usage = -entropy

        # ---- Class entropy (optional) ----
        loss_class_entropy = torch.tensor(0.0, device=targets.device)
        mean_max_prob = torch.tensor(0.0, device=targets.device)

        if self.lambda_class_entropy > 0:
            class_probs = F.softmax(class_logits, dim=-1)
            class_entropy = (
                -(class_probs * torch.log(class_probs + 1e-8)).sum(-1).mean()
            )
            loss_class_entropy = class_entropy
            mean_max_prob = class_probs.max(dim=-1).values.mean()

        loss_reg = (
            self.lambda_usage * loss_usage
            + self.lambda_class_entropy * loss_class_entropy
        )

        stats = {
            "concept_usage_entropy": entropy.detach(),
            "concept_active_count": (area > self.active_area_threshold)
            .float()
            .sum(dim=1)
            .mean()
            .detach(),
            "concept_mean_max_class_prob": mean_max_prob.detach(),
        }

        return loss_reg, stats

    # =========================================================
    # Training Step
    # =========================================================
    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        out = self(imgs)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        loss_ce = self._compute_deep_supervised_ce(out, targets)
        loss_ce = self.criterion(
            F.interpolate(
                out["semantic_logits"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            ),
            targets,
        )
        loss_reg, stats = self._compute_anti_collapse(out, targets)

        loss_total = loss_ce + loss_reg
        # loss_total = loss_ce

        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("train_loss_ce", loss_ce, sync_dist=True)

        if self.anti_collapse:
            self.log("train_loss_reg", loss_reg, sync_dist=True)
            for k, v in stats.items():
                self.log(f"train_{k}", v, sync_dist=True)

        return loss_total

    # =========================================================
    # Validation
    # =========================================================
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
        out_crop = self(crops)

        crop_logits = out_crop["semantic_logits"]
        crop_logits = F.interpolate(
            crop_logits, self.img_size, mode="bilinear", align_corners=False
        )

        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        if is_notebook:
            return logits

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        self.update_metrics(logits, targets_pp, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(imgs[0], targets_pp[0], logits=logits[0])
            self.log_wandb_image(name, plot, commit=False)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    # =========================================================
    # Optimizer
    # =========================================================
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


class PrimitiveBankSemanticModule(LightningModule):
    """
    LightningModule for PrimitiveBankSemanticDecoder (single dataset).

    Supports:
      - Standard CE loss
      - Primitive usage entropy (anti-collapse)
      - Optional assignment entropy (sharpen primitives)
      - Prototype diversity regularization
    """

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
        class_weights: Optional[Sequence[float]] = None,
        # ---- Primitive regularization ----
        anti_collapse: bool = True,
        lambda_usage: float = 1e-2,
        lambda_assignment_entropy: float = 1e-3,
        lambda_diversity: float = 1e-3,
        active_area_threshold: float = 0.02,
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

        # Metrics
        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_idx,
            weight=weight_tensor,
        )

        # Regularization
        self.anti_collapse = anti_collapse
        self.lambda_usage = lambda_usage
        self.lambda_assignment_entropy = lambda_assignment_entropy
        self.lambda_diversity = lambda_diversity
        self.active_area_threshold = active_area_threshold

    # =========================================================
    # Primitive Regularization
    # =========================================================
    def _compute_primitive_regularization(self, out: dict, targets: torch.Tensor):
        if not self.anti_collapse:
            return torch.tensor(0.0, device=targets.device), {}

        primitive_probs = out["primitive_probs"]  # (B,K,H,W)
        B, K, H, W = primitive_probs.shape

        primitive_probs = F.interpolate(
            primitive_probs, self.img_size, mode="bilinear", align_corners=False
        )

        valid = (targets != self.ignore_idx).float()  # (B,H,W)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=targets.device), {}

        valid4 = valid[:, None, :, :]

        # -------------------------
        # 1) Usage entropy (global)
        # -------------------------
        area = (primitive_probs * valid4).sum((-2, -1)) / (
            valid4.sum((-2, -1)) + 1e-6
        )  # (B,K)

        p = area / (area.sum(dim=1, keepdim=True) + 1e-6)
        usage_entropy = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()

        loss_usage = -usage_entropy

        # -------------------------
        # 2) Per-pixel assignment entropy
        # -------------------------
        loss_assign = torch.tensor(0.0, device=targets.device)
        assign_entropy = torch.tensor(0.0, device=targets.device)

        if self.lambda_assignment_entropy > 0:
            entropy_map = -(primitive_probs * torch.log(primitive_probs + 1e-8)).sum(
                dim=1
            )
            assign_entropy = (entropy_map * valid).sum() / (valid.sum() + 1e-6)
            loss_assign = assign_entropy  # minimize entropy

        # -------------------------
        # 3) Prototype diversity (orthogonality)
        # -------------------------
        loss_div = torch.tensor(0.0, device=targets.device)

        if self.lambda_diversity > 0:
            P = self.network.primitive_prototypes  # (K,D)
            P = F.normalize(P, dim=1)
            sim = torch.matmul(P, P.t())  # (K,K)
            identity = torch.eye(K, device=sim.device)
            loss_div = ((sim - identity) ** 2).mean()

        loss_reg = (
            self.lambda_usage * loss_usage
            + self.lambda_assignment_entropy * loss_assign
            + self.lambda_diversity * loss_div
        )

        stats = {
            "primitive_usage_entropy": usage_entropy.detach(),
            "primitive_assignment_entropy": assign_entropy.detach(),
        }

        return loss_reg, stats

    # =========================================================
    # Training
    # =========================================================
    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        out = self(imgs)

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets_pp = torch.stack(targets_pp).long()

        logits = F.interpolate(
            out["semantic_logits"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        loss_ce = self.criterion(logits, targets_pp)

        loss_reg, stats = self._compute_primitive_regularization(out, targets_pp)

        loss_total = loss_ce + loss_reg

        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("train_loss_ce", loss_ce, sync_dist=True)

        if self.anti_collapse:
            self.log("train_loss_reg", loss_reg, sync_dist=True)
            for k, v in stats.items():
                self.log(f"train_{k}", v, sync_dist=True)

        return loss_total

    # =========================================================
    # Validation (same structure as before)
    # =========================================================
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
        out_crop = self(crops)

        crop_logits = F.interpolate(
            out_crop["semantic_logits"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        if is_notebook:
            return logits

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        self.update_metrics(logits, targets_pp, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(imgs[0], targets_pp[0], logits=logits[0])
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
