import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_valid(logits, targets, ignore_idx):
    # logits: [B, C, H, W], targets: [B, H, W]
    if ignore_idx is None or ignore_idx < 0:
        return logits, targets, None
    valid = targets != ignore_idx
    if valid.all():
        return logits, targets, None
    # flatten valid positions
    v = valid.view(-1)
    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])[v]  # [N_valid, C]
    targets = targets.view(-1)[v]  # [N_valid]
    return logits, targets, valid


# class CrossEntropyDiceLoss(nn.Module):
#     def __init__(
#         self,
#         weight=None,
#         ignore_index=255,
#         ce_w=0.5,
#         dice_w=0.5,
#         smooth=1.0,
#     ):
#         super().__init__()
#         self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
#         self.ce_w = ce_w
#         self.dice_w = dice_w
#         self.smooth = smooth
#         self.ignore_index = ignore_index

#     def forward(self, logits, target):
#         ce = self.ce(logits, target)

#         probs = torch.softmax(logits, dim=1)

#         valid_mask = target != self.ignore_index  # [B, H, W]

#         safe_target = target.clone()
#         safe_target[~valid_mask] = 0

#         target_oh = torch.zeros_like(probs).scatter_(1, safe_target.unsqueeze(1), 1)

#         valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
#         probs = probs * valid_mask
#         target_oh = target_oh * valid_mask

#         intersection = (probs * target_oh).sum(dim=(0, 2, 3))
#         cardinality = (probs + target_oh).sum(dim=(0, 2, 3))

#         dice = (
#             1
#             - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
#         )

#         return self.ce_w * ce + self.dice_w * dice


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)  # [B,C,H,W]

        valid_mask = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        target_oh = torch.zeros_like(probs).scatter_(1, safe_target.unsqueeze(1), 1)

        valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]
        probs = probs * valid_mask
        target_oh = target_oh * valid_mask

        intersection = (probs * target_oh).sum(dim=(0, 2, 3))  # [C]
        cardinality = (probs + target_oh).sum(dim=(0, 2, 3))  # [C]

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        # exclude background
        fg_dice = dice_per_class[1:]
        fg_target_mass = target_oh[:, 1:, :, :].sum(dim=(0, 2, 3))  # [C-1]
        present = fg_target_mass > 0

        if present.any():
            dice = 1.0 - fg_dice[present].mean()
        else:
            dice = logits.new_tensor(0.0)

        return ce + dice


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        smooth: float = 1e-5,
        ignore_index: int | None = 255,
        smooth_labels: bool = False,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.smooth_labels = bool(smooth_labels)

    def forward(
        self,
        logits: torch.Tensor,  # [B,1,H,W]
        target: torch.Tensor,  # [B,1,H,W]
    ) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits and target must have same shape, got "
                f"{tuple(logits.shape)} vs {tuple(target.shape)}"
            )

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        valid_mask_f = valid_mask.float()

        # Hard binary target for Dice and for BCE base target.
        # Ignored pixels are safely set to 0 here, then removed by valid_mask.
        target_hard = (target > 0.5).float()
        target_hard = target_hard * valid_mask_f

        # BCE target: optionally smoothed, but only on valid pixels.
        target_bce = target_hard.clone()
        if self.smooth_labels:
            target_bce[valid_mask] = target_bce[valid_mask] * 0.9 + 0.05

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target_bce,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce = (bce * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)

        probs = torch.sigmoid(logits)
        probs = probs * valid_mask_f

        intersection = (probs * target_hard).sum(dim=(0, 2, 3))
        cardinality = (probs + target_hard).sum(dim=(0, 2, 3))

        dice = (
            1.0
            - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        )

        return bce + dice


class CrossEntropyDiceLossNnUnetLike(nn.Module):
    def __init__(self, weight=None, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)

        valid_mask = target != self.ignore_index  # [B, H, W]
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        target_oh = torch.zeros_like(probs).scatter_(1, safe_target.unsqueeze(1), 1)

        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        probs = probs * valid_mask
        target_oh = target_oh * valid_mask

        # sum over batch + spatial dims (batch_dice=True), per class
        intersection = (probs * target_oh).sum(dim=(0, 2, 3))
        cardinality = (probs + target_oh).sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        dice = 1 - dice_per_class[1:].mean()  # exclude background (do_bg=False)

        return ce + dice


class CrossEntropyDiceLossOld(nn.Module):
    def __init__(
        self,
        class_weights=None,
        ignore_index=-1,
        dice_weight=0.5,
        ce_weight=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer(
            "w",
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32),
        )
        self.eps = eps

    def forward(self, logits, targets):
        # Cross-Entropy (respects ignore_index)
        ce = F.cross_entropy(
            logits, targets, weight=self.w, ignore_index=self.ignore_index
        )

        # Dice on valid pixels only
        C = logits.shape[1]
        if self.ignore_index is not None and self.ignore_index >= 0:
            logits_v, targets_v, valid_mask = _mask_valid(
                logits, targets, self.ignore_index
            )
            if valid_mask is None:  # no ignore
                probs = F.softmax(logits, dim=1)
                onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
            else:
                probs = F.softmax(logits_v, dim=1)  # [N_valid, C]
                onehot = F.one_hot(targets_v, C).float()  # [N_valid, C]
                # reshape to [1,C,N_valid] for classwise dice
                probs = probs.t().unsqueeze(0)
                onehot = onehot.t().unsqueeze(0)
                # compute dice below on flattened form
                inter = (probs * onehot).sum(dim=(0, 2))
                denom = probs.sum(dim=(0, 2)) + onehot.sum(dim=(0, 2))
                dice_per_class = (2 * inter + self.eps) / (denom + self.eps)
                dice_loss = 1.0 - dice_per_class.mean()
                return self.ce_weight * ce + self.dice_weight * dice_loss

        # no ignore path (faster)
        probs = F.softmax(logits, dim=1)  # [B,C,H,W]
        onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        inter = (probs * onehot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice_per_class = (2 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


class FocalCrossEntropy(nn.Module):
    def __init__(
        self, gamma=2.0, class_weights=None, ignore_index=-1, reduction="mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer(
            "w",
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, logits, targets):
        # CE per-pixel (no reduction), then apply focal modulating term
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.w,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        # get p_t = exp(-ce)
        pt = torch.exp(-ce)
        focal = (1 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
