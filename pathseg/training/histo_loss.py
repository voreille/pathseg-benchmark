import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_valid(logits, targets, ignore_idx):
    # logits: [B, C, H, W], targets: [B, H, W]
    if ignore_idx is None or ignore_idx < 0:
        return logits, targets, None
    valid = (targets != ignore_idx)
    if valid.all():
        return logits, targets, None
    # flatten valid positions
    v = valid.view(-1)
    logits = logits.permute(0, 2, 3,
                            1).reshape(-1, logits.shape[1])[v]  # [N_valid, C]
    targets = targets.view(-1)[v]  # [N_valid]
    return logits, targets, valid


class CrossEntropyDiceLoss(nn.Module):

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 ce_w=0.5,
                 dice_w=0.5,
                 smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ce_w, self.dice_w, self.smooth = ce_w, dice_w, smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        # dice over non-ignore pixels
        C = logits.shape[1]
        mask = (target != self.ignore_index)
        probs = torch.softmax(logits, dim=1)
        probs = probs * mask.unsqueeze(1)
        target_oh = torch.zeros_like(probs).scatter_(
            1,
            target.clamp_min(0).unsqueeze(1), 1)
        target_oh = target_oh * mask.unsqueeze(1)
        num = 2 * (probs * target_oh).sum(dim=(0, 2, 3)) + self.smooth
        den = (probs + target_oh).sum(dim=(0, 2, 3)) + self.smooth
        dice = 1 - (num / den).mean()
        return self.ce_w * ce + self.dice_w * dice


class CrossEntropyDiceLossOld(nn.Module):

    def __init__(self,
                 class_weights=None,
                 ignore_index=-1,
                 dice_weight=0.5,
                 ce_weight=0.5,
                 eps=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer(
            "w", None if class_weights is None else torch.tensor(
                class_weights, dtype=torch.float32))
        self.eps = eps

    def forward(self, logits, targets):
        # Cross-Entropy (respects ignore_index)
        ce = F.cross_entropy(logits,
                             targets,
                             weight=self.w,
                             ignore_index=self.ignore_index)

        # Dice on valid pixels only
        C = logits.shape[1]
        if self.ignore_index is not None and self.ignore_index >= 0:
            logits_v, targets_v, valid_mask = _mask_valid(
                logits, targets, self.ignore_index)
            if valid_mask is None:  # no ignore
                probs = F.softmax(logits, dim=1)
                onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1,
                                                                   2).float()
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

    def __init__(self,
                 gamma=2.0,
                 class_weights=None,
                 ignore_index=-1,
                 reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer(
            "w", None if class_weights is None else torch.tensor(
                class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        # CE per-pixel (no reduction), then apply focal modulating term
        ce = F.cross_entropy(logits,
                             targets,
                             weight=self.w,
                             ignore_index=self.ignore_index,
                             reduction="none")
        # get p_t = exp(-ce)
        pt = torch.exp(-ce)
        focal = (1 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
