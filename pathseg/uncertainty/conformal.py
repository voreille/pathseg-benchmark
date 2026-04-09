from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class ConformalCalibration:
    thresholds: torch.Tensor  # [K]
    alpha: float
    num_classes: int
    ignore_index: int


def _get_head_b_logits(model_output: object) -> torch.Tensor:
    """
    Extract head-B logits from model output.

    Expected:
      - dict with "logits_b", or
      - dict with "logits", or
      - tensor directly
    """
    if isinstance(model_output, dict):
        if "logits_b" in model_output:
            return model_output["logits_b"]
        if "logits" in model_output:
            return model_output["logits"]
        raise KeyError("Model output dict must contain 'logits_b' or 'logits'.")
    if torch.is_tensor(model_output):
        return model_output
    raise TypeError("Unsupported model output type.")


def _get_quantile(scores: torch.Tensor, alpha: float) -> float:
    """
    Standard conformal quantile using ceil((n+1)*(1-alpha))/n.

    Returns a scalar float threshold.
    """
    if scores.numel() == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores.")

    scores = torch.sort(scores).values
    n = scores.numel()
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(max(k, 1), n)
    return float(scores[k - 1].item())


@torch.no_grad()
def fit_class_conditional_conformal(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    alpha: float = 0.1,
    device: str = "cuda",
    target_key: Optional[str] = None,
) -> ConformalCalibration:
    """
    Fit class-conditional conformal thresholds q_c from a validation/calibration loader.

    Nonconformity score:
        s = 1 - p_true

    Args:
        model: segmentation model
        val_dataloader: loader yielding batches
        num_classes: number of head-B classes
        ignore_index: ignored label in target
        alpha: miscoverage level (0.1 -> ~90% coverage)
        device: model/input device
        target_key: if batch is a dict, use this key for target mask.
                    If None, tries common keys.

    Returns:
        ConformalCalibration with thresholds [K]
    """
    model.eval()
    model.to(device)

    scores_by_class = [[] for _ in range(num_classes)]

    for batch in val_dataloader:
        # -------- get inputs / targets --------
        if isinstance(batch, dict):
            x = batch["image"]
            if target_key is not None:
                y = batch[target_key]
            else:
                # adapt if needed
                for k in ["mask_b", "target_b", "y_b", "label_b", "mask", "target"]:
                    if k in batch:
                        y = batch[k]
                        break
                else:
                    raise KeyError("Could not infer head-B target key from batch dict.")
        elif isinstance(batch, (list, tuple)):
            # adapt if your batch structure differs
            if len(batch) < 2:
                raise ValueError("Expected batch to contain at least (x, y).")
            x, y = batch[0], batch[1]
        else:
            raise TypeError("Unsupported batch type.")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()  # [B,H,W]

        # -------- forward --------
        out = model(x)
        logits = _get_head_b_logits(out)  # [B,K,H,W]

        if logits.shape[-2:] != y.shape[-2:]:
            raise ValueError(
                f"Logits spatial shape {logits.shape[-2:]} != target shape {y.shape[-2:]}"
            )

        probs = F.softmax(logits, dim=1)  # [B,K,H,W]

        # p_true: [B,H,W]
        valid = (y != ignore_index) & (y >= 0) & (y < num_classes)
        if not valid.any():
            continue

        y_clamped = y.clamp(min=0, max=num_classes - 1)
        p_true = probs.gather(1, y_clamped.unsqueeze(1)).squeeze(1)

        scores = 1.0 - p_true  # [B,H,W]

        # collect class-conditional scores
        for c in range(num_classes):
            mask_c = valid & (y == c)
            if mask_c.any():
                scores_by_class[c].append(scores[mask_c].detach().cpu())

    thresholds = torch.empty(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        if len(scores_by_class[c]) == 0:
            # no calibration examples for this class
            thresholds[c] = 1.0
            continue
        class_scores = torch.cat(scores_by_class[c], dim=0)
        thresholds[c] = _get_quantile(class_scores, alpha=alpha)

    return ConformalCalibration(
        thresholds=thresholds,
        alpha=alpha,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )


@torch.no_grad()
def conformal_prediction_sets_from_logits(
    logits_b: torch.Tensor,
    calibration: ConformalCalibration,
) -> torch.Tensor:
    """
    Build conformal prediction sets from head-B logits.

    Args:
        logits_b: [B,K,H,W]
        calibration: fitted conformal thresholds

    Returns:
        pred_set: bool tensor [B,K,H,W]
            pred_set[:, c] == True where class c is plausible
    """
    probs = F.softmax(logits_b, dim=1)  # [B,K,H,W]
    thresholds = calibration.thresholds.to(logits_b.device).view(1, -1, 1, 1)

    # include class c if 1 - p_c <= q_c  <=>  p_c >= 1 - q_c
    pred_set = (1.0 - probs) <= thresholds
    return pred_set


@torch.no_grad()
def compute_conformal_area_bounds(
    logits_b: torch.Tensor,
    calibration: ConformalCalibration,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute simple per-class area summaries from conformal sets.

    Args:
        logits_b: [B,K,H,W]
        calibration: fitted thresholds
        valid_mask: optional bool mask [B,H,W] for valid pixels

    Returns dict with:
        - argmax_area:   [B,K]
        - certain_area:  [B,K]
        - possible_area: [B,K]
        - ambiguous_area:[B]
        - set_size_map:  [B,H,W]
    """
    pred_set = conformal_prediction_sets_from_logits(logits_b, calibration)  # [B,K,H,W]
    probs = F.softmax(logits_b, dim=1)
    pred = probs.argmax(dim=1)  # [B,H,W]

    B, K, H, W = logits_b.shape

    if valid_mask is None:
        valid_mask = torch.ones((B, H, W), dtype=torch.bool, device=logits_b.device)

    set_size = pred_set.sum(dim=1)  # [B,H,W]
    ambiguous = (set_size > 1) & valid_mask
    singleton = (set_size == 1) & valid_mask

    argmax_area = torch.zeros((B, K), dtype=torch.float32, device=logits_b.device)
    certain_area = torch.zeros((B, K), dtype=torch.float32, device=logits_b.device)
    possible_area = torch.zeros((B, K), dtype=torch.float32, device=logits_b.device)

    for c in range(K):
        argmax_area[:, c] = ((pred == c) & valid_mask).sum(dim=(1, 2)).float()

        # "certain": singleton set and argmax == c
        certain_area[:, c] = (singleton & (pred == c)).sum(dim=(1, 2)).float()

        # "possible": class c is in the conformal set
        possible_area[:, c] = (pred_set[:, c] & valid_mask).sum(dim=(1, 2)).float()

    ambiguous_area = ambiguous.sum(dim=(1, 2)).float()

    return {
        "argmax_area": argmax_area,
        "certain_area": certain_area,
        "possible_area": possible_area,
        "ambiguous_area": ambiguous_area,
        "set_size_map": set_size,
        "pred_set": pred_set,
    }

if __name__ == "__main__":
    num_classes_b = 7
    ignore_index = 255

    calibration = fit_class_conditional_conformal(
        model=model,
        val_dataloader=val_dataloader,
        num_classes=num_classes_b,
        ignore_index=ignore_index,
        alpha=0.1,   # ~90% coverage
        device="cuda",
        target_key="mask_b",   # adapt to your batch format
    )

    print(calibration.thresholds)