from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def _dense_mask_target(
    target: Mapping[str, Any],
    *,
    ignore_idx: int,
    target_index: int,
) -> torch.Tensor:
    if "masks" not in target or "labels" not in target:
        raise KeyError(
            f"targets[{target_index}] must contain 'masks' and 'labels'."
        )

    masks = target["masks"]
    labels = target["labels"]
    if not torch.is_tensor(masks) or not torch.is_tensor(labels):
        raise TypeError(
            f"targets[{target_index}]['masks'] and ['labels'] must be tensors."
        )
    if masks.ndim != 3:
        raise ValueError(
            f"targets[{target_index}]['masks'] must have shape [N, H, W], "
            f"got {tuple(masks.shape)}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"targets[{target_index}]['labels'] must have shape [N], "
            f"got {tuple(labels.shape)}."
        )
    if masks.shape[0] != labels.shape[0]:
        raise ValueError(
            f"targets[{target_index}] contains {masks.shape[0]} masks but "
            f"{labels.shape[0]} labels."
        )

    labels = labels.to(device=masks.device, dtype=torch.long)
    dense = torch.full(
        masks.shape[-2:],
        int(ignore_idx),
        dtype=torch.long,
        device=masks.device,
    )
    # Match LightningModule.to_per_pixel_targets_semantic exactly: masks are
    # applied in order, so a later mask wins if targets unexpectedly overlap.
    for mask, label in zip(masks, labels, strict=True):
        dense[mask.bool()] = label
    return dense


def to_dense_semantic_targets(
    targets: Any,
    *,
    ignore_idx: int,
    batch_size: int,
) -> torch.Tensor:
    """Normalize dense or Mask2Former-style semantic targets to ``[B,H,W]``.

    Mask2Former-style inputs are sequences of dictionaries containing
    ``masks: [N,H,W]`` and ``labels: [N]``. Pixels not covered by any mask keep
    ``ignore_idx``; this preserves ignored regions in the source label map.
    """

    if torch.is_tensor(targets):
        dense = targets
    elif isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
        values = tuple(targets)
        if values and all(torch.is_tensor(value) for value in values):
            try:
                dense = torch.stack(values)
            except RuntimeError as error:
                raise ValueError(
                    "Cannot stack dense targets; all validation samples must "
                    "have the same spatial shape within a batch."
                ) from error
        elif values and all(isinstance(value, Mapping) for value in values):
            try:
                dense = torch.stack(
                    tuple(
                        _dense_mask_target(
                            value,
                            ignore_idx=ignore_idx,
                            target_index=index,
                        )
                        for index, value in enumerate(values)
                    )
                )
            except RuntimeError as error:
                raise ValueError(
                    "Cannot stack Mask2Former-style targets; all validation "
                    "samples must have the same spatial shape within a batch."
                ) from error
        else:
            raise TypeError(
                "Targets must be a non-empty sequence containing only dense "
                "tensors or only {'masks', 'labels'} mappings."
            )
    elif isinstance(targets, Mapping):
        dense = _dense_mask_target(
            targets,
            ignore_idx=ignore_idx,
            target_index=0,
        ).unsqueeze(0)
    else:
        raise TypeError(
            "Targets must be a tensor or Mask2Former-style target sequence."
        )

    if dense.ndim == 2 and batch_size == 1:
        dense = dense.unsqueeze(0)
    if dense.ndim == 4 and dense.shape[1] == 1:
        dense = dense[:, 0]
    if dense.ndim != 3:
        raise ValueError(
            "Dense targets must have shape [B,H,W] or [B,1,H,W], "
            f"got {tuple(dense.shape)}."
        )
    if dense.shape[0] != batch_size:
        raise ValueError(
            f"Received {dense.shape[0]} targets for a batch of {batch_size}."
        )
    return dense.long()


def class_fractions(
    targets: torch.Tensor,
    *,
    num_classes: int,
    output_size: tuple[int, int],
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample dense labels into per-token class fractions.

    Returns ``(fractions, valid_fraction)`` with shapes ``[B, C, H, W]`` and
    ``[B, 1, H, W]``.  Fractions sum to one where a token contains at least one
    valid target pixel.  Multiplying the two outputs gives class area fractions
    that correctly account for ignored pixels.
    """

    if targets.ndim == 2:
        targets = targets.unsqueeze(0)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape [B, H, W], [B, 1, H, W], or [H, W]; "
            f"got {tuple(targets.shape)}."
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    targets = targets.long()
    valid = targets != ignore_idx
    invalid_label = valid & ((targets < 0) | (targets >= num_classes))
    if torch.any(invalid_label):
        labels = torch.unique(targets[invalid_label]).detach().cpu().tolist()
        raise ValueError(
            f"Targets contain labels outside [0, {num_classes - 1}]: {labels}."
        )

    safe_targets = targets.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid[:, None]

    class_area = F.adaptive_avg_pool2d(one_hot, output_size)
    valid_fraction = F.adaptive_avg_pool2d(valid[:, None].float(), output_size)
    fractions = class_area / valid_fraction.clamp_min(1e-8)

    return fractions, valid_fraction
