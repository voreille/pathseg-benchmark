from pathlib import Path
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def compute_class_weights_from_ids(
        ids,
        masks_dir: Path,
        num_classes: int,
        ignore_idx: int | None = None) -> torch.Tensor:
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    for image_id in ids:
        mask_path = Path(masks_dir) / f"{image_id}.png"
        mask = torch.from_numpy(np.array(Image.open(mask_path),
                                         dtype=np.int64))
        if ignore_idx is not None and ignore_idx >= 0:
            mask = mask[mask != ignore_idx]
        if mask.numel() == 0:
            continue
        class_counts += torch.bincount(mask.view(-1), minlength=num_classes)

    total = class_counts.sum()
    weights = total / (num_classes * torch.clamp(class_counts, min=1))
    return weights.float()
