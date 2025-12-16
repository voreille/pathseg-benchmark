import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange

from datasets.anorak import ANORAK
from models.histoseg.segmentation_models import HistoViTSeg


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ce_only(outputs: Dict[str, torch.Tensor],
            target: torch.Tensor) -> torch.Tensor:
    """
    Use only the main head (logits). You can add aux heads later.
    - outputs["main"]: [B, C, H, W] (logits)
    - target:          [B, H, W] (Long)
    """
    logits = outputs["main"]
    # if your datamodule returns masks as [B,1,H,W] float -> squeeze + long()
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0, ...]
    target = target.long()
    return F.cross_entropy(logits, target)


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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- datamodule: grab ONE batch ---
    dm = ANORAK(
        "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK",
        devices=1,
        num_workers=0,
        batch_size=1,
        img_size=(448, 448),
    )
    dm.setup()
    loader = dm.train_dataloader()

    images, masks = next(iter(loader))  # one batch
    masks = to_per_pixel_targets_semantic(masks, ignore_idx=255)
    masks = torch.stack(masks)
    images = images / 255.0  # to [0,1] if needed
    images, masks = move_to_device(images,
                                   device), move_to_device(masks, device)

    # --- model ---
    model = HistoViTSeg(num_classes=7)  # your builder (nn.Module)

    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(device)
    model.train()

    print(f"Trainable params: {count_trainable(model):,}")

    # --- optimizer (decoder-only or all trainables) ---
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=3e-4, weight_decay=1e-4)

    # --- overfit loop on the SAME batch ---
    steps = 600
    log_every = 50
    start = time.time()
    for step in trange(steps, desc="one-batch overfit"):
        outputs = model(images)  # expects dict with "main"
        loss = ce_only(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0 or step == 0:
            with torch.no_grad():
                logits = outputs["main"]
                preds = logits.argmax(dim=1)
                # quick pixel accuracy (rough sanity metric)
                tgt = masks[:, 0] if masks.dim(
                ) == 4 and masks.shape[1] == 1 else masks
                tgt = tgt.long()
                correct = (preds == tgt).float().mean().item()
            elapsed = time.time() - start
            print(
                f"[{step+1:4d}/{steps}] loss={loss.item():.4f}  pix-acc={correct:.3f}  ({elapsed:.1f}s)"
            )
            start = time.time()

    print(
        "Done. If loss didn't drop strongly, check masks (dtype/values/interp) and logits pipeline."
    )


if __name__ == "__main__":
    main()
