from __future__ import annotations

from typing import List, Tuple

import click
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.anorak import ANORAKFewShotOld, ANORAKFewShot
from training.tiler import GridPadTiler
from models.histo_encoder import Encoder
from leace.leace import LeaceEraser, LeaceFitter


def parse_devices(devs: List[int]) -> torch.device:
    """Same behavior as your previous CLI: first GPU id, or CPU."""
    if torch.cuda.is_available() and len(devs) > 0:
        torch.cuda.set_device(devs[0])
        return torch.device(f"cuda:{devs[0]}")
    return torch.device("cpu")


@torch.compiler.disable
def to_per_pixel_targets_semantic(
    targets: list[dict],
    ignore_idx: int,
) -> list[Tensor]:
    """Convert list of instance masks into a single-channel semantic map per image."""
    out: list[Tensor] = []
    for t in targets:
        h, w = t["masks"].shape[-2:]
        y = torch.full(
            (h, w),
            ignore_idx,
            dtype=t["labels"].dtype,
            device=t["labels"].device,
        )
        for i, m in enumerate(t["masks"]):
            y[m] = t["labels"][i]
        out.append(y)  # [H,W] long
    return out


@torch.no_grad()
def masks_to_token_hard_nearest(
    targets_sem: Tensor,  # [B,1,H,W] or [B,H,W] long
    grid_size: Tuple[int, int],  # (Ht, Wt)
) -> Tensor:
    """
    Very simple mask -> token mapping:

      - Downsample semantic mask with nearest neighbor to token grid.
      - Each token gets one hard label in [0..C-1] or ignore_idx.

    Returns:
      y_tokens: [B, Q] long   (token labels; may include ignore_idx)
    """
    Ht, Wt = grid_size

    if targets_sem.ndim == 3:
        targets_sem = targets_sem.unsqueeze(1)  # [B,1,H,W]
    assert targets_sem.ndim == 4, "targets_sem must be [B,H,W] or [B,1,H,W]"

    small = F.interpolate(
        targets_sem.float(),
        size=(Ht, Wt),
        mode="nearest",
    ).long()  # [B,1,Ht,Wt]
    small = small.squeeze(1)  # [B,Ht,Wt]

    y_tokens = small.flatten(1)  # [B,Q]
    return y_tokens


def subsample_tokens_balanced_by_image(
    X: Tensor,  # [N, D]
    y: Tensor,  # [N]
    img_ids: Tensor,  # [N]
    num_classes: int,
    max_tokens_per_class: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    For each class c, sample up to max_tokens_per_class tokens,
    while covering as many different images as possible.
    """
    if max_tokens_per_class <= 0:
        return X, y, img_ids

    N = X.shape[0]
    assert y.shape[0] == N and img_ids.shape[0] == N

    selected_indices: list[Tensor] = []

    for c in range(num_classes):
        class_mask = y == c
        idx_c = torch.where(class_mask)[0]

        if idx_c.numel() == 0:
            click.echo(f"Warning: no tokens found for class {c}.")
            continue

        if idx_c.numel() <= max_tokens_per_class:
            click.echo(f"Class {c}: only {idx_c.numel()} tokens available")
            selected_indices.append(idx_c)
            continue

        # Map image_id -> list of token indices (for this class)
        img_ids_c = img_ids[idx_c]
        per_img: dict[int, list[int]] = {}

        for local_i, token_idx in enumerate(idx_c):
            img_id_int = int(img_ids_c[local_i].item())
            if img_id_int not in per_img:
                per_img[img_id_int] = []
            per_img[img_id_int].append(int(token_idx.item()))

        # convert lists to shuffled tensors
        per_img_t: dict[int, Tensor] = {}
        for img_id, idx_list in per_img.items():
            arr = torch.tensor(idx_list, dtype=torch.long)
            perm = torch.randperm(arr.numel())
            per_img_t[img_id] = arr[perm]

        # round-robin sampling
        chosen_for_c: list[Tensor] = []
        count = 0
        while count < max_tokens_per_class:
            all_empty = True
            for img_id, arr in per_img_t.items():
                if arr.numel() == 0:
                    continue
                all_empty = False
                chosen_for_c.append(arr[0:1])
                per_img_t[img_id] = arr[1:]
                count += 1
                if count >= max_tokens_per_class:
                    break
            if all_empty:
                break

        if chosen_for_c:
            selected_indices.append(torch.cat(chosen_for_c, dim=0))

    if not selected_indices:
        return (
            X.new_empty((0, X.shape[1])),
            y.new_empty((0,)),
            img_ids.new_empty((0,)),
        )

    idx = torch.cat(selected_indices, dim=0)
    return X[idx], y[idx], img_ids[idx]


@torch.no_grad()
def accumulate_features_and_labels(
    encoder: Encoder,
    dataloader: DataLoader,
    img_tiler: GridPadTiler,
    target_tiler: GridPadTiler,
    grid_size: Tuple[int, int],
    ignore_idx: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, LeaceEraser | None]:
    """Accumulate encoder token features, class labels, and LEACE stats."""
    Ht, Wt = grid_size

    X_list: list[Tensor] = []
    y_list: list[Tensor] = []
    img_ids_list: list[Tensor] = []

    encoder.eval()
    encoder.to(device)

    global_img_offset = 0
    tile_size = img_tiler.tile

    # positions in token grid, used as concept z
    gy, gx = torch.meshgrid(
        torch.arange(Ht, device=device),
        torch.arange(Wt, device=device),
        indexing="ij",
    )
    positions = torch.stack([gy, gx], dim=-1).reshape(-1, 2).float()  # [Q,2]
    pos_leace_fitter = LeaceFitter(x_dim=encoder.embed_dim, z_dim=2, device=device)

    for imgs, targets in tqdm(dataloader, desc="collect tokens", leave=False):
        B_batch = len(imgs)
        assert B_batch == 1
        img_shape = imgs[0].shape  # [C,H,W]

        if min(img_shape[1:]) < tile_size:
            continue  # skip too small images

        batch_img_ids = torch.arange(
            global_img_offset, global_img_offset + B_batch, dtype=torch.long
        )
        global_img_offset += B_batch

        img_crops, origins, img_sizes = img_tiler.window(imgs)
        img_crops = img_crops.to(device) / 255.0
        Nc = img_crops.shape[0]

        crop_img_ids = torch.empty(Nc, dtype=torch.long)
        for i, ori in enumerate(origins):
            b_idx = ori.img_idx
            crop_img_ids[i] = batch_img_ids[b_idx]

        sem_full = to_per_pixel_targets_semantic(targets, ignore_idx)
        sem_full = [y.unsqueeze(0) for y in sem_full]
        tgt_crops, _, _ = target_tiler.window(sem_full)
        tgt_crops = tgt_crops.to(device)

        tokens = encoder(img_crops)  # [Nc, Q, D]
        Nc, Q, D = tokens.shape
        assert Q == Ht * Wt, f"Q={Q} != Ht*Wt={Ht * Wt}"

        y_tokens = masks_to_token_hard_nearest(
            tgt_crops,
            grid_size=grid_size,
        )  # [Nc, Q]

        valid = y_tokens != ignore_idx  # [Nc, Q] bool

        X_flat = tokens.reshape(Nc * Q, D)  # [Nc*Q, D]
        pos_expanded = positions.unsqueeze(0).expand(Nc, Q, 2)  # [Nc,Q,2]
        pos_expanded = pos_expanded.reshape(Nc * Q, 2)  # [Nc*Q,2]

        # Update LEACE on all tokens (positions concept)
        pos_leace_fitter.update(X_flat, pos_expanded)

        y_flat = y_tokens.reshape(Nc * Q)  # [Nc*Q]
        m = valid.reshape(Nc * Q)  # [Nc*Q]
        token_img_ids = crop_img_ids.unsqueeze(1).expand(Nc, Q).reshape(Nc * Q)

        # keep only valid tokens
        X = X_flat[m].cpu()
        y = y_flat[m].cpu()
        img_ids = token_img_ids[m.cpu()].cpu()

        if X.numel() == 0:
            continue

        X_list.append(X)
        y_list.append(y)
        img_ids_list.append(img_ids)

    if not X_list:
        return (
            torch.empty(0, encoder.embed_dim),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            None,
        )

    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    img_ids_all = torch.cat(img_ids_list, dim=0)

    if pos_leace_fitter.n <= 1:
        pos_eraser = None
    else:
        pos_eraser = pos_leace_fitter.eraser

    return X_all, y_all, img_ids_all, pos_eraser


def build_train_loader(
    root_dir: str,
    devices: List[int],
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int],
    num_classes: int,
    num_metrics: int,
    ignore_idx: int,
    fold: int,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Reuse ANORAKFewShot exactly like your previous CLI."""
    dm = ANORAKFewShotOld(
        root=root_dir,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=num_classes,
        num_metrics=num_metrics,
        ignore_idx=ignore_idx,
        prefetch_factor=prefetch_factor,
        fold=fold,
    )
    dm.setup("fit")
    return dm.train_dataloader()


def build_train_loader_fewshot(
    root_dir: str,
    devices: List[int],
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int],
    num_classes: int,
    num_metrics: int,
    ignore_idx: int,
    fewshot_csv: str,
    n_shot: int = 1,
    support_set: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Reuse ANORAKFewShot exactly like your previous CLI."""
    dm = ANORAKFewShot(
        root=root_dir,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=num_classes,
        num_metrics=num_metrics,
        ignore_idx=ignore_idx,
        prefetch_factor=prefetch_factor,
        n_shot=n_shot,
        support_set=support_set,
        fewshot_csv=fewshot_csv,
        no_train_augmentation=True,
        num_iter_per_epoch=-1,
    )
    dm.setup("fit")
    return dm.train_dataloader()
