from __future__ import annotations

from typing import List, Tuple
import os
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import yaml
except ImportError:
    yaml = None

from datasets.anorak import ANORAKFewShot
from models.histo_protonet_decoder import ProtoNetDecoder
from training.tiler import GridPadTiler, Tiler
from models.histo_encoder import Encoder


# ---------------------------
# helpers
# ---------------------------


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
) -> list[torch.Tensor]:
    """Convert list of instance masks into a single-channel semantic map per image."""
    out: list[torch.Tensor] = []
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
    targets_sem: torch.Tensor,  # [B,1,H,W] or [B,H,W] long
    grid_size: Tuple[int, int],  # (Ht, Wt)
) -> torch.Tensor:
    """
    Very simple mask -> token mapping:

      - Downsample semantic mask with nearest neighbor to token grid.
      - Each token gets one hard label in [0..C-1] or ignore_idx.

    Returns:
      y_tokens: [B, Q] long   (token labels; may include ignore_idx)
      valid:    [B, Q] bool   (True if label != ignore_idx)
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
    X: torch.Tensor,  # [N, D]
    y: torch.Tensor,  # [N]
    img_ids: torch.Tensor,  # [N] (global image index for each token)
    num_classes: int,
    max_tokens_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each class c, sample up to max_tokens_per_class tokens,
    while covering as many different images as possible:

      - build mapping (class c) -> {image_id -> list(token_indices)}
      - for each class, round-robin over images: pick 1 token from each image
        until we reach max_tokens_per_class or run out.
    """
    if max_tokens_per_class <= 0:
        return X, y, img_ids

    N = X.shape[0]
    assert y.shape[0] == N and img_ids.shape[0] == N

    selected_indices: list[torch.Tensor] = []

    for c in range(num_classes):
        class_mask = y == c
        idx_c = torch.where(class_mask)[0]  # token indices for class c

        if idx_c.numel() == 0:
            continue

        if idx_c.numel() <= max_tokens_per_class:
            selected_indices.append(idx_c)
            continue

        # Map image_id -> list of token indices (for this class)
        img_ids_c = img_ids[idx_c]
        per_img: dict[int, torch.Tensor] = {}

        for local_i, token_idx in enumerate(idx_c):
            img_id_int = int(img_ids_c[local_i].item())
            if img_id_int not in per_img:
                per_img[img_id_int] = []
            per_img[img_id_int].append(token_idx)

        # convert lists to shuffled tensors
        for img_id in per_img:
            arr = torch.tensor(per_img[img_id], dtype=torch.long)
            perm = torch.randperm(arr.numel())
            per_img[img_id] = arr[perm]

        # round-robin sampling
        chosen_for_c: list[torch.Tensor] = []
        count = 0
        # keep iterating until we either fill the quota or everything is empty
        while count < max_tokens_per_class:
            all_empty = True
            for img_id, arr in per_img.items():
                if arr.numel() == 0:
                    continue
                all_empty = False
                chosen_for_c.append(arr[0:1])  # pick one token from this image
                per_img[img_id] = arr[1:]  # remove it
                count += 1
                if count >= max_tokens_per_class:
                    break
            if all_empty:
                break

        if chosen_for_c:
            selected_indices.append(torch.cat(chosen_for_c, dim=0))

    if not selected_indices:
        # fallback: no tokens selected at all
        return X.new_empty((0, X.shape[1])), y.new_empty((0,)), img_ids.new_empty((0,))

    idx = torch.cat(selected_indices, dim=0)
    return X[idx], y[idx], img_ids[idx]


def load_leace_proj(path: str, in_dim: int) -> torch.Tensor:
    """
    Load a LEACE projection matrix from a .pt file.

    Accepts either:
      - a Tensor of shape [in_dim, D_leace], or
      - a dict containing key 'proj_matrix' (or 'P') with that tensor.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "proj_matrix" in obj:
            P = obj["proj_matrix"]
        elif "P" in obj:
            P = obj["P"]
        else:
            raise ValueError(
                f"LEACE file {path} is a dict but has no 'proj_matrix' or 'P' key."
            )
    elif torch.is_tensor(obj):
        P = obj
    else:
        raise TypeError(
            f"LEACE file {path} must be a Tensor or dict with 'proj_matrix'/'P'."
        )

    if P.ndim != 2:
        raise ValueError(f"LEACE proj_matrix must be 2D, got shape {tuple(P.shape)}.")
    if P.shape[0] != in_dim:
        raise ValueError(
            f"LEACE proj_matrix first dim {P.shape[0]} != encoder dim {in_dim}."
        )
    return P.float()


@torch.no_grad()
def accumulate_features_and_labels(
    encoder: Encoder,
    dataloader: DataLoader,
    img_tiler: GridPadTiler,
    target_tiler: GridPadTiler,
    grid_size: Tuple[int, int],
    ignore_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Ht, Wt = grid_size

    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    img_ids_list: list[torch.Tensor] = []

    encoder.eval()
    encoder.to(device)

    global_img_offset = 0

    grid_sum = torch.zeros(Ht, Wt, encoder.embed_dim, device=device)
    grid_count = 0
    tile_size = img_tiler.tile

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

        y_tokens = masks_to_token_hard_nearest(
            tgt_crops,
            grid_size=grid_size,
        )  # [Nc, Q]

        X_grid = tokens.reshape(Nc, Ht, Wt, D)  # [Nc, Ht, Wt, D]

        # update grid-wise sum
        grid_sum += X_grid.sum(dim=0)  # [Ht, Wt, D]
        grid_count += Nc

        y = y_tokens.reshape(Nc * Q)
        token_img_ids = crop_img_ids.unsqueeze(1).expand(Nc, Q).reshape(Nc * Q)

        if X_grid.numel() == 0:
            continue

        X_list.append(X_grid.cpu())
        y_list.append(y.cpu())
        img_ids_list.append(token_img_ids.cpu())

    if not X_list:
        return (
            torch.empty(0, encoder.embed_dim),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    grid_mean = (grid_sum / grid_count).cpu()  # [Ht, Wt, D]

    # subtract mean and flatten, then mask ignore_idx
    for idx, X_grid in enumerate(X_list):
        X_grid = X_grid - grid_mean  # [Nc, Ht, Wt, D]
        y = y_list[idx]
        m = y != ignore_idx
        if not m.any():
            # all tokens ignored -> drop this batch
            X_list[idx] = torch.empty(0, encoder.embed_dim)
            img_ids_list[idx] = torch.empty(0, dtype=torch.long)
            y_list[idx] = torch.empty(0, dtype=torch.long)
            continue

        X_flat = X_grid.reshape(-1, encoder.embed_dim)[m]
        y_list[idx] = y[m]
        img_ids_list[idx] = img_ids_list[idx][m]
        X_list[idx] = X_flat

    X_all = torch.cat(X_list, dim=0)  # [N, D]
    y_all = torch.cat(y_list, dim=0)  # [N]
    img_ids_all = torch.cat(img_ids_list, dim=0)

    return X_all, y_all, img_ids_all


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
    dm = ANORAKFewShot(
        root=root_dir,
        devices=devices,  # list[int]
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


# ---------------------------
# Click CLI
# ---------------------------


@click.command()
@click.option("--root-dir", type=str, required=True, help="ANORAK dataset root.")
@click.option("--fold", type=int, default=0, show_default=True)
@click.option(
    "--img-size",
    type=int,
    nargs=2,
    default=(448, 448),
    show_default=True,
    metavar="H W",
    help="Working crop size and ViT input size (must be square for now).",
)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--num-classes", type=int, default=7, show_default=True)
@click.option("--ignore-idx", type=int, default=255, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--prefetch-factor", type=int, default=2, show_default=True)
@click.option("--num-metrics", type=int, default=1, show_default=True)
@click.option(
    "--weighted-blend",
    is_flag=True,
    default=False,
    help="Use Hann weighting during stitching (for images only).",
)
# model / encoder
@click.option(
    "--encoder-id",
    type=str,
    default="h0-mini",
    show_default=True,
    help="Backbone encoder id.",
)
@click.option("--ckpt-path", type=str, default="", show_default=True)
@click.option(
    "--metric",
    type=click.Choice(["L2", "cosine"], case_sensitive=False),
    default="L2",
    show_default=True,
)
@click.option("--no-center-feats", is_flag=True, default=False)
@click.option("--no-normalize-feats", is_flag=True, default=False)
# PCA / proto
@click.option(
    "--proj-dim",
    type=int,
    default=None,
    help="PCA dimension (n_components). If <=0 and --pca-evr<=0, PCA is disabled.",
)
@click.option(
    "--pca-evr",
    type=float,
    default=0.0,
    show_default=True,
    help="PCA explained variance ratio (0<evr<=1). Mutually exclusive with proj-dim>0.",
)
@click.option(
    "--max-tokens-per-class",
    type=int,
    default=0,
    show_default=True,
    help="Max tokens per class (balanced across images). <=0 means use all tokens.",
)
# LEACE
@click.option(
    "--leace-proj-path",
    type=str,
    default="",
    show_default=True,
    help="Optional path to a LEACE projection (.pt). Applied FIRST: X -> X @ P_leace.",
)
# device / output
@click.option(
    "--device",
    "-d",
    type=int,
    multiple=True,
    default=(0,),
    show_default=True,
    help="GPU ids (first is used for encoder).",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Where to save mean/proj_matrix/prototypes (.pt).",
)
@click.option(
    "--yaml-path",
    type=str,
    default="",
    show_default=False,
    help="Optional path to save a lightweight YAML description. "
    "If empty, uses output-path with .yaml extension.",
)
def main(
    root_dir: str,
    fold: int,
    img_size: Tuple[int, int],
    batch_size: int,
    num_classes: int,
    ignore_idx: int,
    num_workers: int,
    prefetch_factor: int,
    num_metrics: int,
    weighted_blend: bool,
    encoder_id: str,
    ckpt_path: str,
    metric: str,
    no_center_feats: bool,
    no_normalize_feats: bool,
    proj_dim: int | None,
    pca_evr: float,
    max_tokens_per_class: int,
    leace_proj_path: str,
    device: tuple[int, ...],
    output_path: str | Path,
    yaml_path: str | Path,
) -> None:
    """Fit ProtoNet mean/PCA/LEACE/prototypes from ANORAK segmentation data."""
    devices = list(device)
    dev = parse_devices(devices)
    img_size = (int(img_size[0]), int(img_size[1]))

    # for now we assume square crops so tiler.tile is a single int
    if img_size[0] != img_size[1]:
        raise ValueError(f"--img-size must be square for now, got {img_size}.")
    tile_size = img_size[0]

    # interpret center/norm flags (shared with ProtoNet head)
    center_enabled = not no_center_feats
    norm_enabled = not no_normalize_feats

    # 1) Build encoder/decoder
    encoder = (
        Encoder(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=False,
            ckpt_path=ckpt_path,
        )
        .to(dev)
        .eval()
    )

    # we now know the ViT token grid from the decoder
    grid_size = tuple(encoder.grid_size)  # (Ht, Wt)

    # 2) Build tilers (image: replicate; targets: constant ignore_idx)
    img_tiler = GridPadTiler(
        tile=tile_size,
        stride=tile_size,
        weighted_blend=weighted_blend,
        pad_mode="replicate",
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=tile_size,
        stride=tile_size,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=float(ignore_idx),
    )

    # 3) Build train dataloader
    train_loader = build_train_loader(
        root_dir=root_dir,
        devices=devices,
        batch_size=1,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=num_classes,
        num_metrics=num_metrics,
        ignore_idx=ignore_idx,
        fold=fold,
        prefetch_factor=prefetch_factor,
    )

    # 4) Accumulate ALL token features + labels + image ids
    click.echo("Accumulating features and token labels...")
    X_all, y_all, img_ids_all = accumulate_features_and_labels(
        encoder=encoder,
        dataloader=train_loader,
        img_tiler=img_tiler,
        target_tiler=tgt_tiler,
        grid_size=grid_size,
        ignore_idx=ignore_idx,
        device=dev,
    )

    if X_all.shape[0] == 0:
        raise RuntimeError("No valid tokens collected – check masks & ignore_idx.")

    click.echo(
        f"Collected {X_all.shape[0]} valid tokens, feature dim = {X_all.shape[1]}."
    )

    # 5) Optional: rebalance tokens by class with maximum image coverage
    X_used, y_used, img_ids_used = subsample_tokens_balanced_by_image(
        X_all,
        y_all,
        img_ids_all,
        num_classes=num_classes,
        max_tokens_per_class=max_tokens_per_class,
    )
    click.echo(
        f"Using {X_used.shape[0]} tokens after balancing "
        f"(max_tokens_per_class={max_tokens_per_class})."
    )

    if X_used.shape[0] == 0:
        raise RuntimeError("No tokens left after balancing – decrease constraints.")

    N_used, encoder_dim = X_used.shape

    # 6) Optional LEACE projection (first projection)
    leace_proj = None
    if leace_proj_path:
        click.echo(f"Loading LEACE projection from {leace_proj_path}...")
        leace_proj = load_leace_proj(leace_proj_path, encoder_dim)
        X_stage0 = X_used @ leace_proj  # [N_used, D_leace]
        stage0_dim = leace_proj.shape[1]
    else:
        X_stage0 = X_used
        stage0_dim = encoder_dim

    # 7) Centering (optional; in stage0 space)
    if center_enabled:
        mean = X_stage0.mean(dim=0)  # [stage0_dim]
        X_centered = X_stage0 - mean
    else:
        mean = torch.zeros(stage0_dim, dtype=X_stage0.dtype)
        X_centered = X_stage0

    # 8) PCA (optional) – choose by proj-dim OR explained variance ratio
    proj_dim_val = proj_dim if proj_dim is not None else 0
    pca_evr_val = float(pca_evr) if pca_evr is not None else 0.0

    if proj_dim_val > 0 and pca_evr_val > 0.0:
        raise ValueError("Use either --proj-dim or --pca-evr, not both.")

    pca_enabled = (proj_dim_val > 0) or (pca_evr_val > 0.0)

    if pca_enabled:
        if PCA is None:
            raise ImportError(
                "scikit-learn is required for PCA. Install with `pip install scikit-learn`."
            )

        if proj_dim_val > 0:
            click.echo(f"Fitting PCA to dimension {proj_dim_val}...")
            pca = PCA(n_components=proj_dim_val)
        else:
            if not (0.0 < pca_evr_val <= 1.0):
                raise ValueError("--pca-evr must be in (0,1] when used.")
            click.echo(
                f"Fitting PCA to reach explained variance ratio {pca_evr_val}..."
            )
            pca = PCA(n_components=pca_evr_val)

        X_np = X_centered.numpy()
        Z_np = pca.fit_transform(X_np)  # [N_used, D_proj]
        Z = torch.from_numpy(Z_np).float()
        proj_matrix = torch.from_numpy(
            pca.components_.T
        ).float()  # [stage0_dim, D_proj]
    else:
        click.echo("PCA disabled – using identity projection.")
        proj_matrix = torch.eye(stage0_dim, stage0_dim, dtype=torch.float32)
        Z = X_centered

    # 9) Normalization (optional) in final space
    if norm_enabled:
        Z = F.normalize(Z, dim=-1)
    D_proj = Z.shape[1]

    # 10) Compute class prototypes (on used tokens in final space)
    click.echo("Computing prototypes...")
    C = num_classes
    prototypes = torch.zeros(C, D_proj, dtype=torch.float32)
    class_counts = torch.zeros(C, dtype=torch.float32)

    for c in range(C):
        mask_c = y_used == c
        if mask_c.any():
            Zc = Z[mask_c]
            prototypes[c] = Zc.mean(dim=0)
            class_counts[c] = float(mask_c.sum().item())

    # 11) Save everything (.pt)
    payload = {
        # stats for the pipeline
        "mean": mean,  # [stage0_dim]
        "proj_matrix": proj_matrix,  # [stage0_dim, D_proj]
        "prototypes": prototypes,  # [C, D_proj]
        "class_counts": class_counts,  # [C]
        "num_classes": C,
        "embedding_dim": encoder_dim,  # original encoder feature dim
        "encoder_dim": encoder_dim,
        "stage0_dim": stage0_dim,  # dim after LEACE (if any)
        "proj_dim": D_proj,  # final feature dim used in head
        "leace_proj_matrix": leace_proj,  # None or [encoder_dim, stage0_dim]
        # config for ProtoNet head reconstruction
        "head_config": {
            "metric": metric.lower(),
            "center_feats": center_enabled,
            "normalize_feats": norm_enabled,
            "use_leace": bool(leace_proj is not None),
            "use_pca": pca_enabled,
            "num_prototypes": C,
            "embedding_dim": D_proj,
        },
        # extra meta for reproducibility
        "meta": {
            "encoder_id": encoder_id,
            "ckpt_path": ckpt_path,
            "img_size": img_size,
            "grid_size": grid_size,
            "ignore_idx": ignore_idx,
            "tile": tile_size,
            "weighted_blend": weighted_blend,
            "pca_mode": (
                "none"
                if not pca_enabled
                else (
                    "n_components" if proj_dim_val > 0 else "explained_variance_ratio"
                )
            ),
            "proj_dim_arg": proj_dim,
            "pca_evr_arg": pca_evr,
            "max_tokens_per_class": max_tokens_per_class,
            "leace_proj_path": leace_proj_path,
            "num_tokens_used": int(X_used.shape[0]),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    click.echo(
        f"[OK] saved to {output_path}  "
        f"prototypes={list(prototypes.shape)}, mean={list(mean.shape)}, "
        f"final_dim={D_proj}, stage0_dim={stage0_dim}, encoder_dim={encoder_dim}"
    )

    # 12) Save a lightweight YAML description (no tensors)
    if yaml is None:
        if yaml_path:
            click.echo(
                "PyYAML not installed, cannot write YAML description. "
                "Install with `pip install pyyaml`."
            )
        return

    # default yaml_path = same stem as output_path with .yaml
    if not yaml_path:
        stem = os.path.splitext(output_path)[0]
        yaml_path = stem + ".yaml"

    meta = payload["meta"]
    head_cfg = payload["head_config"]

    yaml_desc = {
        "output_path": os.path.abspath(output_path),
        "yaml_path": os.path.abspath(yaml_path),
        "shapes": {
            "mean": list(payload["mean"].shape),
            "proj_matrix": list(payload["proj_matrix"].shape),
            "prototypes": list(payload["prototypes"].shape),
            "class_counts": list(payload["class_counts"].shape),
            "leace_proj_matrix": (
                list(payload["leace_proj_matrix"].shape)
                if payload["leace_proj_matrix"] is not None
                else None
            ),
        },
        "head_config": head_cfg,
        "meta": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()},
            "num_classes": int(payload["num_classes"]),
            "embedding_dim": int(payload["embedding_dim"]),
            "stage0_dim": int(payload["stage0_dim"]),
            "proj_dim": int(payload["proj_dim"]),
        },
    }

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as f:
        yaml.safe_dump(yaml_desc, f, sort_keys=False)

    click.echo(f"[OK] also wrote YAML description to {yaml_path}")


if __name__ == "__main__":
    main()
