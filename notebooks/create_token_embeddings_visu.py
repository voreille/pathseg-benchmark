#!/usr/bin/env python
"""
Run UMAP / t-SNE visualizations of patch tokens for different settings.

- sweeps over:
    * magnification (via root_dir)
    * pad_mode ("constant" / "replicate")
    * encoder_id (currently fixed, but easy to extend)

- saves figures under: results/token_viz/{encoder_id}/{magnification}/pad-{pad_mode}
- optionally saves embeddings as embeddings.npz
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append("..")  # so that datasets/models modules are found

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from umap import UMAP

from datasets.anorak import ANORAKFewShot
from histo_utils.macenko_torch import normalize_and_unmix
from models.histo_encoder import Encoder
from training.tiler import GridPadTiler

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

GLOBAL_SEED = 1337

# Toggle this to enable / disable saving embeddings (X, y, b) to disk
SAVE_EMBEDDINGS = True

# Base output directory for all results (figures + optional embeddings)
OUT_ROOT = Path("results/token_viz")

# Fixed encoder for now (you can later loop over several)
FIXED_ENCODER_ID = "h0-mini"

# Settings to sweep: magnification + pad_mode
SETTINGS = [
    dict(
        root_dir="/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x",
        magnification_name="10x",
        pad_mode="constant",
    ),
    dict(
        root_dir="/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x",
        magnification_name="10x",
        pad_mode="replicate",
    ),
    dict(
        root_dir="/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK",
        magnification_name="20x",
        pad_mode="constant",
    ),
    dict(
        root_dir="/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK",
        magnification_name="20x",
        pad_mode="replicate",
    ),
]

# =============================================================================
# SEEDING
# =============================================================================


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LABEL UTILITIES
# =============================================================================

@torch.compiler.disable
def to_per_pixel_targets_semantic(
    targets: List[dict],
    ignore_idx: int,
) -> List[torch.Tensor]:
    """Convert list of instance masks into a single-channel semantic map per image."""
    out: List[torch.Tensor] = []
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


def _labels_nearest_downsample(
    tgt_crops: torch.Tensor,  # (N,1,H,W) long
    grid_size: Tuple[int, int],  # (Gh, Gw)
    ignore_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast path: nearest downsample -> per-token label directly.
    Returns y_hard: (N, Q) long, valid: (N, Q) bool
    """
    N, _, H, W = tgt_crops.shape
    Gh, Gw = grid_size
    ys = F.interpolate(tgt_crops.float(), size=(Gh, Gw), mode="nearest").long()
    ys = ys.squeeze(1)  # (N, Gh, Gw)
    valid = (ys != ignore_idx)  # (N, Gh, Gw)
    y_hard = ys.reshape(N, Gh * Gw)  # (N, Q)
    valid = valid.reshape(N, Gh * Gw)  # (N, Q)
    return y_hard, valid


def _labels_with_purity_pool(
    tgt_crops: torch.Tensor,  # (N,1,H,W) long
    num_classes: int,
    ignore_idx: int,
    patch_size: int,
    bg_idx: int = 0,
    purity_thresh: Optional[float] = None,
    renorm_exclude_ignore: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precision path: compute class proportions per token by average-pooling
    one-hot maps over non-overlapping windows of size (patch_size, patch_size).
    Returns y_hard: (N, Q) long, valid: (N, Q) bool
    """
    N, _, H, W = tgt_crops.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "tile_size must be divisible by patch_size"
    Gh, Gw = H // patch_size, W // patch_size
    Q = Gh * Gw

    lab = tgt_crops.squeeze(1)  # (N,H,W), long
    one_hot = F.one_hot(
        torch.clamp(lab, 0, num_classes - 1),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).float()  # (N,C,H,W)

    ignore_map = (lab == ignore_idx).float().unsqueeze(1)  # (N,1,H,W)

    pool = torch.nn.AvgPool2d(kernel_size=patch_size, stride=patch_size, ceil_mode=False)
    cls_frac = pool(one_hot)    # (N,C,Gh,Gw)
    ign_frac = pool(ignore_map) # (N,1,Gh,Gw)

    if renorm_exclude_ignore:
        denom = (1.0 - ign_frac).clamp_min(1e-6)
        cls_frac = cls_frac / denom
        cls_frac = cls_frac.clamp(0, 1)

    purity, y_hard = cls_frac.max(dim=1)  # purity: (N,Gh,Gw), y_hard: (N,Gh,Gw)

    valid = (ign_frac.squeeze(1) < 1.0)
    if purity_thresh is not None:
        valid = valid & (purity >= purity_thresh)

    y_hard = y_hard.reshape(N, Q)
    valid = valid.reshape(N, Q)
    return y_hard, valid


def masks_to_token_labels_from_semantic(
    tgt_crops: torch.Tensor,  # (N,1,tile,tile) long with {0..C-1, ignore}
    *,
    num_classes: int,
    grid_size: Tuple[int, int],  # encoder.grid_size (Gh,Gw)
    ignore_idx: int = 255,
    bg_idx: int = 0,
    purity_thresh: Optional[float] = None,  # if None => fast nearest downsample
    renorm_exclude_ignore: bool = True,
    drop_background_only: bool = True,
    patch_size: Optional[int] = None,  # needed for purity path if not inferrable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y_hard, valid):
      y_hard: (N,Q) long in [0..num_classes-1]
      valid:  (N,Q) bool indicating which tokens to keep
    """
    Gh, Gw = grid_size
    _, _, H, W = tgt_crops.shape
    use_purity = purity_thresh is not None

    if use_purity:
        if patch_size is None:
            assert H % Gh == 0 and W % Gw == 0, "cannot infer patch_size"
            patch_size = H // Gh
        y_hard, valid = _labels_with_purity_pool(
            tgt_crops=tgt_crops,
            num_classes=num_classes,
            ignore_idx=ignore_idx,
            patch_size=patch_size,
            bg_idx=bg_idx,
            purity_thresh=purity_thresh,
            renorm_exclude_ignore=renorm_exclude_ignore,
        )
    else:
        y_hard, valid = _labels_nearest_downsample(
            tgt_crops=tgt_crops,
            grid_size=(Gh, Gw),
            ignore_idx=ignore_idx,
        )

    if drop_background_only:
        N = y_hard.shape[0]
        y_grid = y_hard.reshape(N, Gh, Gw)
        v_grid = valid.reshape(N, Gh, Gw)

        crop_all_bg_or_ignore = ((~v_grid) | (y_grid == bg_idx)).all(dim=(1, 2))
        if crop_all_bg_or_ignore.any():
            m = crop_all_bg_or_ignore.nonzero(as_tuple=True)[0]
            v_grid[m] = v_grid[m] & (y_grid[m] != bg_idx)
            valid = v_grid.reshape(N, Gh * Gw)

    return y_hard, valid


# =============================================================================
# STAIN NORMALIZATION
# =============================================================================

def stain_normalize_batch(imgs: List[torch.Tensor]) -> List[torch.Tensor]:
    imgs_norm: List[torch.Tensor] = []
    for img in imgs:
        Io = torch.quantile(img.float().reshape(-1, 3), q=0.999, dim=0).max().item()
        img_norm, _, _, _ = normalize_and_unmix(img, Io=Io)
        img_norm = img_norm.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
        imgs_norm.append(img_norm)
    return imgs_norm


# =============================================================================
# TOKEN EMBEDDING COLLECTION
# =============================================================================

@torch.no_grad()
def collect_token_embeddings(
    encoder: torch.nn.Module,
    dataloader,
    *,
    num_classes: int = 7,
    ignore_idx: int = 255,
    tile_size: int = 448,
    stride: int = 448,
    n_per_class: int = 2000,
    bg_idx: int = 0,
    include_background: bool = True,
    purity_thresh: Optional[float] = None,
    renorm_exclude_ignore: bool = True,
    drop_background_only: bool = True,
    progress: bool = True,
    stain_normalize: bool = True,
    seed: int = 0,
    img_pad_mode: str = "constant",
    tgt_pad_mode: str = "constant",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int]]:
    device = next(encoder.parameters()).device
    encoder.eval()
    torch.manual_seed(seed)

    img_tiler = GridPadTiler(
        tile=tile_size,
        stride=stride,
        weighted_blend=False,
        pad_mode=img_pad_mode,
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=tile_size,
        stride=stride,
        weighted_blend=False,
        pad_mode=tgt_pad_mode,
        pad_value=float(ignore_idx),
    )

    all_X, all_y, all_b = [], [], []

    it = tqdm(dataloader, desc="gather tokens", leave=False) if progress else dataloader
    for batch_idx, (imgs, targets) in enumerate(it):

        if stain_normalize:
            imgs = stain_normalize_batch(imgs)

        crops, _, _ = img_tiler.window(imgs)
        crops = (crops.to(device) / 255.0)

        sem_list = to_per_pixel_targets_semantic(targets, ignore_idx)
        sem_list = [y.unsqueeze(0) for y in sem_list]  # (1,H,W)
        tgt_crops, _, _ = tgt_tiler.window(sem_list)   # (N,1,tile,tile), long

        tokens = encoder(crops)  # (N,Q,D)
        N, Q, D = tokens.shape

        y_hard, valid = masks_to_token_labels_from_semantic(
            tgt_crops.to(device).long(),
            num_classes=num_classes,
            grid_size=encoder.grid_size,  # e.g. (32,32)
            ignore_idx=ignore_idx,
            bg_idx=bg_idx,
            purity_thresh=purity_thresh,
            renorm_exclude_ignore=renorm_exclude_ignore,
            drop_background_only=drop_background_only,
        )

        y_flat = y_hard.reshape(-1)
        m_flat = valid.reshape(-1)
        if not include_background:
            m_flat = m_flat & (y_flat != bg_idx)

        if m_flat.any():
            Xv = tokens.reshape(N * Q, D)[m_flat].cpu()
            yv = y_flat[m_flat].cpu()
            bv = torch.full((Xv.shape[0],), batch_idx, dtype=torch.long)

            all_X.append(Xv)
            all_y.append(yv)
            all_b.append(bv)

    if not all_X:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            {},
            {},
        )

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)
    b_all = torch.cat(all_b, dim=0)

    X_sel, y_sel, b_sel = [], [], []
    seen_per_class: Dict[int, int] = {}
    kept_per_class: Dict[int, int] = {}
    g = torch.Generator().manual_seed(seed)

    for cls in range(num_classes):
        cls_mask = (y_all == cls)
        n_cls = int(cls_mask.sum())
        seen_per_class[cls] = n_cls
        if n_cls == 0:
            kept_per_class[cls] = 0
            continue
        n_keep = min(n_cls, n_per_class)
        kept_per_class[cls] = n_keep

        idxs = torch.nonzero(cls_mask, as_tuple=True)[0]
        idxs_sel = idxs[torch.randperm(n_cls, generator=g)[:n_keep]]

        X_sel.append(X_all[idxs_sel])
        y_sel.append(y_all[idxs_sel])
        b_sel.append(b_all[idxs_sel])

    X = torch.cat(X_sel, dim=0)
    y = torch.cat(y_sel, dim=0)
    b = torch.cat(b_sel, dim=0)
    return X, y, b, seen_per_class, kept_per_class


# =============================================================================
# DIM-RED + PLOTTING
# =============================================================================

def _safe_l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def embed_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
    pca_dim: int = 50,
    l2_normalize: bool = True,
) -> np.ndarray:
    Xr = PCA(n_components=min(pca_dim, X.shape[1]),
             random_state=seed).fit_transform(X)
    if l2_normalize:
        Xr = _safe_l2_normalize(Xr)

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    Z = reducer.fit_transform(Xr)
    return Z


def embed_tsne(
    X: np.ndarray,
    perplexity: float = 30,
    metric: str = "cosine",
    seed: int = 42,
    pca_dim: int = 50,
    n_iter: int = 1000,
    l2_normalize: bool = True,
) -> np.ndarray:
    Xr = PCA(n_components=min(pca_dim, X.shape[1]),
             random_state=seed).fit_transform(X)
    if l2_normalize:
        Xr = _safe_l2_normalize(Xr)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        metric=metric,
        random_state=seed,
        max_iter=n_iter,
    )
    Z = tsne.fit_transform(Xr)
    return Z


def scatter_2d(
    Z: np.ndarray,
    y: np.ndarray,
    title: str = "UMAP of patch tokens",
    out_path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(7, 6))
    for c in sorted(np.unique(y)):
        m = (y == c)
        plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.6, label=str(c))
    plt.legend(title="class", markerscale=4, frameon=False)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()


def scatter_by_batch(
    Z: np.ndarray,
    b: np.ndarray,
    title: str = "UMAP colored by image (batch_idx)",
    max_legend: int = 25,
    out_path: Optional[Path] = None,
) -> None:
    Z = np.asarray(Z)
    b = np.asarray(b)
    uniq = np.unique(b)
    plt.figure(figsize=(7, 6))
    for i, bi in enumerate(uniq):
        m = (b == bi)
        label = f"img {bi}" if i < max_legend else None
        plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.6, label=label)
    if len(uniq) <= max_legend:
        plt.legend(title="image id", markerscale=4, frameon=False, ncol=2)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()


# =============================================================================
# SINGLE SETTING RUNNER
# =============================================================================

def run_single_setting(
    *,
    root_dir: str,
    magnification_name: str,
    pad_mode: str,
    encoder_id: str,
    out_root: Path,
    seed: int = GLOBAL_SEED,
    save_embeddings: bool = SAVE_EMBEDDINGS,
) -> None:
    set_global_seed(seed)

    # --- datamodule / loader ---
    dm = ANORAKFewShot(
        root_dir=root_dir,
        devices=1,
        num_workers=0,
        fold=0,
        img_size=(448, 448),
        batch_size=1,
        num_classes=7,
        ignore_idx=255,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # --- encoder ---
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_id=encoder_id)
    encoder = encoder.to(device).eval()

    # --- embeddings ---
    X, y, b, seen, kept = collect_token_embeddings(
        encoder=encoder,
        dataloader=train_loader,
        num_classes=7,
        ignore_idx=255,
        tile_size=448,
        stride=448,
        n_per_class=9000,
        include_background=True,   # often nicer for histo viz
        drop_background_only=True,
        purity_thresh=None,        # nearest downsample
        stain_normalize=False,     # toggle if you want
        seed=seed,
        img_pad_mode=pad_mode,
        tgt_pad_mode=pad_mode,
    )

    print(f"[{magnification_name} | pad={pad_mode} | encoder={encoder_id}]")
    print("per-class seen:", seen)
    print("per-class kept:", kept)
    print("X", X.shape, "y", y.shape)

    if X.numel() == 0:
        print("No tokens collected, skipping this setting.")
        return

    X_np = X.numpy()
    y_np = y.numpy()
    b_np = b.numpy()

    # --- output dir ---
    out_dir = out_root / encoder_id / magnification_name / f"pad-{pad_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- optionally save embeddings ---
    if save_embeddings:
        np.savez(
            out_dir / "embeddings.npz",
            X=X_np,
            y=y_np,
            b=b_np,
            seen=np.array(list(seen.items()), dtype=object),
            kept=np.array(list(kept.items()), dtype=object),
        )

    # --- UMAP ---
    Z_umap = embed_umap(
        X_np,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        seed=seed,
        pca_dim=50,
        l2_normalize=True,
    )
    scatter_2d(
        Z_umap,
        y_np,
        title=f"UMAP tokens ({encoder_id}, {magnification_name}, pad={pad_mode})",
        out_path=out_dir / "umap_by_class.png",
    )
    scatter_by_batch(
        Z_umap,
        b_np,
        title=f"UMAP tokens by image ({encoder_id}, {magnification_name}, pad={pad_mode})",
        out_path=out_dir / "umap_by_batch.png",
    )

    # --- t-SNE ---
    Z_tsne = embed_tsne(
        X_np,
        perplexity=30,
        metric="cosine",
        seed=seed,
        pca_dim=50,
        n_iter=1000,
        l2_normalize=True,
    )
    scatter_2d(
        Z_tsne,
        y_np,
        title=f"t-SNE tokens ({encoder_id}, {magnification_name}, pad={pad_mode})",
        out_path=out_dir / "tsne_by_class.png",
    )
    scatter_by_batch(
        Z_tsne,
        b_np,
        title=f"t-SNE tokens by image ({encoder_id}, {magnification_name}, pad={pad_mode})",
        out_path=out_dir / "tsne_by_batch.png",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    for cfg in SETTINGS:
        run_single_setting(
            root_dir=cfg["root_dir"],
            magnification_name=cfg["magnification_name"],
            pad_mode=cfg["pad_mode"],
            encoder_id=FIXED_ENCODER_ID,
            out_root=OUT_ROOT,
            seed=GLOBAL_SEED,
            save_embeddings=SAVE_EMBEDDINGS,
        )


if __name__ == "__main__":
    main()
