# %%
import sys

sys.path.append("..")

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from umap import UMAP

from datasets.anorak import ANORAKFewShot
from histo_utils.macenko_torch import normalize_and_unmix
from models.histo_linear_decoder import LinearDecoderBackbone
from training.tiler import GridPadTiler
from training.linear_semantic import LinearSemantic

# %%
root_dir = "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# %%
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
            (h, w), ignore_idx, dtype=t["labels"].dtype, device=t["labels"].device
        )
        for i, m in enumerate(t["masks"]):
            y[m] = t["labels"][i]
        out.append(y)  # [H,W] long
    return out


# %%
def _device_of(mod: torch.nn.Module) -> torch.device:
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


def _labels_nearest_downsample(
    tgt_crops: torch.Tensor,  # (N,1,H,W) long
    grid_size: Tuple[int, int],  # (Gh, Gw) e.g. (32,32)
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
    valid = ys != ignore_idx  # (N, Gh, Gw)
    y_hard = ys.reshape(N, Gh * Gw)  # (N, Q)
    valid = valid.reshape(N, Gh * Gw)  # (N, Q)
    return y_hard, valid


def _labels_with_purity_pool(
    tgt_crops: torch.Tensor,  # (N,1,H,W) long
    num_classes: int,
    ignore_idx: int,
    patch_size: int,  # e.g. 14 for 448->32
    bg_idx: int = 0,
    purity_thresh: Optional[float] = None,  # e.g. 0.7
    renorm_exclude_ignore: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precision path: compute class proportions per token by average-pooling
    one-hot maps over non-overlapping windows of size (patch_size, patch_size).
    Returns y_hard: (N, Q) long, valid: (N, Q) bool
    """
    N, _, H, W = tgt_crops.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        "tile_size must be divisible by patch_size"
    )
    Gh, Gw = H // patch_size, W // patch_size
    Q = Gh * Gw

    lab = tgt_crops.squeeze(1)  # (N,H,W), long
    # class maps (exclude ignore):
    one_hot = (
        F.one_hot(torch.clamp(lab, 0, num_classes - 1), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )  # (N,C,H,W)

    # ignore fraction per token
    ignore_map = (lab == ignore_idx).float().unsqueeze(1)  # (N,1,H,W)

    # average over patch windows
    pool = torch.nn.AvgPool2d(
        kernel_size=patch_size, stride=patch_size, ceil_mode=False
    )
    cls_frac = pool(one_hot)  # (N,C,Gh,Gw), each entry in [0,1]
    ign_frac = pool(ignore_map)  # (N,1,Gh,Gw)

    if renorm_exclude_ignore:
        denom = (1.0 - ign_frac).clamp_min(1e-6)  # avoid /0
        cls_frac = cls_frac / denom  # re-normalize over non-ignore pixels
        cls_frac = cls_frac.clamp(0, 1)

    # hard label and purity
    purity, y_hard = cls_frac.max(dim=1)  # purity: (N,Gh,Gw), y_hard: (N,Gh,Gw)

    # validity: not fully ignore; and (optional) purity threshold
    valid = ign_frac.squeeze(1) < 1.0  # has at least one non-ignore pixel
    if purity_thresh is not None:
        valid = valid & (purity >= purity_thresh)

    y_hard = y_hard.reshape(N, Q)  # (N,Q)
    valid = valid.reshape(N, Q)  # (N,Q)
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
    patch_size: Optional[int] = None,  # needed when using purity path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y_hard, valid):
      y_hard: (N,Q) long in [0..num_classes-1]
      valid:  (N,Q) bool indicating which tokens to keep
    """
    Gh, Gw = grid_size
    _, _, H, W = tgt_crops.shape
    # Decide path: purity vs nearest
    use_purity = purity_thresh is not None

    if use_purity:
        if patch_size is None:
            # infer patch_size from tile and grid
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
        # Fast nearest-downsample path
        y_hard, valid = _labels_nearest_downsample(
            tgt_crops=tgt_crops, grid_size=(Gh, Gw), ignore_idx=ignore_idx
        )

    if drop_background_only:
        # If a whole crop is background/ignore, drop its bg tokens:
        # detect tokens that are bg but came from an all-bg-or-ignore crop.
        N = y_hard.shape[0]
        Q = y_hard.shape[1]
        y_grid = y_hard.reshape(N, Gh, Gw)
        v_grid = valid.reshape(N, Gh, Gw)

        crop_all_bg_or_ignore = ((~v_grid) | (y_grid == bg_idx)).all(dim=(1, 2))  # (N,)
        if crop_all_bg_or_ignore.any():
            m = crop_all_bg_or_ignore.nonzero(as_tuple=True)[0]
            # set bg tokens in those crops as invalid
            v_grid[m] = v_grid[m] & (y_grid[m] != bg_idx)
            valid = v_grid.reshape(N, Q)

    return y_hard, valid


# %%
def stain_normalize_batch(imgs: list[torch.Tensor]) -> list[torch.Tensor]:
    imgs_norm = []
    for img in imgs:
        Io = torch.quantile(img.float().reshape(-1, 3), q=0.999, dim=0).max().item()
        img_norm, _, _, _ = normalize_and_unmix(img, Io=Io)
        img_norm = img_norm.permute(2, 0, 1)
        imgs_norm.append(img_norm)

    return imgs_norm


# %%
@torch.no_grad()
def collect_token_embeddings(
    encoder,
    dataloader,
    *,
    num_classes: int = 7,
    ignore_idx: int = 255,
    tile_size: int = 448,
    stride: int = 448,
    n_per_class: int = 2000,
    bg_idx: int = 0,
    include_background: bool = True,
    purity_thresh: Optional[
        float
    ] = None,  # None => nearest downsample; set e.g. 0.7 for purity pooling
    renorm_exclude_ignore: bool = True,
    drop_background_only: bool = True,
    progress: bool = True,
    stain_normalize: bool = False,
    seed: int = 0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[int, int],
    Dict[int, int],
]:
    device = next(encoder.parameters()).device
    encoder.eval()
    torch.manual_seed(seed)

    img_tiler = GridPadTiler(
        tile=tile_size,
        stride=stride,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=tile_size,
        stride=stride,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=float(ignore_idx),
    )

    all_X, all_y, all_b, all_pos = [], [], [], []

    it = tqdm(dataloader, desc="gather tokens", leave=False) if progress else dataloader

    Ht, Wt = encoder.grid_size
    gy, gx = torch.meshgrid(
        torch.arange(Ht, device=device),
        torch.arange(Wt, device=device),
        indexing="ij",
    )
    positions = torch.stack([gy, gx], dim=-1).reshape(-1, 2).float()  # [Q,2]

    for batch_idx, (imgs, targets) in enumerate(it):
        # images come as list[tensor(H,W,3)] or similar; your tiler handles lists

        if stain_normalize:
            imgs = stain_normalize_batch(imgs)

        crops, _, _ = img_tiler.window(imgs)
        crops = crops.to(device) / 255.0

        sem_list = to_per_pixel_targets_semantic(targets, ignore_idx)
        sem_list = [y.unsqueeze(0) for y in sem_list]  # make (1,H,W)
        tgt_crops, _, _ = tgt_tiler.window(sem_list)  # (N,1,tile,tile), long

        # Encode → tokens [N,Q,D]
        tokens = encoder(crops)  # (N,Q,D)
        N, Q, D = tokens.shape
        X_flat = tokens.reshape(N * Q, D)  # (N*Q,D)
        pos_expanded = positions.unsqueeze(0).expand(N, Q, 2)  # [N,Q,2]
        pos_expanded = pos_expanded.reshape(N * Q, 2)  # [N*Q,2]

        # === NEW: label per token via downsample or purity pooling ===
        y_hard, valid = masks_to_token_labels_from_semantic(
            tgt_crops.to(device).long(),
            num_classes=num_classes,
            grid_size=encoder.grid_size,  # e.g. (32,32)
            ignore_idx=ignore_idx,
            bg_idx=bg_idx,
            purity_thresh=purity_thresh,  # None => nearest
            renorm_exclude_ignore=renorm_exclude_ignore,
            drop_background_only=drop_background_only,
            # patch_size can be inferred; pass explicitly if you prefer:
            # patch_size=tile_size // encoder.grid_size[0],
        )
        # ============================================================

        y_flat = y_hard.reshape(-1)
        m_flat = valid.reshape(-1)
        if not include_background:
            m_flat = m_flat & (y_flat != bg_idx)

        if m_flat.any():
            Xv = X_flat[m_flat].cpu()
            yv = y_flat[m_flat].cpu()
            bv = torch.full((Xv.shape[0],), batch_idx, dtype=torch.long)
            pos_v = pos_expanded[m_flat].cpu()

            all_X.append(Xv)
            all_y.append(yv)
            all_b.append(bv)
            all_pos.append(pos_v)

    if not all_X:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            {},
            {},
        )

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)
    b_all = torch.cat(all_b, dim=0)
    pos_all = torch.cat(all_pos, dim=0)

    # Balanced sampling
    X_sel, y_sel, b_sel, pos_sel = [], [], [], []
    seen_per_class, kept_per_class = {}, {}
    g = torch.Generator().manual_seed(seed)

    for cls in range(num_classes):
        cls_mask = y_all == cls
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
        b_sel.append(b_all[idxs_sel])  # <-- NEW
        pos_sel.append(pos_all[idxs_sel])  # <-- NEW

    X = torch.cat(X_sel, dim=0)
    y = torch.cat(y_sel, dim=0)
    b = torch.cat(b_sel, dim=0)  # <-- NEW
    pos = torch.cat(pos_sel, dim=0)  # <-- NEW
    return X, y, b, pos, seen_per_class, kept_per_class


# %%


def _safe_l2_normalize(X: np.ndarray) -> np.ndarray:
    # row-wise L2 normalization with zero-safe eps
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def embed_pca(
    X,
    seed=42,
    pca_dim=50,
    l2_normalize=True,
):
    # PCA before UMAP often helps + speeds up
    Xr = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed).fit_transform(X)

    if l2_normalize:
        Xr = _safe_l2_normalize(Xr)

    return Xr


def embed_umap(
    X,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    seed=42,
    pca_dim=50,
    l2_normalize=True,
):
    # PCA before UMAP often helps + speeds up
    Xr = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed).fit_transform(X)

    if l2_normalize:
        Xr = _safe_l2_normalize(Xr)

    reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed
    )
    Z = reducer.fit_transform(Xr)
    return Z


def embed_tsne(
    X,
    perplexity=30,
    metric="cosine",
    seed=42,
    pca_dim=50,
    n_iter=1000,
    l2_normalize=True,
):
    Xr = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed).fit_transform(X)

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


def scatter_2d(Z, y, title="UMAP of patch tokens"):
    plt.figure(figsize=(7, 6))
    for c in sorted(np.unique(y)):
        m = y == c
        plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.6, label=str(c))
    plt.legend(title="class", markerscale=4, frameon=False)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.show()


def scatter_by_batch(Z, b, title="UMAP colored by image (batch_idx)", max_legend=25):
    Z = np.asarray(Z)
    b = np.asarray(b)
    uniq = np.unique(b)
    plt.figure(figsize=(7, 6))
    # use a cycling tab20; if >20, matplotlib cycles automatically
    for i, bi in enumerate(uniq):
        m = b == bi
        label = f"img {bi}" if i < max_legend else None
        plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.6, label=label)
    if len(uniq) <= max_legend:
        plt.legend(title="image id", markerscale=4, frameon=False, ncol=2)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.show()


def scatter_by_position_rgb(
    Z: np.ndarray,
    pos: np.ndarray,
    Ht: int,
    Wt: int,
    title: str = "Embedding colored by (gy,gx) position",
):
    # normalize to [0,1]
    gy = pos[:, 0] / max(Ht - 1, 1)
    gx = pos[:, 1] / max(Wt - 1, 1)

    # simple scheme: R = x, B = y, G = 0
    colors = np.stack([gx, np.zeros_like(gx), gy], axis=1)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.7, c=colors)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.show()


# %%
dm = ANORAKFewShot(
    root_dir,
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

val_loader = dm.val_dataloader()

# %%
linear_decoder = LinearDecoderBackbone(encoder_id="h0-mini", num_classes=7, img_size=(448, 448))
pl_model = LinearSemantic.load_from_checkpoint(
    "/home/valentin/workspaces/benchmark-vfm-ss/data/anorak/ulykhaa9/checkpoints/epoch=35-step=40000.ckpt",
    network=linear_decoder,
)
encoder = pl_model.network

# encoder = Encoder(encoder_id="h0-mini")
device = torch.device("cuda:1")
encoder = encoder.to(device)

# %%
X, y, b, pos, seen, kept = collect_token_embeddings(
    encoder=encoder.eval(),
    dataloader=val_loader,  # or train_loader
    num_classes=7,
    ignore_idx=255,
    tile_size=448,
    stride=448,
    n_per_class=9000,
    include_background=True,  # often nicer for histo viz
    drop_background_only=True,
    purity_thresh=None,
    stain_normalize=False,
)

print("per-class seen:", seen)
print("per-class kept:", kept)
print("X", X.shape, "y", y.shape)

Ht, Wt = encoder.grid_size
# %%
Z_pca = embed_pca(X, pca_dim=50, l2_normalize=True)
scatter_2d(Z_pca, y, title="PCA of patch tokens")
# %%
scatter_by_batch(Z_pca, b.numpy(), title="UMAP colored by image (batch_idx)")
# %%
scatter_by_position_rgb(
    Z_pca, pos.numpy(), Ht=Ht, Wt=Wt, title="UMAP colored by position (gy,gx)"
)
# %%
Z_tsne = embed_tsne(X, perplexity=30, metric="cosine", n_iter=1000, l2_normalize=True)
scatter_2d(Z_tsne, y, title="t-SNE of patch tokens")
# %%
scatter_by_batch(Z_tsne, b.numpy(), title="t-SNE colored by image (batch_idx)")
# %%
scatter_by_position_rgb(
    Z_tsne, pos.numpy(), Ht=Ht, Wt=Wt, title="UMAP colored by position (gy,gx)"
)
# %%
from leace.leace import LeaceFitter

device = torch.device("cuda:1")
X_dev = X.to(device)  # [N, D]
pos_dev = pos.to(device)  # [N, 2]

# 1) fit LEACE on positions
fitter = LeaceFitter(
    x_dim=X_dev.shape[1],
    z_dim=2,
    method="leace",  # or "orth"
    device=device,
)
fitter.update(X_dev, pos_dev)
eraser = fitter.eraser  # LeaceEraser

# 2) apply LEACE to embeddings
X_leace = eraser(X_dev).cpu()
# %%
Z_tsne = embed_tsne(
    X_leace, perplexity=30, metric="cosine", n_iter=1000, l2_normalize=True
)
scatter_2d(Z_tsne, y, title="t-SNE of patch tokens")
# %%
scatter_by_batch(Z_tsne, b.numpy(), title="t-SNE colored by image (batch_idx)")
# %%
scatter_by_position_rgb(
    Z_tsne, pos.numpy(), Ht=Ht, Wt=Wt, title="UMAP colored by position (gy,gx)"
)

# %%
Z_umap = embed_umap(X, n_neighbors=15, min_dist=0.1, metric="cosine")
scatter_2d(Z_umap, y, title="UMAP of patch tokens")
# %%
scatter_by_batch(Z_umap, b.numpy(), title="UMAP colored by image (batch_idx)")
