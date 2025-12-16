# %%
import sys

sys.path.append('..')

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm import tqdm
from umap import UMAP

from datasets.anorak import ANORAKFewShot
from histo_utils.macenko_torch import normalize_and_unmix
from models.histo_encoder import Encoder
from models.histo_protonet_decoder import masks_to_token_hard_from_semantic
from training.tiler import GridPadTiler

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
        y = torch.full((h, w),
                       ignore_idx,
                       dtype=t["labels"].dtype,
                       device=t["labels"].device)
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
    purity_thresh: Optional[float] = None,
    renorm_exclude_ignore: bool = True,
    drop_background_only: bool = True,
    progress: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int]]:
    """
    Returns balanced random subset of tokens across classes.
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    torch.manual_seed(seed)

    img_tiler = GridPadTiler(tile=tile_size,
                             stride=stride,
                             weighted_blend=False,
                             pad_mode="replicate",
                             pad_value=0.0)
    tgt_tiler = GridPadTiler(tile=tile_size,
                             stride=stride,
                             weighted_blend=False,
                             pad_mode="constant",
                             pad_value=float(ignore_idx))

    # temporary lists
    all_X: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    it = tqdm(dataloader, desc="gather tokens",
              leave=False) if progress else dataloader
    for imgs, targets in it:
        # imgs_norm = []
        # for img in imgs:
        #     Io = torch.quantile(img.float().reshape(-1, 3), q=0.999, dim=0).max().item()
        #     img_norm, _, _, _ = normalize_and_unmix(img, Io=Io)
        #     img_norm = img_norm.permute(2, 0, 1)
        #     imgs_norm.append(img_norm)

        # crops, _, _ = img_tiler.window(imgs_norm)
        crops, _, _ = img_tiler.window(imgs)
        crops = (crops.to(device) / 255.0)

        sem_list = to_per_pixel_targets_semantic(targets, ignore_idx)
        sem_list = [y.unsqueeze(0) for y in sem_list]
        tgt_crops, _, _ = tgt_tiler.window(sem_list)

        tokens = encoder(crops)  # [N,Q,D]
        N, Q, D = tokens.shape
        y_hard, valid = masks_to_token_hard_from_semantic(
            tgt_crops.to(device),
            num_classes=num_classes,
            grid_size=encoder.grid_size,
            ignore_idx=ignore_idx,
            bg_idx=bg_idx,
            renorm_exclude_ignore=renorm_exclude_ignore,
            drop_background_only=drop_background_only,
            purity_thresh=purity_thresh,
        )

        y_flat = y_hard.reshape(-1)
        m_flat = valid.reshape(-1)
        if not include_background:
            m_flat = m_flat & (y_flat != bg_idx)

        if m_flat.any():
            Xv = tokens.reshape(N * Q, D)[m_flat].cpu()
            yv = y_flat[m_flat].cpu()
            all_X.append(Xv)
            all_y.append(yv)

    if not all_X:
        return torch.empty(0), torch.empty(0, dtype=torch.long), {}, {}

    # concatenate everything
    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    # select randomly up to n_per_class per class
    X_sel, y_sel = [], []
    seen_per_class, kept_per_class = {}, {}

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
        perm = torch.randperm(
            n_cls, generator=torch.Generator().manual_seed(seed))[:n_keep]
        idxs_sel = idxs[perm]

        X_sel.append(X_all[idxs_sel])
        y_sel.append(y_all[idxs_sel])

    X = torch.cat(X_sel, dim=0)
    y = torch.cat(y_sel, dim=0)
    return X, y, seen_per_class, kept_per_class


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

# %%
encoder = Encoder(encoder_id="h0-mini")
device = torch.device("cuda:1")
encoder = encoder.to(device)

# %%
X, y, seen, kept = collect_token_embeddings(
    encoder=encoder.eval(),
    dataloader=train_loader,  # or train_loader
    num_classes=7,
    ignore_idx=255,
    tile_size=448,
    stride=448,
    n_per_class=9000,
    include_background=True,  # often nicer for histo viz
    purity_thresh=0.9,  # require ≥90% pure tokens
)

print("per-class seen:", seen)
print("per-class kept:", kept)
print("X", X.shape, "y", y.shape)

# %%

# or: from sklearn.manifold import TSNE

umap = UMAP(
    n_neighbors=10,
    min_dist=0.05,
    spread=1.5,
    metric="cosine",
    random_state=0,
    n_jobs=24,
)
Z = umap.fit_transform(X.numpy())  # [M,2]

# %%
# Create 7 discrete colors (for classes 0–6)
cmap = ListedColormap(plt.cm.tab10.colors[:7])

# Define bin edges centered on integers 0–6
bounds = np.arange(-0.5, 7.5, 1)  # [-0.5, 0.5, 1.5, ..., 6.5]
norm = BoundaryNorm(bounds, cmap.N, clip=True)

plt.figure(figsize=(7, 7))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=y.numpy(), s=3, cmap=cmap, norm=norm)
cbar = plt.colorbar(sc, ticks=np.arange(0, 7))
cbar.set_label("Class")
plt.title("UMAP of spatial tokens")
plt.show()
# %%
print(encoder.grid_size)
# %%
