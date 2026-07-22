# %%
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import openslide
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import umap
from histoseg_plugin.core.tiling.tile import generate_tiles_from_tissue
from histoseg_plugin.core.tissue.segmentation import segment_tissue
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from pathseg.models.encoder import build_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
def tile_slide(
    wsi: openslide.OpenSlide,
    tile_level: int = 0,
    tile_size: int = 224,
    step_size: int = 224,
) -> np.ndarray:
    contours, holes = segment_tissue(wsi)
    coords = generate_tiles_from_tissue(
        wsi,
        contours,
        holes,
        tile_level=tile_level,
        tile_size=tile_size,
        step_size=step_size,
        max_workers=1,
    )
    return coords


class TileDataset(Dataset):
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        transforms: Optional[Any] = None,
    ):
        self.wsi = wsi
        self.coords = list(coords)
        self.tile_size_lvl = int(attrs["patch_size"])
        self.tile_level = int(attrs["patch_level"])
        self.transforms = transforms or T.ToTensor()

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, y0 = self.coords[idx]
        tile = self.wsi.read_region(
            (int(x0), int(y0)),
            self.tile_level,
            (self.tile_size_lvl, self.tile_size_lvl),
        ).convert("RGB")
        tile = self.transforms(tile)
        return tile, torch.tensor([x0, y0], dtype=torch.long)


def build_tile_transform(
    px_mean: Sequence[float],
    px_std: Sequence[float],
) -> Any:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=px_mean, std=px_std),
        ]
    )


def get_mpp(wsi: openslide.OpenSlide, level: int = 0) -> float:
    mpp_x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is None or mpp_y is None:
        raise ValueError(
            "WSI is missing MPP properties (openslide.mpp-x / openslide.mpp-y)."
        )

    if abs(float(mpp_x) - float(mpp_y)) > 1e-8:
        raise ValueError(
            f"Non-square pixels not supported (mpp_x={mpp_x}, mpp_y={mpp_y})."
        )

    return float(mpp_x)


# %%
slide_path = Path(
    "/home/valentin/workspaces/pathseg-benchmark/data/wsi/CPTAC/C3L-02219-22.svs"
)
slide = openslide.OpenSlide(slide_path)
mpp = get_mpp(slide)
print(f"Slide MPP at level 0: {mpp} microns per pixel")
# %%
slide.level_downsamples
# %%
coords = tile_slide(slide, tile_level=0, tile_size=224, step_size=224)

# %%
model, model_attrs = build_encoder("h0-mini")

# %%
model.to("cuda")
model.eval()
# %%
dataset = TileDataset(
    wsi=slide,
    coords=coords,
    attrs={"patch_size": 224, "patch_level": 0},
    transforms=build_tile_transform(
        px_mean=model_attrs["pixel_mean"],
        px_std=model_attrs["pixel_std"],
    ),
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# %%

embeddings = []
all_coords = []

for batch in tqdm(dataloader):
    tiles, batch_coords = batch
    tiles = tiles.to("cuda", non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # H0-mini/ViT-like output: CLS token
            emb = model(tiles)[:, 0, :]

    embeddings.append(emb.float().cpu())
    all_coords.append(batch_coords.cpu())

embeddings = torch.cat(embeddings, dim=0).numpy()
all_coords = torch.cat(all_coords, dim=0).numpy()

print("embeddings:", embeddings.shape)
print("coords:", all_coords.shape)

# %%

# Optional but usually helpful before UMAP/t-SNE
X = StandardScaler().fit_transform(embeddings)

# Optional PCA pre-reduction before UMAP/t-SNE
# Speeds things up and removes some noise
n_pca = min(50, X.shape[1], X.shape[0])
X_pca = PCA(n_components=n_pca, random_state=0).fit_transform(X)

print("X_pca:", X_pca.shape)

# %%
rng = np.random.default_rng(0)

max_points = 5000  # increase later if useful
n = X_pca.shape[0]

if n > max_points:
    idx = rng.choice(n, size=max_points, replace=False)
else:
    idx = np.arange(n)

X_plot = X_pca[idx]
coords_plot = all_coords[idx]

print("plot points:", X_plot.shape[0])

# %%
# If needed:
# pip install umap-learn plotly


umap_coords = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=0,
).fit_transform(X_plot)

print("umap:", umap_coords.shape)

# %%

tsne_coords = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    metric="euclidean",
    random_state=0,
    verbose=1,
).fit_transform(X_plot)

print("tsne:", tsne_coords.shape)

# %%

df_umap = pd.DataFrame(
    {
        "x": umap_coords[:, 0],
        "y": umap_coords[:, 1],
        "slide_x": coords_plot[:, 0],
        "slide_y": coords_plot[:, 1],
    }
)

fig = px.scatter(
    df_umap,
    x="x",
    y="y",
    hover_data=["slide_x", "slide_y"],
    title="UMAP of tile embeddings",
    width=900,
    height=700,
)

fig.show()

# %%
df_tsne = pd.DataFrame(
    {
        "x": tsne_coords[:, 0],
        "y": tsne_coords[:, 1],
        "slide_x": coords_plot[:, 0],
        "slide_y": coords_plot[:, 1],
    }
)

fig = px.scatter(
    df_tsne,
    x="x",
    y="y",
    hover_data=["slide_x", "slide_y"],
    title="t-SNE of tile embeddings",
    width=900,
    height=700,
)

fig.show()

# %%
out_dir = Path("embedding_explorer")
out_dir.mkdir(exist_ok=True)

fig_umap = px.scatter(
    df_umap,
    x="x",
    y="y",
    hover_data=["slide_x", "slide_y"],
    title="UMAP of tile embeddings",
    width=1000,
    height=800,
)

fig_tsne = px.scatter(
    df_tsne,
    x="x",
    y="y",
    hover_data=["slide_x", "slide_y"],
    title="t-SNE of tile embeddings",
    width=1000,
    height=800,
)

fig_umap.write_html(out_dir / "umap.html")
fig_tsne.write_html(out_dir / "tsne.html")

df_umap.to_csv(out_dir / "umap_points.csv", index=False)
df_tsne.to_csv(out_dir / "tsne_points.csv", index=False)

print("Saved to:", out_dir.resolve())

# %%

