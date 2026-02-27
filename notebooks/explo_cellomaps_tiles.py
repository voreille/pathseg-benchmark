# %%
from pathlib import Path

import openslide
import pandas as pd
from PIL import Image
from histoseg_plugin.storage.factory import build_tiling_store_from_dir

# %%
masks_dir = Path(
    "/home/valentin/workspaces/pathseg-benchmark/data/wsi/cellomaps/TCGA_growth_pattern_masks"
)
wsi_dir = Path("/mnt/nas6/data/lung_tcga/data/tcga_luad")

# %%

wsi_ids = [p.stem for p in wsi_dir.glob("*.tif")]
wsi_ids_annotated = list(set([p.stem.split("_")[0] for p in masks_dir.glob("*.png")]))
classes = set([p.stem.split("_")[1] for p in masks_dir.glob("*.png")])
print(f"Total WSIs: {len(wsi_ids)}")
print(f"Total annotated WSIs: {len(wsi_ids_annotated)}")
print(f"Classes: {classes}")
# %%
len(wsi_ids_annotated)

# %%
wsi_ids_matched = [wsi_id for wsi_id in wsi_ids if wsi_id in wsi_ids_annotated]


# %%
assert len(wsi_ids_matched) == len(wsi_ids_annotated)
# %%
wsi = openslide.OpenSlide(str(wsi_dir / f"{wsi_ids_matched[0]}.tif"))

tiles_dir = Path("/home/valentin/workspaces/pathseg-benchmark/data/wsi/cellomaps/tiles")
tiling_store = build_tiling_store_from_dir(
    slides_root=wsi_dir,
    root_dir=tiles_dir,
)
# %%

wsi_id = wsi_ids_matched[0]
coords, cont_idx, attrs = tiling_store.load_coords(wsi_id)
# %%
attrs

# %%
wsi_id

# %%
len(coords)

# %%
type(coords)

# %%
