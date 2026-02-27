# %%
from pathlib import Path
import shutil

import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

# %%
masks_dir = Path(
    "/home/valentin/workspaces/pathseg-benchmark/data/wsi/cellomaps/TCGA_growth_pattern_masks"
)
wsi_dir = Path("/mnt/nas6/data/lung_tcga/data/tcga_luad")
thumbnail_dir = Path("/home/valentin/workspaces/pathseg-benchmark/data/wsi/cellomaps/thumbnails")

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
# %%
# %%
metadata_list = []
for wsi_id in tqdm(wsi_ids_matched, desc="Processing WSIs"):
    wsi_path = wsi_dir / f"{wsi_id}.tif"
    masks_path = [f for f in masks_dir.glob(f"{wsi_id}_*.png")]
    classes = [p.stem.split("_")[1] for p in masks_path]
    mask_width, mask_height = -1, -1
    for mask_path in masks_path:
        mask = Image.open(mask_path)
        if mask_width == -1 and mask_height == -1:
            mask_width, mask_height = mask.size
        else:
            assert (mask_width, mask_height) == mask.size, (
                f"Mask size mismatch for {mask_path}"
            )

    wsi = openslide.OpenSlide(str(wsi_path))
    width, height = wsi.dimensions
    thumbnail_path = thumbnail_dir / f"{wsi_id}.png"
    wsi.get_thumbnail((mask_width, mask_height)).save(thumbnail_path)

    metadata = {
        "wsi_id": wsi_id,
        "wsi_path": str(wsi_path),
        "width": width,
        "height": height,
        "mpp-x": float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan")),
        "mpp-y": float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, "nan")),
        "mask_width": mask_width,
        "mask_height": mask_height,
        "mask_width_downsample_factor": width / mask_width,
        "mask_height_downsample_factor": height / mask_height,
        "Acinar": "Acinar" in classes,
        "Papillary": "Papillary" in classes,
        "Solid": "Solid" in classes,
        "Micropapillary": "Micropapillary" in classes,
        "Lepidic": "Lepidic" in classes,
        "Normal": "Normal" in classes,
    }
    metadata_list.append(metadata)

# %%
metadata_df = pd.DataFrame(metadata_list)

# %%
metadata_df.to_csv("cellomaps_metadata.csv", index=False)

# %%
metadata_df.head()
# %%
wsi_id

# %%
wsi.level_dimensions

# %%
wsi = openslide.OpenSlide(str(wsi_dir / f"TCGA-05-4249-01Z-00-DX1.tif"))

# %%
wsi.level_dimensions


# %%
wsi.level_downsamples

# %%
