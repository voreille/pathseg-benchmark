# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from pathseg.utils.load_experiment import load_experiment

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def _make_discrete_cmap(num_classes: int) -> ListedColormap:
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.arange(num_classes))
    return ListedColormap(colors)


num_classes = 7
cmap = _make_discrete_cmap(num_classes=num_classes)

# %%
config_path = "/home/valentin/workspaces/pathseg-benchmark/configs/anorak_ignite_multitask_viz.yaml"
checkpoint_path = "/home/valentin/workspaces/pathseg-benchmark/runs/checkpoints/n5dunhn8/epoch=26-step=40000.ckpt"

bundle = load_experiment(
    config_path=config_path,
    ckpt_path=checkpoint_path,
    stage="predict",
    overrides={
        "data.init_args.num_workers": 0,
        "data.init_args.batch_size": 1,
        "trainer.logger": False,
    },
    device=device,
)

dm = bundle["datamodule"]
model = bundle["model"]
model.eval()
# %%
dm.predict_splits

# %%
data_loader = dm.predict_dataloader()[1]
# %%

for batch_idx, batch in enumerate(data_loader):
    # imgs, targets_raw, img_ids = batch
    imgs, targets, source_ids, image_ids = batch
    if image_ids[0] != "train221_Da454":
        continue
    crops, origins, img_sizes = model.window_imgs_semantic(imgs)
    crops = crops.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        crop_logits = model(crops)
    if isinstance(crop_logits, dict):
        crop_logits = crop_logits["logits_b"] 
    crop_logits = F.interpolate(crop_logits, model.img_size, mode="bilinear")

    crops = crops.cpu()
    crop_logits = crop_logits.cpu()
    break

device = next(model.parameters()).device



# %%
n_crops = len(crops)

fig, axes = plt.subplots(n_crops, 2, figsize=(8, 4 * n_crops))

# If only 1 crop, axes is 1D → make it 2D for consistency
if n_crops == 1:
    axes = axes.reshape(1, -1)

for crop_idx, crop in enumerate(crops):
    # Left: image
    axes[crop_idx, 0].imshow(crop.permute(1, 2, 0))
    axes[crop_idx, 0].set_title(f"Crop {crop_idx}")
    axes[crop_idx, 0].axis("off")

    # Right: logits
    axes[crop_idx, 1].imshow(
        crop_logits[crop_idx].argmax(dim=0),
        cmap=cmap,
        vmin=-0.5,
        vmax=num_classes - 0.5,
        interpolation="nearest",
    )
    axes[crop_idx, 1].set_title(f"Crop {crop_idx} logits")
    axes[crop_idx, 1].axis("off")

plt.tight_layout()
plt.show()
# %%
# plot color mapping
plt.figure(figsize=(6, 1))
plt.imshow(
    np.arange(num_classes).reshape(1, -1),
    cmap=cmap,
    vmin=-0.5,
    vmax=num_classes - 0.5,
    aspect="auto",
)
# %%
