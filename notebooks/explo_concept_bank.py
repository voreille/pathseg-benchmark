# %%
import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pathseg.utils.load_experiment import load_experiment

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
config_path = (
    "/home/valentin/workspaces/pathseg-benchmark/configs/anorak_concept_semantic.yaml"
)
checkpoint_path = "/home/valentin/workspaces/pathseg-benchmark/runs/checkpoints/66n6pk0s/epoch=35-step=40000.ckpt"

bundle = load_experiment(
    config_path=config_path,
    ckpt_path=checkpoint_path,
    stage="predict",
    overrides={
        "data.init_args.num_workers": 0,
        "data.init_args.batch_size": 16,
        "trainer.logger": False,
    },
    device=device,
)

dm = bundle["datamodule"]
model = bundle["model"]
model.eval()

# %%
data_loader = dm.predict_center_crop_dataloader()

for batch_idx, batch in enumerate(data_loader):
    imgs, targets_raw, img_ids, metas = batch
    imgs = imgs.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(imgs)
    if batch_idx == 1:
        break

device = next(model.parameters()).device


# %%
def to_per_pixel_targets_semantic(targets, ignore_idx):
    per_pixel_targets = []
    for target in targets:
        H, W = target["masks"].shape[-2:]
        per_pixel = torch.full(
            (H, W),
            ignore_idx,
            dtype=torch.long,
            device=device,
        )
        for i, mask in enumerate(target["masks"]):
            per_pixel[mask] = target["labels"][i]
        per_pixel_targets.append(per_pixel)
    return torch.stack(per_pixel_targets)


targets = to_per_pixel_targets_semantic(targets_raw, dm.ignore_idx)

b = 15
img = imgs[b]
target = targets[b]

semantic_logits = outputs["semantic_logits"][b].float()
primitive_logits = outputs["primitive_logits"][b].float()
primitive_probs = outputs["primitive_probs"][b].float()

# -------------------------------------------------------
# Upsample to image resolution
# -------------------------------------------------------
H, W = target.shape

if semantic_logits.shape[-2:] != (H, W):
    semantic_logits = F.interpolate(
        semantic_logits.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0]

if primitive_logits.shape[-2:] != (H, W):
    primitive_logits = F.interpolate(
        primitive_logits.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0]

if primitive_probs.shape[-2:] != (H, W):
    primitive_probs = F.interpolate(
        primitive_probs.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0]

# %%
# Determine GT and Pred dominant class
# -------------------------------------------------------
valid = target != dm.ignore_idx
preds = semantic_logits.argmax(dim=0)


# %%
num_classes = semantic_logits.shape[0]

base_colors = [
    "#000000",
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

cmap = ListedColormap(base_colors[: num_classes + 1])


def to_numpy_img(x):
    x = x.detach().cpu().float()
    x = x.permute(1, 2, 0)
    x = (x - x.min()) / (x.max() + 1e-8)
    return x.numpy()


img_np = to_numpy_img(img)

# %%
primitive_probs.shape
# %%
rows = 1
cols = 5
fig = plt.figure(figsize=(22, 14))

# Row 1
plt.subplot(rows, cols, 1)
plt.imshow(img_np)
plt.title("Input")
plt.axis("off")

plt.subplot(rows, cols, 2)
plt.imshow(target.cpu(), cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title("GT")
plt.axis("off")

plt.subplot(rows, cols, 3)
plt.imshow(preds.cpu(), cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title("Prediction")
plt.axis("off")

plt.subplot(rows, cols, 4)
plt.imshow(primitive_probs[2, :, :].cpu(), cmap="viridis", vmin=0, vmax=1.0)
plt.title("Primitive Probabilities")
plt.axis("off")

plt.subplot(rows, cols, 5)
plt.imshow(primitive_probs[2, :, :].cpu(), cmap="viridis", vmin=0, vmax=1.0)
plt.title("Primitive Probabilities")
plt.axis("off")



# %%
legend_elements = [
    Patch(facecolor=base_colors[i], label=f"Class {i}") for i in range(num_classes + 1)
]

plt.figure(figsize=(6, 2))
plt.legend(handles=legend_elements, loc="center", ncol=min(num_classes + 1, 10))
plt.axis("off")
plt.show()
# %%
class_id = 2
# %%
linear_head = model.network.head
# %%

# %%
plt.plot(linear_head.weight[1, :].cpu().detach().numpy())

# %%
