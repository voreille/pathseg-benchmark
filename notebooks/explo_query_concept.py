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
checkpoint_path = "/home/valentin/workspaces/pathseg-benchmark/runs/checkpoints/xdue93zu/epoch=35-step=40000.ckpt"

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
concept_mask_logits = outputs["concept_mask_logits"][b].float()
concept_class_logits = outputs["concept_class_logits"][b].float()

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

if concept_mask_logits.shape[-2:] != (H, W):
    concept_mask_logits = F.interpolate(
        concept_mask_logits.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0]

# -------------------------------------------------------
# Probabilities
# -------------------------------------------------------
def safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp(min=eps, max=1.0 - eps))


mask_probs = torch.sigmoid(concept_mask_logits)
class_probs = torch.softmax(concept_class_logits, dim=-1)
semantic_probs = torch.softmax(semantic_logits, dim=0)


# mask_probs:  (B,K,H,W)
# class_probs: (B,K,C)

semantic_probs_direct = torch.einsum("kc,khw->chw", class_probs, mask_probs)  # (C,H,W)

semantic_logits_recomputed = safe_log(semantic_probs_direct)                     # (C,H,W)
semantic_probs_recomputed = semantic_logits_recomputed.softmax(dim=0)            # (C,H,W)

pred = semantic_logits.argmax(0)

# -------------------------------------------------------
# Determine GT and Pred dominant class
# -------------------------------------------------------
valid = target != dm.ignore_idx


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
semantic_probs.shape
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
plt.imshow(pred.cpu(), cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title("Prediction")
plt.axis("off")

plt.subplot(rows, cols, 4)
plt.imshow(semantic_probs_direct[2, :, :].cpu(), cmap="viridis", vmin=0, vmax=1.0)
plt.title("Prediction")
plt.axis("off")

plt.subplot(rows, cols, 5)
plt.imshow(semantic_probs_recomputed[2, :, :].cpu(), cmap="viridis", vmin=0, vmax=1.0)
plt.title("Prediction")
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
semantic_logits.shape
mask = pred == class_id
plt.plot(mask_probs[:, mask].mean(dim=-1).detach().cpu().numpy())


# %%
plt.plot(class_probs[:, class_id].detach().cpu().numpy())
# %%
mask_probs.shape

# %%
plt.imshow(mask_probs[28, :, :].cpu(), cmap="viridis", vmin=0, vmax=1.0)

# %%
value, index = torch.topk(class_probs[:, class_id], 3)
plt.imshow(
    (mask_probs[index, None, :, :] * class_probs[index, :, None, None])
    .sum(dim=0)
    .softmax(dim=0)
    .argmax(dim=0)
    .cpu(),
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)


# %%
mask_probs[index, :, :].shape


# %%
