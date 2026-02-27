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
# -------------------------------------------------------
# Convert targets to per-pixel
# -------------------------------------------------------
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


# %%
def dominant_non_bg_class(mask, ignore_idx, bg_idx=0):
    """
    Returns dominant class in mask excluding ignore_idx.
    If dominant is background, returns next most frequent non-background class.
    If only background present, returns background.
    """
    valid = mask != ignore_idx
    vals = mask[valid]

    if vals.numel() == 0:
        return bg_idx

    counts = torch.bincount(vals)

    # sort classes by frequency descending
    sorted_classes = torch.argsort(counts, descending=True)

    for cls in sorted_classes:
        cls = int(cls)
        if cls != bg_idx and counts[cls] > 0:
            return cls

    return bg_idx


# %%
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
mask_probs = torch.sigmoid(concept_mask_logits)
class_probs = torch.softmax(concept_class_logits, dim=-1)
semantic_probs = torch.softmax(semantic_logits, dim=0)

pred = semantic_logits.argmax(0)

# -------------------------------------------------------
# Determine GT and Pred dominant class
# -------------------------------------------------------
valid = target != dm.ignore_idx
gt_class = dominant_non_bg_class(target, dm.ignore_idx, bg_idx=0)
pred_class = dominant_non_bg_class(pred, dm.ignore_idx, bg_idx=0)

# -------------------------------------------------------
# Rank concepts for GT class
# -------------------------------------------------------
valid_f = valid.float()
denom = valid_f.sum() + 1e-6
area = (mask_probs * valid_f).sum((1, 2)) / denom
score = area * class_probs[:, gt_class]

topk = 12
top_idx = torch.topk(score, k=min(topk, mask_probs.shape[0])).indices

# -------------------------------------------------------
# Colormap (discrete)
# -------------------------------------------------------
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


# -------------------------------------------------------
# Helper: normalize image
# -------------------------------------------------------
def to_numpy_img(x):
    x = x.detach().cpu().float()
    x = x.permute(1, 2, 0)
    x = (x - x.min()) / (x.max() + 1e-8)
    return x.numpy()


img_np = to_numpy_img(img)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
rows = 4
cols = 6
fig = plt.figure(figsize=(22, 14))

# Row 1
plt.subplot(rows, cols, 1)
plt.imshow(img_np)
plt.title("Input")
plt.axis("off")

plt.subplot(rows, cols, 2)
plt.imshow(target.cpu(), cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title(f"GT (class {gt_class})")
plt.axis("off")

plt.subplot(rows, cols, 3)
plt.imshow(pred.cpu(), cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.title(f"Prediction (class {pred_class})")
plt.axis("off")

error_map = (pred != target) & valid
plt.subplot(rows, cols, 4)
plt.imshow(error_map.cpu(), cmap="Reds")
plt.title("Error map")
plt.axis("off")

plt.subplot(rows, cols, 5)
plt.imshow(semantic_probs[gt_class].cpu(), cmap="viridis")
plt.title(f"P(GT class {gt_class})")
plt.axis("off")

plt.subplot(rows, cols, 6)
plt.imshow(semantic_probs[pred_class].cpu(), cmap="viridis")
plt.title(f"P(Pred class {pred_class})")
plt.axis("off")

# Row 2: concept masks
for i, k in enumerate(top_idx[:cols]):
    plt.subplot(rows, cols, cols + i + 1)
    plt.imshow(mask_probs[k].cpu(), cmap="magma")
    plt.title(f"k={k.item()}")
    plt.axis("off")

# Row 3: concept contributions
for i, k in enumerate(top_idx[:cols]):
    contrib = mask_probs[k] * class_probs[k, gt_class]
    plt.subplot(rows, cols, 2 * cols + i + 1)
    plt.imshow(contrib.cpu(), cmap="viridis")
    plt.title(f"contrib k={k.item()}")
    plt.axis("off")

# Row 4: class preference per concept
for i, k in enumerate(top_idx[:cols]):
    plt.subplot(rows, cols, 3 * cols + i + 1)
    probs = class_probs[k].detach().cpu()
    plt.bar(range(num_classes), probs)
    plt.title(f"class dist k={k.item()}")
    plt.ylim(0, 1)
    plt.xticks(range(num_classes))

plt.tight_layout()
plt.show()

# Legend
legend_elements = [
    Patch(facecolor=base_colors[i], label=f"Class {i}") for i in range(num_classes + 1)
]

plt.figure(figsize=(6, 2))
plt.legend(handles=legend_elements, loc="center", ncol=min(num_classes+1, 10))
plt.axis("off")
plt.show()
# %%
num_queries = 64
num_classes = 7

activation_matrix = torch.zeros(num_queries, num_classes, device=device)
contribution_matrix = torch.zeros(num_queries, num_classes, device=device)
pixel_count = torch.zeros(num_classes, device=device)
# %%

for batch in data_loader:
    imgs, targets_raw, *_ = batch
    imgs = imgs.to(device)

    with torch.inference_mode():
        out = model(imgs)

    mask_logits = out["concept_mask_logits"].to(device).float()  # (B,K,h,w)
    class_logits = out["concept_class_logits"].to(device).float()  # (B,K,C)

    class_probs = torch.softmax(class_logits, dim=-1)  # (B,K,C)

    # targets (B,H,W)
    targets = to_per_pixel_targets_semantic(targets_raw, dm.ignore_idx).to(device)

    # choose the interpolation size from targets (safest)
    H, W = targets.shape[-2:]
    mask_logits = F.interpolate(
        mask_logits,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    mask_probs = torch.sigmoid(mask_logits)  # (B,K,H,W)

    B, K, H, W = mask_probs.shape
    C = class_probs.shape[-1]

    mask_probs_flat = mask_probs.reshape(B, K, H * W)  # (B,K,HW)
    targets_flat = targets.reshape(B, H * W)  # (B,HW)

    sel_f = None  # to keep memory stable; optional

    for c in range(num_classes):
        sel = targets_flat == c  # (B,HW) bool
        n_pix = sel.sum()
        if n_pix == 0:
            continue

        sel_f = sel[:, None, :].float()  # (B,1,HW)

        # mask-only alignment: sum over pixels of class c
        act_k = (mask_probs_flat * sel_f).sum(dim=(0, 2))  # (K,)

        # true contribution: mask * P(c|k)
        contrib_k = (
            mask_probs_flat
            * class_probs[:, :, c][:, :, None]  # (B,K,1)
            * sel_f  # (B,1,HW)
        ).sum(dim=(0, 2))  # (K,)

        activation_matrix[:, c] += act_k
        contribution_matrix[:, c] += contrib_k
        pixel_count[c] += n_pix

# normalize by total pixels per class
activation_matrix = activation_matrix / (pixel_count[None] + 1e-6)
contribution_matrix = contribution_matrix / (pixel_count[None] + 1e-6)
# %%
plt.figure(figsize=(10, 6))
plt.imshow(contribution_matrix.cpu(), aspect="auto", cmap="viridis")
plt.colorbar()
plt.xlabel("Class")
plt.ylabel("Query")
plt.title("Query contribution per class")
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.imshow(activation_matrix.cpu(), aspect="auto", cmap="viridis")
plt.colorbar()
plt.xlabel("Class")
plt.ylabel("Query")
plt.title("Query activation per class")
plt.show()
# %%
row_norm = activation_matrix / (activation_matrix.sum(dim=1, keepdim=True) + 1e-6)

plt.imshow(row_norm.cpu(), aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Row-normalized query-class distribution")
plt.show()

# %%
top_class = row_norm.argmax(dim=1)
unique, counts = torch.unique(top_class, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))

# %%
row_norm = contribution_matrix / (contribution_matrix.sum(dim=1, keepdim=True) + 1e-6)
top_query_per_class = row_norm.argmax(dim=0)
for c in range(num_classes):
    q = top_query_per_class[c].item()
    print(f"Class {c} top query: {q}, activation: {row_norm[q, c]:.4f}")
# %%
activation_sum = activation_matrix.sum(dim=1)

plt.hist(activation_sum.cpu().numpy(), bins=30)
plt.title("Total activation per query")
plt.show()

# %%
row_norm = activation_matrix / (activation_matrix.sum(dim=1, keepdim=True) + 1e-6)
entropy = -(row_norm * torch.log(row_norm + 1e-8)).sum(dim=1)

# %%
for i, batch in enumerate(data_loader):
    imgs, targets_raw, img_ids, metas = batch
    imgs = imgs.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(imgs)
    if i == 1:
        break


# %%
query_idx = 8
top_query_mask = outputs["concept_mask_logits"][:, query_idx, ...].sigmoid()
# %%
top_query_mask = F.interpolate(
    top_query_mask.unsqueeze(1),
    size=targets.shape[-2:],
    mode="bilinear",
    align_corners=False,
)[:, 0, ...]
# %%
# %%
# %%

# --- if you don't already have targets in per-pixel form ---
targets = to_per_pixel_targets_semantic(targets_raw, dm.ignore_idx).to(
    device
)  # (B,H,W)

# --- pick query and (optionally) a class to visualize contribution ---
query_idx = 51
class_idx = None  # e.g. 6; set to an int to show contribution map too

# --- pull tensors ---
sem_logits = outputs["semantic_logits"].float()  # (B,C,h,w)
mask_logits = outputs["concept_mask_logits"].float()  # (B,K,h,w)
class_logits = outputs["concept_class_logits"].float()  # (B,K,C)

B = imgs.shape[0]
H, W = targets.shape[-2:]

# --- upsample semantic + query mask to target resolution ---
sem_logits = F.interpolate(
    sem_logits, size=(H, W), mode="bilinear", align_corners=False
)  # (B,C,H,W)
pred = sem_logits.argmax(dim=1)  # (B,H,W)

q_mask = torch.sigmoid(mask_logits[:, query_idx, ...])  # (B,h,w)
q_mask = F.interpolate(
    q_mask.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
)[:, 0]  # (B,H,W)

# --- optional contribution map: q_mask * P(class_idx | query_idx) ---
contrib = None
if class_idx is not None:
    class_probs = torch.softmax(class_logits, dim=-1)  # (B,K,C)
    w = class_probs[:, query_idx, class_idx]  # (B,)
    contrib = q_mask * w[:, None, None]  # (B,H,W)

# --- discrete cmap for GT/pred ---
num_classes = sem_logits.shape[1]

legend_elements = [
    Patch(facecolor=base_colors[i], label=f"Class {i}") for i in range(num_classes+1)
]


# --- plot batch grid ---
cols = 4 + (1 if class_idx is not None else 0)  # input, GT, pred, qmask, (+ contrib)
rows = B
fig = plt.figure(figsize=(4.2 * cols, 3.2 * rows))

for i in range(B):
    img_np = to_numpy_img(imgs[i])
    gt_i = targets[i].detach().cpu()
    pred_i = pred[i].detach().cpu()
    q_i = q_mask[i].detach().cpu()

    # Input
    ax = plt.subplot(rows, cols, i * cols + 1)
    ax.imshow(img_np)
    ax.set_title(f"Input\n{img_ids[i] if isinstance(img_ids, (list, tuple)) else ''}")
    ax.axis("off")

    # GT
    ax = plt.subplot(rows, cols, i * cols + 2)
    ax.imshow(gt_i, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax.set_title("GT")
    ax.axis("off")

    # Pred
    ax = plt.subplot(rows, cols, i * cols + 3)
    ax.imshow(pred_i, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax.set_title("Pred")
    ax.axis("off")

    # Query mask
    ax = plt.subplot(rows, cols, i * cols + 4)
    ax.imshow(q_i, cmap="magma")
    ax.set_title(f"Query {query_idx} mask")
    ax.axis("off")

    # Contribution
    if class_idx is not None:
        ax = plt.subplot(rows, cols, i * cols + 5)
        ax.imshow(contrib[i].detach().cpu(), cmap="viridis")
        ax.set_title(f"Query {query_idx} contrib to class {class_idx}")
        ax.axis("off")

plt.tight_layout()
plt.show()

# --- legend (separate) ---
plt.figure(figsize=(10, 2))
plt.legend(handles=legend_elements, loc="center", ncol=min(num_classes+1, 10))
plt.axis("off")
plt.show()

# %%
