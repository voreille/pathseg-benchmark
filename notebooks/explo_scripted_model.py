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
model = torch.jit.load(
    "/home/valentin/workspaces/pathseg-benchmark/models/twotasks/scripted_model.pt"
).to(device)


# %%
def _make_discrete_cmap(num_classes: int) -> ListedColormap:
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.arange(num_classes))
    return ListedColormap(colors)


num_classes = 7
cmap = _make_discrete_cmap(num_classes=num_classes)

# %%
config_path = "/home/valentin/workspaces/pathseg-benchmark/models/twotasks/anorak_ignite_multitask.yaml"

bundle = load_experiment(
    config_path=config_path,
    stage="fit",
    overrides={
        "data.init_args.num_workers": 0,
        "data.init_args.batch_size": 1,
        "trainer.logger": False,
    },
    device=device,
)

dm = bundle["datamodule"]


# %%
data_loader = dm.train_dataloader()
# %%

for batch_idx, batch in enumerate(data_loader):
    # imgs, targets_raw, img_ids = batch
    imgs, targets, source_ids, image_ids = batch
    imgs = (imgs / 255.0).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(imgs)
    if isinstance(logits, dict):
        logits = logits["logits_b"]
    logits = F.interpolate(logits, imgs.shape[2:], mode="bilinear")

    logits = logits.cpu()
    imgs = imgs.cpu()
    break


# %%

logits.shape
# %%


plt.imshow(
    logits[0, ...].argmax(dim=0),
    cmap=cmap,
    vmin=-0.5,
    vmax=num_classes - 0.5,
    interpolation="nearest",
)

plt.tight_layout()
plt.show()

# %%
plt.imshow(
    imgs[0, ...].permute(1, 2, 0),
)
# %%
