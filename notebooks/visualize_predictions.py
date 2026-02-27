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

