# %%
import os

import torch
import yaml
from matplotlib import pyplot as plt

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
    stage="validate_random_crop",
    overrides={
        "data.init_args.num_workers": 0,
        "data.init_args.batch_size": 1,
        "trainer.logger": False,
    },
    device=device,
)
dm = bundle["datamodule"]
model = bundle["model"]
trainer = bundle["trainer"]

val_loader = dm.val_dataloader()
# %%
for batch in val_loader:
    imgs, targets = batch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(imgs)
    break
