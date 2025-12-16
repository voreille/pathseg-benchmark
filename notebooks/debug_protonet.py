import sys

sys.path.append('..')

import torch

from datasets.anorak_fewshot import ANORAK_FS
from models.protonet_layer import ProtoNet
from models.histo_encoder import Uni2Encoder

device = torch.device("cuda:1")

data_module = ANORAK_FS(
    root="/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK",
    devices=1,
    num_workers=0,
    batch_size=1,
    img_size=(448, 448),
    num_classes=7,
    num_metrics=1,
    n_training_samples=2,
)

# Setup the data module
data_module.setup()

train_loader = data_module.train_dataloader()

IMG_SIZE = 448
PATCH_SIZE = 14
encoder = Uni2Encoder(img_size=(IMG_SIZE, IMG_SIZE),
                      patch_size=PATCH_SIZE).to(device)
protonet = ProtoNet(img_size=IMG_SIZE, patch_size=PATCH_SIZE).to(device)

protonet.fit(train_loader, encoder)
