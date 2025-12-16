# %%
import sys

sys.path.append('..')

from torchinfo import summary

from models.histoseg.segmentation_models import HistoViTSeg
# %%

model = HistoViTSeg(encoder_id="uni2")
model.to("cuda")

# %%
summary(
    model,
    input_size=(1, 3, 448, 448),  # (batch, C, H, W)
    depth=2,
    col_names=("input_size", "output_size", "num_params"),
)

# %%
