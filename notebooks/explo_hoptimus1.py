# %%
from huggingface_hub import login
import torch
import timm
from torchvision import transforms

# %%
# Login to the Hugging Face hub, using your user access token that can be found here:
# https://huggingface.co/settings/tokens.
login()

# %%
model = timm.create_model("hf-hub:bioptimus/H-optimus-1",
                          pretrained=True,
                          init_values=1e-5,
                          dynamic_img_size=False)
model.to("cuda")
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.707223, 0.578729, 0.703617),
                         std=(0.211883, 0.230117, 0.177517)),
])

# %%
input = torch.rand(3, 224, 224)
input = transforms.ToPILImage()(input)

# %%
# We recommend using mixed precision for faster inference.
with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.inference_mode():
        features = model(transform(input).unsqueeze(0).to("cuda"))

# %%
assert features.shape == (1, 1536)

# %%
print(len(model.blocks))


# %%
