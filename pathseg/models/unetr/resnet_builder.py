import torch
import torch.nn as nn
from torchvision import models


# --- Exact trunk: 4 × (Linear -> BN -> ReLU) -> 1024 ---
def make_four_layer_trunk(in_dim: int, hidden: int = 1024) -> nn.Sequential:
    layers = []
    dim_prev = in_dim
    for _ in range(4):
        layers.append(nn.Linear(dim_prev, hidden,
                                bias=False))  # idx: 0,3,6,9 ...
        layers.append(nn.BatchNorm1d(hidden))  # idx: 1,4,7,10 ...
        layers.append(nn.ReLU(inplace=True))  # idx: 2,5,8,11 ...
        dim_prev = hidden
    return nn.Sequential(*layers)


class ResNetWithTrunk1024(nn.Module):
    """
    Backbone (tv ResNet, fc=Identity) + the 4-layer trunk to 1024-d embedding.
    forward(x) -> h in R^{1024}
    """

    def __init__(self, base_encoder: str = "resnet101"):
        super().__init__()
        if base_encoder not in {
                "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        }:
            raise ValueError(f"Unsupported base_encoder: {base_encoder}")
        backbone = getattr(models, base_encoder)(weights=None)
        in_dim = backbone.fc.in_features  # 2048 for RN50/101/152, 512 for RN18/34
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.trunk = make_four_layer_trunk(in_dim=in_dim, hidden=1024)

    def forward(self, x):
        feats = self.backbone(x)  # [N, in_dim]
        h = self.trunk(feats)  # [N, 1024]
        return h


def _strip_prefix(k: str, prefix: str):
    return k[len(prefix):] if k.startswith(prefix) else None


def load_moco_resnet_embedding(checkpoint_path: str,
                               device: str = "cpu",
                               base_encoder: str = "resnet101",
                               verbose: bool = True) -> nn.Module:
    """
    Load a model that outputs the 1024-d embedding (h) from a MoCo v2 checkpoint
    trained with FourLayerHead (trunk/proj). Only loads encoder_q.{backbone, head.trunk}.
    Ignores encoder_k.* and encoder_q.head.proj.*.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # support both layouts

    model = ResNetWithTrunk1024(base_encoder=base_encoder).to(device)

    # --- Collect and remap backbone params: encoder_q.backbone.* -> backbone.*
    back_sd = {}
    for k, v in state.items():
        k0 = k[7:] if k.startswith("module.") else k
        p = _strip_prefix(k0, "encoder_q.backbone.")
        if p is not None and not p.startswith("fc."):
            back_sd["backbone." + p] = v
    miss_b, unexp_b = model.load_state_dict(back_sd, strict=False)
    if verbose and (miss_b or unexp_b):
        print("[Backbone load] missing:", miss_b)
        print("[Backbone load] unexpected:", unexp_b)

    # --- Load trunk exactly: encoder_q.head.trunk.N.* -> trunk.N.*
    # We build the exact structure, so names should align 1:1.
    trunk_assign = {}
    for k, v in state.items():
        k0 = k[7:] if k.startswith("module.") else k
        if k0.startswith("encoder_q.head.trunk."):
            # local = k0.replace("encoder_q.head.trunk.", "trunk.")
            local = k0.replace("encoder_q.head.trunk.", "")
            trunk_assign[local] = v

    miss_h, unexp_h = model.trunk.load_state_dict(trunk_assign, strict=False)
    if verbose and (miss_h or unexp_h):
        print("[Trunk load] missing:", miss_h)
        print("[Trunk load] unexpected:", unexp_h)

    # Note: we intentionally ignore encoder_q.head.proj.* (projection to output_dim)
    model.eval()
    return model
