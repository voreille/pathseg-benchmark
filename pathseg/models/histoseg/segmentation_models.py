from typing import Dict, Optional, Iterable
import torch
import torch.nn as nn

from .encoder import ViTEncoderPyramidHooks, ResNetPyramidAdapter
from .segfpn import SegFPN


class HistoViTSeg(nn.Module):
    """
    Lightning-friendly wrapper that builds:
      ViT (UNI2) -> ViTEncoderPyramidHooks -> SegFPN -> SegModel

    YAML:
      network:
        class_path: models.histoseg.builder.Uni2Seg
        init_args:
          encoder_name: "hf-hub:MahmoodLab/UNI2-h"
          img_size: [448, 448]
          num_classes: 6
    """

    def __init__(
            self,
            encoder_id: str = "uni2",
            num_classes: int = 7,
            dropout: float = 0.2,
            deep_supervision: bool = True,
            img_skip_ch: int = 64,
            extract_layers: Iterable[int] = (6, 12, 18, 24),
            pyramid_channels: Optional[Dict[
                str, int]] = None,  # {"s4":64,"s8":128,"s16":256,"s32":256}
    ):
        super().__init__()

        self.save_hyperparameters = getattr(
            nn.Module, "save_hyperparameters",
            lambda *a, **k: None)  # no-op if not LightningModule

        from ..histo_encoder import build_encoder as build_vit_encoder
        vit, vit_meta = build_vit_encoder(encoder_id=encoder_id)

        pixel_mean = torch.tensor(vit_meta["pixel_mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(vit_meta["pixel_std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        # 3) Channel schedule (per-scale)
        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        # 4) Encoder adapter (hooks)

        self.encoder_adapter = ViTEncoderPyramidHooks(
            vit=vit,
            pyramid_channels=pyramid_channels,
            embed_dim=vit_meta["embed_dim"],
        )

        self.decoder = SegFPN(
            num_classes=num_classes,
            pyramid_channels=pyramid_channels,
            img_skip_ch=img_skip_ch,
            dropout=dropout,
            deep_supervision=deep_supervision,
            assert_shapes=True,
        )

        # so the lightning module can find it and freeze if needed
        self.encoder = vit

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std

        ps = int(self.encoder.patch_embed.patch_size[0])
        H, W = x.shape[-2:]
        assert H % ps == 0 and W % ps == 0, f"Input size ({H},{W}) not divisible by patch size {ps}"

        features = self.encoder_adapter(x)
        return self.decoder(x, features)


class Resnet50Seg(nn.Module):

    def __init__(
            self,
            num_classes: int = 7,
            dropout: float = 0.2,
            deep_supervision: bool = True,
            img_skip_ch: int = 64,
            pyramid_channels: Optional[Dict[
                str, int]] = None,  # {"s4":64,"s8":128,"s16":256,"s32":256}
    ):
        from torchvision.models import resnet50, ResNet50_Weights

        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        self.encoder_adapter = ResNetPyramidAdapter(
            resnet=resnet,
            pyramid_channels=pyramid_channels,
        )
        self.decoder = SegFPN(
            num_classes=num_classes,
            pyramid_channels=pyramid_channels,
            img_skip_ch=img_skip_ch,
            dropout=dropout,
            deep_supervision=deep_supervision,
            assert_shapes=True,
        )
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.encoder = resnet

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.encoder_adapter(x)
        return self.decoder(x, features)


class MocoResNetSeg(nn.Module):

    def __init__(
            self,
            checkpoint_path: str,
            base_encoder: str = "resnet101",
            num_classes: int = 7,
            dropout: float = 0.2,
            deep_supervision: bool = True,
            img_skip_ch: int = 64,
            pyramid_channels: Optional[Dict[
                str, int]] = None,  # {"s4":64,"s8":128,"s16":256,"s32":256}
    ):

        from .resnet_builder import load_moco_resnet_embedding
        super().__init__()

        resnet_wrap = load_moco_resnet_embedding(
            checkpoint_path=checkpoint_path,
            device="cpu",
            base_encoder=base_encoder,
            verbose=True,
        )
        resnet = resnet_wrap.backbone

        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        self.encoder_adapter = ResNetPyramidAdapter(
            resnet=resnet,
            pyramid_channels=pyramid_channels,
        )
        self.decoder = SegFPN(
            num_classes=num_classes,
            pyramid_channels=pyramid_channels,
            img_skip_ch=img_skip_ch,
            dropout=dropout,
            deep_supervision=deep_supervision,
            assert_shapes=True,
        )
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.encoder = resnet

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.encoder_adapter(x)
        return self.decoder(x, features)
