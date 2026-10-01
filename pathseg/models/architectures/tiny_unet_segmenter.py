from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from pathseg.models.decoders import MultiTaskDecoder
from pathseg.models.semantic_segmenter import SemanticSegmenter

FeatureMaps = tuple[torch.Tensor, ...]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class TinyUNetEncoder(nn.Module):
    """Tiny U-Net backbone producing multiscale decoder feature maps."""

    output_stride = 2

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.embed_dim = base_channels

        # Channels corresponding to the returned feature-map tuple.
        self.feature_channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )

        # 448 -> 224
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

        # 224 -> 112
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(
                base_channels,
                base_channels * 2,
            ),
        )

        # 112 -> 56
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(
                base_channels * 2,
                base_channels * 4,
            ),
        )

        # 56 -> 112
        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        # 112 -> 224
        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

    def forward_feature_maps(self, imgs: torch.Tensor) -> FeatureMaps:
        level_224 = self.stem(imgs)
        level_112 = self.down1(level_224)
        level_56 = self.down2(level_112)

        decoded_112 = self.up1(level_56, level_112)
        decoded_224 = self.up2(decoded_112, level_224)

        return (
            decoded_224,
            decoded_112,
            level_56,
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.forward_feature_maps(imgs)[0]


class DeepSupervisionLinearDecoder(nn.Module):
    """Project and fuse logits from several feature-map resolutions."""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        *,
        deep_supervision: bool = True,
        fusion: Literal["sum", "mean"] = "mean",
    ) -> None:
        super().__init__()

        if not in_channels:
            raise ValueError("At least one feature level is required.")

        if fusion not in {"sum", "mean"}:
            raise ValueError(f"Unsupported fusion mode: {fusion!r}")

        self.num_classes = out_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        self.fusion = fusion

        self.projections = nn.ModuleList(
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=1,
            )
            for channels in in_channels
        )

    def forward(self, feature_maps: FeatureMaps) -> torch.Tensor:
        if len(feature_maps) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature maps, "
                f"got {len(feature_maps)}."
            )

        # Highest-resolution U-Net output: 224 × 224.
        output_size = feature_maps[0].shape[-2:]

        final_logits = self.projections[0](feature_maps[0])

        if not self.deep_supervision:
            return final_logits

        logits_by_level = [final_logits]

        for projection, feature_map in zip(
            self.projections[1:],
            feature_maps[1:],
            strict=True,
        ):
            logits = projection(feature_map)
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            logits_by_level.append(logits)

        fused_logits = torch.stack(logits_by_level, dim=0).sum(dim=0)

        if self.fusion == "mean":
            fused_logits = fused_logits / len(logits_by_level)

        return fused_logits


class IgniteAnorakTinySegmenter(SemanticSegmenter):
    def __init__(
        self,
        ignite_num_classes: int,
        anorak_num_classes: int,
        *,
        base_channels: int = 16,
        deep_supervision: bool = True,
        supervision_fusion: Literal["sum", "mean"] = "mean",
        upsample_logits: bool = True,
        interpolation_mode: str = "bilinear",
    ) -> None:
        encoder = TinyUNetEncoder(
            in_channels=3,
            base_channels=base_channels,
        )

        decoder = MultiTaskDecoder(
            heads={
                "ignite": DeepSupervisionLinearDecoder(
                    in_channels=encoder.feature_channels,
                    out_channels=ignite_num_classes,
                    deep_supervision=deep_supervision,
                    fusion=supervision_fusion,
                ),
                "anorak": DeepSupervisionLinearDecoder(
                    in_channels=encoder.feature_channels,
                    out_channels=anorak_num_classes,
                    deep_supervision=deep_supervision,
                    fusion=supervision_fusion,
                ),
            }
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            # The fused decoder output is 224×224 and must be brought
            # to the 448×448 input resolution.
            upsample_logits=upsample_logits,
            interpolation_mode=interpolation_mode,
        )
