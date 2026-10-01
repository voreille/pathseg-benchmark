from __future__ import annotations

import torch
from torch import nn

from pathseg.models.decoders import LinearDecoder, MultiTaskDecoder
from pathseg.models.encoder import Encoder
from pathseg.models.semantic_segmenter import SemanticSegmenter


class TinyEncoder(nn.Module):
    """Small convolutional encoder with an output stride of 16."""

    output_stride = 16

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.embed_dim = embedding_dim

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(24, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward_feature_maps(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features = self.layers(imgs)
        return (features,)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convenience forward; SemanticSegmenter uses forward_feature_maps."""
        return self.layers(imgs)


class IgniteAnorakTinySegmenter(SemanticSegmenter):
    """Tiny shared encoder with linear heads for IGNITE and ANORAK."""

    def __init__(
        self,
        ignite_num_classes: int,
        anorak_num_classes: int,
        *,
        embedding_dim: int = 32,
        upsample_logits: bool = True,
        interpolation_mode: str = "bilinear",
    ) -> None:
        encoder = TinyEncoder(
            in_channels=3,
            embedding_dim=embedding_dim,
        )

        decoder = MultiTaskDecoder(
            heads={
                "ignite": LinearDecoder(
                    in_channels=encoder.embed_dim,
                    out_channels=ignite_num_classes,
                ),
                "anorak": LinearDecoder(
                    in_channels=encoder.embed_dim,
                    out_channels=anorak_num_classes,
                ),
            }
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            upsample_logits=upsample_logits,
            interpolation_mode=interpolation_mode,
        )
