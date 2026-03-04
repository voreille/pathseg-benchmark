import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder


class ConvUpsampleHead(nn.Module):
    """
    Learned upsampling head that operates on ViT token grid features.

    Input:  feats (B, N, D)
    Output: logits (B, K, out_h, out_w)
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        grid_size: tuple[int, int],  # (Hgrid, Wgrid)
        out_size: tuple[int, int],  # (Hout, Wout) e.g. (img_size, img_size)
        hidden_dim: int | None = None,
        num_refine_blocks: int = 2,
        norm: str = "gn",  # "gn" or "bn" or "ln2d"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.out_size = out_size

        if hidden_dim is None:
            hidden_dim = embed_dim // 2

        def make_norm(c: int):
            if norm == "gn":
                # 32 groups is common; ensure divisible
                g = 32
                while c % g != 0 and g > 1:
                    g //= 2
                return nn.GroupNorm(g, c)
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "ln2d":
                # LayerNorm over channel dim per spatial location
                return nn.GroupNorm(1, c)
            raise ValueError(f"Unknown norm: {norm}")

        # Project D -> hidden_dim on the feature map
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            make_norm(hidden_dim),
            nn.GELU(),
        )

        # A few local refinement convs at grid resolution
        blocks = []
        for _ in range(num_refine_blocks):
            blocks += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                make_norm(hidden_dim),
                nn.GELU(),
            ]
        self.refine = nn.Sequential(*blocks)

        # Final classifier at (upsampled) resolution
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)
        B, N, D = tokens.shape
        Hg, Wg = self.grid_size
        assert N == Hg * Wg, f"Expected N==Hg*Wg ({N} vs {Hg * Wg})"

        # (B, D, Hg, Wg)
        x = tokens.transpose(1, 2).reshape(B, D, Hg, Wg)

        x = self.proj(x)
        x = self.refine(x)

        # Upsample features, then classify
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        logits = self.classifier(x)
        return logits


class ConvDecoder(Encoder):
    def __init__(self, encoder_id, num_classes, img_size, **kwargs):
        super().__init__(encoder_id=encoder_id, img_size=img_size, **kwargs)

        # grid_size comes from your Encoder (patch grid)
        self.decode_head = ConvUpsampleHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            grid_size=tuple(self.grid_size),  # (Hg, Wg)
            out_size=(img_size, img_size),  # or actual crop H,W
            hidden_dim=self.embed_dim // 2,
            num_refine_blocks=2,
            norm="gn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = super().forward(x)  # (B, N, D)
        logits = self.decode_head(tokens)  # (B, K, img_size, img_size)
        return logits
