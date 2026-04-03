from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock2d(nn.Module):
    """
    ConvNeXt-like block for dense feature refinement.

    Works in a reduced hidden space:
    - depthwise conv for spatial smoothing / local mixing
    - channel MLP with 1x1 convs
    - residual connection
    - optional layer scale
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.norm = LayerNorm2d(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        if self.gamma is not None:
            x = x * self.gamma[:, None, None]

        return residual + x


class ProtoProj(nn.Module):
    """
    Efficient prototype projection head for ViT feature maps.

    Idea:
    - first compress the encoder feature map
    - refine spatially in the compressed space
    - project to proto_dim
    - add a skip projection to proto_dim

    This is usually much cheaper than doing 3x3 residual blocks at `in_dim`.
    """

    def __init__(
        self,
        in_dim: int,
        proto_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        kernel_size: int = 7,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            # simple default: reduce aggressively but not too much
            hidden_dim = max(proto_dim, in_dim // 4)

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock2d(
                    dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    kernel_size=kernel_size,
                )
                for _ in range(depth)
            ]
        )

        self.out_proj = nn.Sequential(
            LayerNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, proto_dim, kernel_size=1, bias=True),
        )

        self.skip_proj = nn.Conv2d(in_dim, proto_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)

        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x + skip


class ProtoProjLite(nn.Module):
    def __init__(self, in_dim: int, proto_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(proto_dim, in_dim // 4)

        self.block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                padding=3,
                groups=hidden_dim,
                bias=False,
            ),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, proto_dim, 1, bias=True),
        )
        self.skip = nn.Conv2d(in_dim, proto_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ConvRefineBlockGN(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        num_groups: int = 8,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        if dim % num_groups != 0:
            raise ValueError("dim must be divisible by num_groups")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)

        if self.gamma is not None:
            x = x * self.gamma[:, None, None]

        return residual + x


class ProtoProjGN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proto_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(proto_dim, in_dim // 4)

        if hidden_dim % num_groups != 0:
            raise ValueError("hidden_dim must be divisible by num_groups")

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                ConvRefineBlockGN(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    num_groups=num_groups,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.Conv2d(hidden_dim, proto_dim, kernel_size=1, bias=True),
        )

        self.skip_proj = nn.Conv2d(in_dim, proto_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x + skip
