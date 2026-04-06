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


class ProtoProjMultiScale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.dw3 = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False)
        self.dw7 = nn.Conv2d(out_dim, out_dim, 7, padding=3, groups=out_dim, bias=False)
        self.dw11 = nn.Conv2d(
            out_dim, out_dim, 11, padding=5, groups=out_dim, bias=False
        )

        self.fuse = nn.Conv2d(out_dim * 3, out_dim, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.reduce(x)
        x3 = self.act(self.dw3(x))
        x7 = self.act(self.dw7(x))
        x11 = self.act(self.dw11(x))
        x = torch.cat([x3, x7, x11], dim=1)
        x = self.fuse(x)
        return x


class ProtoProjDilated(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.b1 = nn.Conv2d(
            out_dim, out_dim, 3, padding=1, dilation=1, groups=out_dim, bias=False
        )
        self.b2 = nn.Conv2d(
            out_dim, out_dim, 3, padding=2, dilation=2, groups=out_dim, bias=False
        )
        self.b4 = nn.Conv2d(
            out_dim, out_dim, 3, padding=4, dilation=4, groups=out_dim, bias=False
        )
        self.b8 = nn.Conv2d(
            out_dim, out_dim, 3, padding=8, dilation=8, groups=out_dim, bias=False
        )

        self.fuse = nn.Conv2d(out_dim * 4, out_dim, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.reduce(x)
        y1 = self.act(self.b1(x))
        y2 = self.act(self.b2(x))
        y4 = self.act(self.b4(x))
        y8 = self.act(self.b8(x))
        y = torch.cat([y1, y2, y4, y8], dim=1)
        y = self.fuse(y)
        return y


class ProtoProjMultiScaleRefined(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(out_dim * expansion)

        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

        self.dw3 = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False
        )
        self.dw7 = nn.Conv2d(
            out_dim, out_dim, kernel_size=7, padding=3, groups=out_dim, bias=False
        )
        self.dw11 = nn.Conv2d(
            out_dim, out_dim, kernel_size=11, padding=5, groups=out_dim, bias=False
        )
        self.dw15 = nn.Conv2d(
            out_dim, out_dim, kernel_size=15, padding=7, groups=out_dim, bias=False
        )

        self.fuse = nn.Conv2d(out_dim * 4, out_dim, kernel_size=1, bias=False)

        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=False),
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)

        x3 = self.act(self.dw3(x))
        x7 = self.act(self.dw7(x))
        x11 = self.act(self.dw11(x))
        x15 = self.act(self.dw15(x))

        y = torch.cat([x3, x7, x11, x15], dim=1)
        y = self.fuse(y)
        y = y + self.refine(y)
        return y


class ProtoProjDilatedScaleAttn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.b1 = nn.Conv2d(
            out_dim, out_dim, 3, padding=1, dilation=1, groups=out_dim, bias=False
        )
        self.b2 = nn.Conv2d(
            out_dim, out_dim, 3, padding=2, dilation=2, groups=out_dim, bias=False
        )
        self.b4 = nn.Conv2d(
            out_dim, out_dim, 3, padding=4, dilation=4, groups=out_dim, bias=False
        )
        self.b8 = nn.Conv2d(
            out_dim, out_dim, 3, padding=8, dilation=8, groups=out_dim, bias=False
        )

        self.act = nn.GELU()
        self.scale_logits = nn.Parameter(torch.zeros(4))
        self.fuse = nn.Conv2d(out_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)

        y1 = self.act(self.b1(x))
        y2 = self.act(self.b2(x))
        y4 = self.act(self.b4(x))
        y8 = self.act(self.b8(x))

        w = torch.softmax(self.scale_logits, dim=0)
        y = w[0] * y1 + w[1] * y2 + w[2] * y4 + w[3] * y8
        y = self.fuse(y)
        return y


class ProtoProjDilatedSpatialScaleAttn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.b1 = nn.Conv2d(
            out_dim, out_dim, 3, padding=1, dilation=1, groups=out_dim, bias=False
        )
        self.b2 = nn.Conv2d(
            out_dim, out_dim, 3, padding=2, dilation=2, groups=out_dim, bias=False
        )
        self.b4 = nn.Conv2d(
            out_dim, out_dim, 3, padding=4, dilation=4, groups=out_dim, bias=False
        )
        self.b8 = nn.Conv2d(
            out_dim, out_dim, 3, padding=8, dilation=8, groups=out_dim, bias=False
        )

        self.act = nn.GELU()

        self.scale_attn = nn.Conv2d(out_dim * 4, 4, kernel_size=1, bias=True)
        self.fuse = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)

        y1 = self.act(self.b1(x))
        y2 = self.act(self.b2(x))
        y4 = self.act(self.b4(x))
        y8 = self.act(self.b8(x))

        y_cat = torch.cat([y1, y2, y4, y8], dim=1)  # [B, 4C, H, W]
        attn = torch.softmax(self.scale_attn(y_cat), dim=1)  # [B, 4, H, W]

        y = (
            attn[:, 0:1] * y1
            + attn[:, 1:2] * y2
            + attn[:, 2:3] * y4
            + attn[:, 3:4] * y8
        )
        y = self.fuse(y)
        return y


class ProtoProjPSP(nn.Module):
    """
    PSP-like context head for prototype matching.

    Idea
    ----
    Build a descriptor that mixes:
    - local feature map
    - coarse pooled context at several spatial scales

    This is useful when prototype matching should depend on
    broader morphology / architecture, not only very local texture.

    Input:
        x: [B, C_in, H, W]

    Output:
        y: [B, C_out, H, W]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pool_bins: tuple[int, ...] = (1, 2, 4, 8),
        branch_dim: int | None = None,
        use_residual_refine: bool = True,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pool_bins = tuple(int(x) for x in pool_bins)

        if len(self.pool_bins) == 0:
            raise ValueError("pool_bins must contain at least one value")

        if branch_dim is None:
            branch_dim = out_dim // 2
        self.branch_dim = int(branch_dim)

        if self.branch_dim < 1:
            raise ValueError("branch_dim must be >= 1")

        # local branch
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

        # pooled context branches
        self.pool_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((bin_size, bin_size)),
                    nn.Conv2d(in_dim, self.branch_dim, kernel_size=1, bias=False),
                    nn.GELU(),
                )
                for bin_size in self.pool_bins
            ]
        )

        fuse_in_dim = out_dim + len(self.pool_bins) * self.branch_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in_dim, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.use_residual_refine = bool(use_residual_refine)
        if self.use_residual_refine:
            hidden_dim = max(out_dim, out_dim * 2)
            self.refine = nn.Sequential(
                nn.Conv2d(out_dim, hidden_dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape [B,C,H,W], got {tuple(x.shape)}"
            )

        h, w = x.shape[-2:]

        local = self.reduce(x)
        branches = [local]

        for pool_proj in self.pool_projs:
            pooled = pool_proj(x)  # [B, branch_dim, b, b]
            pooled = F.interpolate(
                pooled,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            branches.append(pooled)

        y = torch.cat(branches, dim=1)
        y = self.fuse(y)

        if self.use_residual_refine:
            y = y + self.refine(y)

        return y


class ProtoProjPSPAttention(nn.Module):
    """
    PSP-like prototype projection with spatial scale attention.

    Input:
        x: [B, C_in, H, W]

    Output:
        y: [B, C_out, H, W]

    Structure
    ---------
    1. Local branch from full-resolution feature map
    2. Several pooled context branches (PSP-like)
    3. Upsample pooled branches back to HxW
    4. Predict per-pixel attention over branches
    5. Weighted sum of branches
    6. Fuse + optional residual refine

    Notes
    -----
    - All branches are projected to the same channel dimension (out_dim)
    - Attention is spatial: each pixel chooses how much to use each scale
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pool_bins: tuple[int, ...] = (1, 2, 4, 8),
        use_residual_refine: bool = True,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pool_bins = tuple(int(x) for x in pool_bins)

        if len(self.pool_bins) == 0:
            raise ValueError("pool_bins must contain at least one value")

        # Local full-resolution branch
        self.local_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # Pooled PSP branches
        self.pool_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((bin_size, bin_size)),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.GELU(),
                )
                for bin_size in self.pool_bins
            ]
        )

        num_branches = 1 + len(self.pool_bins)  # local + pooled branches

        # Predict per-pixel attention over scales
        self.scale_attn = nn.Conv2d(
            out_dim * num_branches,
            num_branches,
            kernel_size=1,
            bias=True,
        )

        # Fuse attended context
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.use_residual_refine = bool(use_residual_refine)
        if self.use_residual_refine:
            hidden_dim = max(out_dim, out_dim * 2)
            self.refine = nn.Sequential(
                nn.Conv2d(out_dim, hidden_dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")

        h, w = x.shape[-2:]

        branches = []

        # Local branch
        local = self.local_proj(x)  # [B, out_dim, H, W]
        branches.append(local)

        # PSP branches
        for pool_proj in self.pool_projs:
            pooled = pool_proj(x)  # [B, out_dim, b, b]
            pooled = F.interpolate(
                pooled,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            branches.append(pooled)

        # [B, num_branches*out_dim, H, W]
        stacked_for_attn = torch.cat(branches, dim=1)

        # [B, num_branches, H, W]
        attn = torch.softmax(self.scale_attn(stacked_for_attn), dim=1)

        # Weighted sum over branches
        y = 0.0
        for i, b in enumerate(branches):
            y = y + attn[:, i : i + 1] * b

        y = self.fuse(y)

        if self.use_residual_refine:
            y = y + self.refine(y)

        return y
