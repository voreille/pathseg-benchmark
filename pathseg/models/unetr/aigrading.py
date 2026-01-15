import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------------


def get_norm2d(norm_type: str,
               num_features: int,
               num_groups: int = 32) -> nn.Module:
    """
    Returns a 2D normalization layer based on norm_type:
        - "bn": BatchNorm2d
        - "group": GroupNorm(num_groups, num_features)
        - "instance": InstanceNorm2d
        - "layer": GroupNorm(1, num_features)  (LayerNorm over channels)
    """
    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "group":
        groups = min(num_groups, num_features)
        if num_features % groups != 0:
            groups = 1
        return nn.GroupNorm(groups, num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == "layer":
        # LayerNorm over channels -> GroupNorm with 1 group
        return nn.GroupNorm(1, num_features)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# -------------------------------------------------------------------------
# Basic ResNet blocks
# -------------------------------------------------------------------------


class IdentityBlock(nn.Module):
    """
    Identity block (no conv on shortcut).
    Expects input channels == filters3.
    """

    def __init__(self,
                 channels: int,
                 filters,
                 kernel_size=3,
                 norm_type="bn",
                 num_groups=32):
        super().__init__()
        filters1, filters2, filters3 = filters
        assert filters3 == channels, "IdentityBlock input channels must equal filters3."

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, filters1, kernel_size=1, bias=False)
        self.bn1 = get_norm2d(norm_type, filters1, num_groups)

        self.conv2 = nn.Conv2d(filters1,
                               filters2,
                               kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self.bn2 = get_norm2d(norm_type, filters2, num_groups)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = get_norm2d(norm_type, filters3, num_groups)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    """
    Conv block (with conv shortcut).
    """

    def __init__(self,
                 in_channels: int,
                 filters,
                 kernel_size=3,
                 stride=2,
                 norm_type="bn",
                 num_groups=32):
        super().__init__()
        filters1, filters2, filters3 = filters

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels,
                               filters1,
                               kernel_size=1,
                               stride=stride,
                               bias=False)
        self.bn1 = get_norm2d(norm_type, filters1, num_groups)

        self.conv2 = nn.Conv2d(filters1,
                               filters2,
                               kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self.bn2 = get_norm2d(norm_type, filters2, num_groups)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = get_norm2d(norm_type, filters3, num_groups)

        self.shortcut_conv = nn.Conv2d(in_channels,
                                       filters3,
                                       kernel_size=1,
                                       stride=stride,
                                       bias=False)
        self.shortcut_bn = get_norm2d(norm_type, filters3, num_groups)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        out = self.relu(out)
        return out


class ConvBlockAtrous(nn.Module):
    """
    Conv block (with conv shortcut) and atrous/dilated conv for the middle layer.
    In TF: dilation_rate=(2,2) for the 3x3 conv, stride usually (1,1).
    """

    def __init__(self,
                 in_channels: int,
                 filters,
                 kernel_size=3,
                 stride=1,
                 dilation=2,
                 norm_type="bn",
                 num_groups=32):
        super().__init__()
        filters1, filters2, filters3 = filters

        padding = dilation  # "same" padding

        self.conv1 = nn.Conv2d(in_channels,
                               filters1,
                               kernel_size=1,
                               stride=stride,
                               bias=False)
        self.bn1 = get_norm2d(norm_type, filters1, num_groups)

        self.conv2 = nn.Conv2d(filters1,
                               filters2,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        self.bn2 = get_norm2d(norm_type, filters2, num_groups)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = get_norm2d(norm_type, filters3, num_groups)

        self.shortcut_conv = nn.Conv2d(in_channels,
                                       filters3,
                                       kernel_size=1,
                                       stride=stride,
                                       bias=False)
        self.shortcut_bn = get_norm2d(norm_type, filters3, num_groups)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        out = self.relu(out)
        return out


# -------------------------------------------------------------------------
# Attention helpers (depth pooling etc.)
# -------------------------------------------------------------------------


def depth_pool_mean(x: torch.Tensor) -> torch.Tensor:
    # Keras depth_pool in Res50 file: reduce_mean over channels
    return x.mean(dim=1, keepdim=True)


def depth_pool_sum(x: torch.Tensor) -> torch.Tensor:
    # Keras depth_pool in cross_attention_pooling: reduce_sum over channels
    return x.sum(dim=1, keepdim=True)


def cross_attention_pooling(x1: torch.Tensor, x2: torch.Tensor):
    """
    cross_attention_pooling(x1, x2) as in the TF code:
        satt = sigmoid(depth_pool_sum(x1 * x2))  -> shape (B,1,H,W)
        sattc = 1 - satt
    Returns:
        satt, sattc
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape for cross_attention_pooling."

    satt = torch.sigmoid(depth_pool_sum(x1 * x2))  # (B,1,H,W)
    sattc = 1.0 - satt
    return satt, sattc


# -------------------------------------------------------------------------
# Pyramid Pooling (PSP-like) module
# -------------------------------------------------------------------------


class PyramidPoolingModule(nn.Module):
    """
    PSP-like pyramid pooling:
      - For each pool_factor p:
          y = AdaptiveAvgPool2d(output_size=(p, p))
          y = Conv2d(in_channels -> out_channels)
          y = Norm + ReLU
          y = upsample back to (H,W)
      - Concatenate input x with all pooled features along channel dim.

    This approximates the TF pool_block behavior in a more standard way.
    """

    def __init__(self,
                 in_channels: int,
                 pool_factors=(1, 2, 3, 6),
                 out_channels_per_pool=128,
                 norm_type="bn",
                 num_groups=32):
        super().__init__()
        self.pool_factors = pool_factors
        self.paths = nn.ModuleList()

        for _ in pool_factors:
            conv = nn.Conv2d(in_channels,
                             out_channels_per_pool,
                             kernel_size=1,
                             bias=False)
            bn = get_norm2d(norm_type, out_channels_per_pool, num_groups)
            self.paths.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))

    def forward(self, x):
        B, C, H, W = x.shape
        out = [x]

        for p, path in zip(self.pool_factors, self.paths):
            pooled = F.adaptive_avg_pool2d(x, output_size=(p, p))
            pooled = path(pooled)
            pooled = F.interpolate(pooled,
                                   size=(H, W),
                                   mode="bilinear",
                                   align_corners=False)
            out.append(pooled)

        return torch.cat(out, dim=1)


# -------------------------------------------------------------------------
# Three-stream ResNet with attention (supres50_a, b, c + poolRes50_attention)
# -------------------------------------------------------------------------


class SupRes50A(nn.Module):
    """
    supres50_a(img) -> Sa1, Sa2, Sa3
    """

    def __init__(self, in_channels: int, norm_type="bn", num_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups

        # Initial conv after AvgPool(4x4)
        self.conv1 = nn.Conv2d(in_channels,
                               16,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.bn1 = get_norm2d(norm_type, 16, num_groups)
        self.relu = nn.ReLU(inplace=True)

        # Stage 2
        self.stage2a = ConvBlock(16, [16, 16, 64],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage2b = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage2c = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 3 (atrous)
        self.stage3a = ConvBlockAtrous(64, [32, 32, 128],
                                       kernel_size=3,
                                       stride=1,
                                       dilation=2,
                                       norm_type=norm_type,
                                       num_groups=num_groups)
        self.stage3b = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3c = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3d = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Channel-wise attention (Sa2)
        self.sa2_fc1 = nn.Linear(128, 64)
        self.sa2_fc2 = nn.Linear(64, 128)

        # Spatial-wise attention (Sa3)
        self.sa3_conv = nn.Conv2d(1, 128, kernel_size=7, padding=3, bias=False)

    def forward(self, img):
        # AvgPool 4x4
        x = F.avg_pool2d(img, kernel_size=4, stride=4)

        x = self.relu(self.bn1(self.conv1(x)))

        # Stage 2
        x = self.stage2a(x)
        x = self.stage2b(x)
        x = self.stage2c(x)

        # Stage 3
        x = self.stage3a(x)
        x = self.stage3b(x)
        x = self.stage3c(x)
        x = self.stage3d(x)

        Sa1 = x  # (B,128,H,W)

        # Sa2: GlobalAveragePooling + FC + sigmoid
        gap = F.adaptive_avg_pool2d(x, output_size=1).view(x.size(0),
                                                           -1)  # (B,128)
        Sa2 = self.sa2_fc1(gap)
        Sa2 = F.relu(Sa2)
        Sa2 = self.sa2_fc2(Sa2)
        Sa2 = torch.sigmoid(Sa2)  # (B,128)

        # Sa3: depth_pool_mean + conv + sigmoid
        Sa3 = depth_pool_mean(x)  # (B,1,H,W)
        Sa3 = self.sa3_conv(Sa3)  # (B,128,H,W)
        Sa3 = torch.sigmoid(Sa3)

        return Sa1, Sa2, Sa3


class SupRes50B(nn.Module):
    """
    supres50_b(img_b, Sa2, Sa3) -> Sb1, Sb2, Sb3
    """

    def __init__(self, in_channels: int, norm_type="bn", num_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(in_channels,
                               16,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.bn1 = get_norm2d(norm_type, 16, num_groups)
        self.relu = nn.ReLU(inplace=True)

        # Stage 2
        self.stage2a = ConvBlock(16, [16, 16, 64],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage2b = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage2c = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 3
        self.stage3a = ConvBlock(64, [32, 32, 128],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage3b = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3c = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3d = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 4 (atrous)
        self.stage4a = ConvBlockAtrous(128, [64, 64, 256],
                                       kernel_size=3,
                                       stride=1,
                                       dilation=2,
                                       norm_type=norm_type,
                                       num_groups=num_groups)
        self.stage4b = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4c = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4d = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4e = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4f = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Sb2: channel-wise attention
        self.sb2_fc1 = nn.Linear(256, 128)
        self.sb2_fc2 = nn.Linear(128, 256)

        # Sb3: spatial-wise attention
        self.sb3_conv = nn.Conv2d(1, 256, kernel_size=7, padding=3, bias=False)

    def forward(self, img, Sa2, Sa3):
        # AvgPool 2x2
        x = F.avg_pool2d(img, kernel_size=2, stride=2)

        x = self.relu(self.bn1(self.conv1(x)))

        # Stage 2
        x = self.stage2a(x)
        x = self.stage2b(x)
        x = self.stage2c(x)

        # Stage 3
        x = self.stage3a(x)
        x = self.stage3b(x)
        x = self.stage3c(x)
        x = self.stage3d(x)

        # Apply Sa2 (channel), Sa3 (spatial)
        # Sa2: (B,128) but x has 128 channels here
        B, C, H, W = x.shape
        assert C == Sa2.shape[
            1], "Sa2 channel dim must match feature channels."
        Sa2_ = Sa2.view(B, C, 1, 1)  # (B,C,1,1)
        Sa3_ = Sa3  # (B,C,H,W) from supres50_a

        f1 = x * Sa2_
        f1 = f1 * Sa3_
        x = x + f1

        # Stage 4
        x = self.stage4a(x)
        x = self.stage4b(x)
        x = self.stage4c(x)
        x = self.stage4d(x)
        x = self.stage4e(x)
        x = self.stage4f(x)
        Sb1 = x  # (B,256,H,W)

        # Sb2
        gap = F.adaptive_avg_pool2d(x, output_size=1).view(x.size(0),
                                                           -1)  # (B,256)
        Sb2 = self.sb2_fc1(gap)
        Sb2 = F.relu(Sb2)
        Sb2 = self.sb2_fc2(Sb2)
        Sb2 = torch.sigmoid(Sb2)  # (B,256)

        # Sb3
        Sb3 = depth_pool_mean(x)  # (B,1,H,W)
        Sb3 = self.sb3_conv(Sb3)  # (B,256,H,W)
        Sb3 = torch.sigmoid(Sb3)

        return Sb1, Sb2, Sb3


class SupRes50C(nn.Module):
    """
    supres50_c(img_c, Sb2, Sb3) -> Sc
    """

    def __init__(self, in_channels: int, norm_type="bn", num_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(in_channels,
                               16,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.bn1 = get_norm2d(norm_type, 16, num_groups)
        self.relu = nn.ReLU(inplace=True)

        # Stage 2
        self.stage2a = ConvBlock(16, [16, 16, 64],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage2b = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage2c = IdentityBlock(64, [16, 16, 64],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 3
        self.stage3a = ConvBlock(64, [32, 32, 128],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage3b = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3c = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage3d = IdentityBlock(128, [32, 32, 128],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 4
        self.stage4a = ConvBlock(128, [64, 64, 256],
                                 kernel_size=3,
                                 stride=2,
                                 norm_type=norm_type,
                                 num_groups=num_groups)
        self.stage4b = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4c = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4d = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4e = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage4f = IdentityBlock(256, [64, 64, 256],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

        # Stage 5 (atrous)
        self.stage5a = ConvBlockAtrous(256, [128, 128, 512],
                                       kernel_size=3,
                                       stride=1,
                                       dilation=2,
                                       norm_type=norm_type,
                                       num_groups=num_groups)
        self.stage5b = IdentityBlock(512, [128, 128, 512],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)
        self.stage5c = IdentityBlock(512, [128, 128, 512],
                                     kernel_size=3,
                                     norm_type=norm_type,
                                     num_groups=num_groups)

    def forward(self, img, Sb2, Sb3):
        x = self.relu(self.bn1(self.conv1(img)))

        # Stage 2
        x = self.stage2a(x)
        x = self.stage2b(x)
        x = self.stage2c(x)

        # Stage 3
        x = self.stage3a(x)
        x = self.stage3b(x)
        x = self.stage3c(x)
        x = self.stage3d(x)

        # Stage 4
        x = self.stage4a(x)
        x = self.stage4b(x)
        x = self.stage4c(x)
        x = self.stage4d(x)
        x = self.stage4e(x)
        x = self.stage4f(x)

        # Apply Sb2, Sb3
        B, C, H, W = x.shape
        assert C == Sb2.shape[
            1], "Sb2 channel dim must match feature channels."
        Sb2_ = Sb2.view(B, C, 1, 1)
        Sb3_ = Sb3

        f2 = x * Sb2_
        f2 = f2 * Sb3_
        x = x + f2

        # Stage 5 (atrous)
        x = self.stage5a(x)
        x = self.stage5b(x)
        x = self.stage5c(x)

        Sc = x  # (B,512,H,W)
        return Sc


class PoolRes50Attention(nn.Module):
    """
    poolRes50_attention(img) -> Sa1, Sb1, Sc
    (matches TF poolRes50_attention)
    """

    def __init__(self, in_channels: int, norm_type="bn", num_groups=32):
        super().__init__()
        self.sa = SupRes50A(in_channels,
                            norm_type=norm_type,
                            num_groups=num_groups)
        self.sb = SupRes50B(in_channels,
                            norm_type=norm_type,
                            num_groups=num_groups)
        self.sc = SupRes50C(in_channels,
                            norm_type=norm_type,
                            num_groups=num_groups)

    def forward(self, img):
        Sa1, Sa2, Sa3 = self.sa(img)
        Sb1, Sb2, Sb3 = self.sb(img, Sa2, Sa3)
        Sc = self.sc(img, Sb2, Sb3)
        return Sa1, Sb1, Sc  # x1, x2, o0


# -------------------------------------------------------------------------
# Self-cross PSP head (selfCrossPsp) translated to PyTorch
# -------------------------------------------------------------------------


class SelfCrossPSP(nn.Module):
    """
    PyTorch version of selfCrossPsp(n_classes, img_c) using the PoolRes50Attention backbone.

    Args:
        in_channels: number of channels of input image (img_c).
        num_classes: number of output segmentation classes.
        norm_type: one of {"bn", "group", "instance", "layer"}.
        group_norm_groups: groups for GroupNorm if norm_type=="group".
    """

    def __init__(
        self,
        in_channels: int=3,
        num_classes: int=7,
        norm_type: str = "bn",
        group_norm_groups: int = 32,
    ):
        super().__init__()
        self.backbone = PoolRes50Attention(in_channels,
                                           norm_type=norm_type,
                                           num_groups=group_norm_groups)

        # After backbone:
        # x1 = Sa1 (C=128), x2 = Sb1 (C=256), x3/o0 = Sc (C=512)
        # Then TF code:
        #   nc_o = channels of o0 (512)
        #   x1 -> Conv2D(nc_o), x2 -> Conv2D(nc_o)

        self.proj1 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            get_norm2d(norm_type, 512, group_norm_groups),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            get_norm2d(norm_type, 512, group_norm_groups),
            nn.ReLU(inplace=True),
        )

        pool_factors = (1, 2, 3, 6)
        self.ppm1 = PyramidPoolingModule(512,
                                         pool_factors=pool_factors,
                                         out_channels_per_pool=128,
                                         norm_type=norm_type,
                                         num_groups=group_norm_groups)
        self.ppm2 = PyramidPoolingModule(512,
                                         pool_factors=pool_factors,
                                         out_channels_per_pool=128,
                                         norm_type=norm_type,
                                         num_groups=group_norm_groups)
        self.ppm3 = PyramidPoolingModule(512,
                                         pool_factors=pool_factors,
                                         out_channels_per_pool=128,
                                         norm_type=norm_type,
                                         num_groups=group_norm_groups)

        # After PPM: channels = 512 + 4*128 = 512 + 512 = 1024
        ppm_out_channels = 512 + 4 * 128

        # Final conv head
        self.conv_head1 = nn.Sequential(
            nn.Conv2d(ppm_out_channels * 3, 256, kernel_size=1, bias=False),
            get_norm2d(norm_type, 256, group_norm_groups),
            nn.ReLU(inplace=True),
        )
        self.conv_head2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1, bias=False),
            get_norm2d(norm_type, 32, group_norm_groups),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Backbone
        x1, x2, o0 = self.backbone(x)
        x3 = o0

        # Project x1, x2 to nc_o (512)
        x1 = self.proj1(x1)  # (B,512,H1,W1)
        x2 = self.proj2(x2)  # (B,512,H2,W2)
        # x3 already (B,512,*,*)

        # Apply PSP on each
        x1 = self.ppm1(x1)
        x2 = self.ppm2(x2)
        x3 = self.ppm3(x3)

        # All should now have same spatial dims for cross-attention
        # (Given the architecture, they will)
        s12, s12c = cross_attention_pooling(x1, x2)
        s13, s13c = cross_attention_pooling(x1, x3)
        s23, s23c = cross_attention_pooling(x2, x3)

        # o13 = s12c * x3 + x3
        o1 = x3 * s12c + x3

        # o22 = s13 * x3 + x3
        o2 = x3 * s13 + x3

        # o32 = s23 * x3 + x3
        o3 = x3 * s23 + x3

        o = torch.cat([o1, o2, o3], dim=1)  # concat along channels

        o = self.conv_head1(o)
        o = self.conv_head2(o)
        o = self.classifier(o)

        # TF code: resize_image(o, (8,8)) with bilinear -> scale_factor 8
        o = F.interpolate(o,
                          scale_factor=8.0,
                          mode="bilinear",
                          align_corners=False)

        return o


# -------------------------------------------------------------------------
# Small sanity check
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # Input: batch of RGB images  (B, 3, H, W)
    model = SelfCrossPSP(in_channels=3, num_classes=21, norm_type="bn")
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:",
          y.shape)  # (2, 21, 256*8?, 256*8?) depending on backbone strides
