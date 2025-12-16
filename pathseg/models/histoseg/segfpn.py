# seg_model.py
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- small blocks ----------
class Conv2DBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, dropout=0.2, num_groups=8):
        super().__init__()
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleConv(nn.Module):
    """bilinear x2 → 3×3 conv → 3×3 conv (no transposed convs)"""

    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv1 = Conv2DBlock(in_ch, out_ch, 3, dropout)
        self.conv2 = Conv2DBlock(out_ch, out_ch, 3, dropout)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode="bilinear",
                          align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# assume Conv2DBlock and UpsampleConv are defined as before


class FuseBlock(nn.Module):
    """Concat [low, up(high)] → 1×1 reduce → 3×3 smooth"""

    def __init__(self, low_ch, high_ch_up, out_ch, dropout=0.2):
        super().__init__()
        self.reduce = nn.Conv2d(low_ch + high_ch_up,
                                out_ch,
                                kernel_size=1,
                                bias=False)
        self.smooth = Conv2DBlock(out_ch, out_ch, 3, dropout)

    def forward(self, low, up_high):
        x = torch.cat([low, up_high], dim=1)
        x = self.reduce(x)
        x = self.smooth(x)
        return x


class SegFPN(nn.Module):

    def __init__(self,
                 num_classes,
                 pyramid_channels=None,
                 dropout=0.2,
                 deep_supervision=True,
                 img_skip_ch=64,
                 assert_shapes=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.assert_shapes = assert_shapes

        self.img_to_s1 = nn.Sequential(
            Conv2DBlock(3, 32, dropout=dropout),
            Conv2DBlock(32, img_skip_ch, dropout=dropout),
        )

        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}
        self.pyramid_channels = pyramid_channels
        C4, C8, C16, C32 = (pyramid_channels["s4"], pyramid_channels["s8"],
                            pyramid_channels["s16"], pyramid_channels["s32"])

        self.up32_to_16 = UpsampleConv(C32, C16, dropout=dropout)
        self.fuse16 = FuseBlock(C16, C16, C16, dropout=dropout)

        self.up16_to_8 = UpsampleConv(C16, C8, dropout=dropout)
        self.fuse8 = FuseBlock(C8, C8, C8, dropout=dropout)

        self.up8_to_4 = UpsampleConv(C8, C4, dropout=dropout)
        self.fuse4 = FuseBlock(C4, C4, C4, dropout=dropout)

        self.head = nn.Sequential(
            Conv2DBlock(C4 + img_skip_ch, max(C4 // 2, 32), dropout=dropout),
            nn.Conv2d(max(C4 // 2, 32), num_classes, kernel_size=1),
        )
        if deep_supervision:
            self.aux8 = nn.Conv2d(C8, num_classes, kernel_size=1)
            self.aux16 = nn.Conv2d(C16, num_classes, kernel_size=1)

    def forward(self, x, feats: Dict[str,
                                     torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        if self.assert_shapes:
            assert feats["s4"].shape[1] == self.pyramid_channels["s4"]
            assert feats["s8"].shape[1] == self.pyramid_channels["s8"]
            assert feats["s16"].shape[1] == self.pyramid_channels["s16"]
            assert feats["s32"].shape[1] == self.pyramid_channels["s32"]

        s1 = self.img_to_s1(x)
        s4, s8, s16, s32 = feats["s4"], feats["s8"], feats["s16"], feats["s32"]

        p16 = self.fuse16(s16, self.up32_to_16(s32))  # 1/16
        p8 = self.fuse8(s8, self.up16_to_8(p16))  # 1/8
        p4 = self.fuse4(s4, self.up8_to_4(p8))  # 1/4

        s1_1_4 = F.interpolate(s1,
                               size=p4.shape[-2:],
                               mode="bilinear",
                               align_corners=False)
        logits_1_4 = self.head(torch.cat([p4, s1_1_4], dim=1))
        main = F.interpolate(logits_1_4,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)

        out = {"main": main}
        if self.deep_supervision:
            out["aux8"] = F.interpolate(self.aux8(p8),
                                        size=(H, W),
                                        mode="bilinear",
                                        align_corners=False)
            out["aux16"] = F.interpolate(self.aux16(p16),
                                         size=(H, W),
                                         mode="bilinear",
                                         align_corners=False)
        return out
