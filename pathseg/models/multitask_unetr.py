from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.unetr.pyramid_encoder import ViTEncoderPyramidHooks


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        num_groups: int = 8,
    ):
        super().__init__()
        pad = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageStem(nn.Module):
    def __init__(self, out_ch: int = 32, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(3, out_ch, k=3, num_groups=num_groups),
            ConvGNAct(out_ch, out_ch, k=3, num_groups=num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AddRefineBlock(nn.Module):
    """
    x_high -> upsample to skip size -> project
    x_skip -> project
    add -> refine
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_groups: int = 8,
    ):
        super().__init__()
        self.proj_high = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.proj_skip = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            ConvGNAct(out_ch, out_ch, k=3, num_groups=num_groups),
            ConvGNAct(out_ch, out_ch, k=3, num_groups=num_groups),
        )

    def forward(self, x_high: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_high = F.interpolate(
            x_high,
            size=x_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.proj_high(x_high) + self.proj_skip(x_skip)
        x = self.refine(x)
        return x


class SharedSegDecoder(nn.Module):
    """
    Assumes feats already come resized from the adapter:
      - s32: coarsest
      - s16
      - s8
      - s4: finest pyramid level
    """

    def __init__(
        self,
        pyramid_channels: dict[str, int],
        decoder_dim: int = 128,
        img_skip_ch: int = 32,
        num_groups: int = 8,
    ):
        super().__init__()

        c4 = pyramid_channels["s4"]
        c8 = pyramid_channels["s8"]
        c16 = pyramid_channels["s16"]
        c32 = pyramid_channels["s32"]

        self.img_stem = ImageStem(out_ch=img_skip_ch, num_groups=num_groups)

        self.proj32 = nn.Conv2d(c32, decoder_dim, kernel_size=1, bias=False)
        self.up16 = AddRefineBlock(
            in_ch=decoder_dim,
            skip_ch=c16,
            out_ch=decoder_dim,
            num_groups=num_groups,
        )
        self.up8 = AddRefineBlock(
            in_ch=decoder_dim,
            skip_ch=c8,
            out_ch=decoder_dim,
            num_groups=num_groups,
        )
        self.up4 = AddRefineBlock(
            in_ch=decoder_dim,
            skip_ch=c4,
            out_ch=decoder_dim,
            num_groups=num_groups,
        )

        self.out = nn.Sequential(
            ConvGNAct(
                decoder_dim + img_skip_ch, decoder_dim, k=3, num_groups=num_groups
            ),
            ConvGNAct(decoder_dim, decoder_dim, k=3, num_groups=num_groups),
        )

    def forward(self, x: torch.Tensor, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h, w = x.shape[-2:]

        s1 = self.img_stem(x)
        s4, s8, s16, s32 = feats["s4"], feats["s8"], feats["s16"], feats["s32"]

        x32 = self.proj32(s32)
        x16 = self.up16(x32, s16)
        x8 = self.up8(x16, s8)
        x4 = self.up4(x8, s4)

        s1_to_x4 = F.interpolate(
            s1,
            size=x4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec = self.out(torch.cat([x4, s1_to_x4], dim=1))
        dec = F.interpolate(dec, size=(h, w), mode="bilinear", align_corners=False)
        return dec


class TwoHeadViTPyramidSeg(nn.Module):
    def __init__(
        self,
        encoder_id: str = "h0-mini",
        num_classes_a: int = 16,
        num_classes_b: int = 7,
        extract_layers: tuple[int, ...] = (3, 6, 9, 12),
        pyramid_channels: dict[str, int] | None = None,
        decoder_dim: int = 128,
        img_skip_ch: int = 32,
        condition_b_on_a: bool = False,
        num_groups: int = 8,
    ):
        super().__init__()

        from pathseg.models.encoder import build_encoder as build_vit_encoder

        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b

        vit, vit_meta = build_vit_encoder(encoder_id=encoder_id)
        self.encoder = vit

        pixel_mean = torch.tensor(vit_meta["pixel_mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(vit_meta["pixel_std"]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        if pyramid_channels is None:
            pyramid_channels = {"s4": 96, "s8": 128, "s16": 192, "s32": 256}

        self.encoder_adapter = ViTEncoderPyramidHooks(
            vit=vit,
            pyramid_channels=pyramid_channels,
            embed_dim=vit_meta["embed_dim"],
            extract_layers=extract_layers,
        )

        self.decoder = SharedSegDecoder(
            pyramid_channels=pyramid_channels,
            decoder_dim=decoder_dim,
            img_skip_ch=img_skip_ch,
            num_groups=num_groups,
        )

        self.condition_b_on_a = bool(condition_b_on_a)

        self.head_a_refiner = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim, k=3, num_groups=num_groups),
            ConvGNAct(decoder_dim, decoder_dim, k=3, num_groups=num_groups),
        )
        self.head_a = nn.Conv2d(decoder_dim, num_classes_a, kernel_size=1)

        in_dim_b = decoder_dim + (num_classes_a if self.condition_b_on_a else 0)
        self.head_b_refiner = nn.Sequential(
            ConvGNAct(in_dim_b, decoder_dim, k=3, num_groups=num_groups),
            ConvGNAct(decoder_dim, decoder_dim, k=3, num_groups=num_groups),
        )
        self.head_b = nn.Conv2d(decoder_dim, num_classes_b, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x_norm = (x - self.pixel_mean) / self.pixel_std

        feats = self.encoder_adapter(x_norm)
        dec = self.decoder(x, feats)

        feat_a = self.head_a_refiner(dec)
        logits_a = self.head_a(feat_a)

        if self.condition_b_on_a:
            probs_a = F.softmax(logits_a, dim=1)
            feat_b = torch.cat([dec, probs_a], dim=1)
        else:
            feat_b = dec

        feat_b = self.head_b_refiner(feat_b)
        logits_b = self.head_b(feat_b)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
        }
