from typing import Tuple

import torch
from torch import nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class Transforms(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        max_brightness_delta=32,
        max_contrast_factor=0.5,
        saturation_factor=0.5,
        max_hue_delta=18,
    ):
        super().__init__()

        self.img_size = img_size

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_jitter = T.ScaleJitter(
            target_size=img_size,
            scale_range=scale_range,
            antialias=True,
        )

        self.random_crop = T.RandomCrop(img_size)

    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(img,
                                      brightness_factor=self.random_factor(
                                          self.max_brightness_factor))

        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )

        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(
                    self.max_saturation_factor),
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )

        return img

    def color_jitter(self, img):
        img = self.brightness(img)

        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)

        return img

    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def crop(self, img, target: dict):
        img_crop, target_crop = self.random_crop(img, target)

        mask_sums = target_crop["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            return self.crop(img, target)

        target_crop["masks"] = target_crop["masks"][non_empty_mask]
        target_crop["labels"] = target_crop["labels"][non_empty_mask]

        return img_crop, target_crop

    def forward(self, img, target: dict):
        img = self.color_jitter(img)

        img, target = self.random_horizontal_flip(img, target)

        img, target = self.scale_jitter(img, target)

        img, target = self.pad(img, target)

        return self.crop(img, target)


class CustomTransforms(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        max_brightness_delta=32,
        max_contrast_factor=0.5,
        saturation_factor=0.5,
        max_hue_delta=18,
    ):
        super().__init__()

        self.img_size = img_size

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_range = scale_range

        self.random_crop = T.RandomCrop(img_size)

    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(img,
                                      brightness_factor=self.random_factor(
                                          self.max_brightness_factor))

        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )

        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(
                    self.max_saturation_factor),
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )

        return img

    def color_jitter(self, img):
        img = self.brightness(img)

        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)

        return img

    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding, padding_mode="edge")
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def crop(self, img, target: dict):
        img_crop, target_crop = self.random_crop(img, target)

        mask_sums = target_crop["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            return self.crop(img, target)

        target_crop["masks"] = target_crop["masks"][non_empty_mask]
        target_crop["labels"] = target_crop["labels"][non_empty_mask]

        return img_crop, target_crop

    def forward(self, img, target: dict):
        img = self.color_jitter(img)

        img, target = self.random_horizontal_flip(img, target)

        c, h, w = img.shape
        # this change is to control the scale for varying input size with same magnification
        # similar results can be obtained by swapping the h and w, but find it less intuitive
        target_size = max(h, w)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            # target_size=(w, h),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        # if img.shape[-2] < self.img_size[-2] or img.shape[-1] < self.img_size[
        #         -1]:
        # print("Before scale jitter:", h, w)
        # print("After scale jitter:", img.shape[1], img.shape[2])

        img, target = self.pad(img, target)

        return self.crop(img, target)


def _as_tuple(x) -> Tuple[int, int]:
    return (int(x), int(x)) if isinstance(x, (int, float)) else (int(x[0]),
                                                                 int(x[1]))


def _next_multiple(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k  # ceil to multiple


class CustomTransformsVaryingSize(nn.Module):
    """
    Train-time transforms (histopath-safe) with ViT patch=14 and FPN /32:
      - ColorJitter → HFlip → ScaleJitter (square target around native scale)
      - If both dims ≥ img_size: random crop to img_size (prefer non-empty).
      - Else:
          * crop only the dims that exceed img_size,
          * pad each dim up to next multiple of lcm_align, then cap at img_size,
          * no final forced crop (max size is img_size; smaller is allowed).

    Image padding: replicate
    Mask padding: constant 0
    """

    def __init__(
            self,
            img_size: tuple[int, int],
            scale_range: tuple[float, float],
            max_brightness_delta=32,
            max_contrast_factor=0.5,
            saturation_factor=0.5,
            max_hue_delta=18,
            disable_color_jitter: bool = False,
            lcm_align: int = 224,  # LCM(14, 32)
    ):
        super().__init__()

        self.img_size = _as_tuple(img_size)
        self.scale_range = scale_range
        self.lcm_align = int(lcm_align)

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = float(max_contrast_factor)
        self.max_saturation_factor = float(saturation_factor)
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.random_crop_to_out = T.RandomCrop(self.img_size)

        self.disable_color_jitter = disable_color_jitter

    # ---------- color jitter ----------
    def random_factor(self, factor: float, center: float = 1.0) -> float:
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(img,
                                      brightness_factor=self.random_factor(
                                          self.max_brightness_factor))
        return img

    def contrast(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img,
                                    contrast_factor=self.random_factor(
                                        self.max_contrast_factor))
        return img

    def saturation_and_hue(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(img,
                                      saturation_factor=self.random_factor(
                                          self.max_saturation_factor))
        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img,
                               hue_factor=self.random_factor(
                                   self.max_hue_delta, center=0.0))
        return img

    def color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        img = self.brightness(img)
        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)
        return img

    # ---------- helpers ----------
    def _crop_dims_down_to_max(self, img: torch.Tensor, target: dict,
                               max_h: int, max_w: int):
        """Randomly crop ONLY the dimensions that exceed (max_h, max_w)."""
        _, h, w = img.shape
        out_h = min(h, max_h)
        out_w = min(w, max_w)
        if out_h == h and out_w == w:
            return img, target

        top = 0 if h == out_h else int(torch.randint(0, h - out_h + 1, (1, )))
        left = 0 if w == out_w else int(torch.randint(0, w - out_w + 1, (1, )))

        img_c = F.crop(img, top=top, left=left, height=out_h, width=out_w)

        tgt_c = target
        if "masks" in target and target["masks"] is not None:
            tgt_c = dict(target)
            tgt_c["masks"] = F.crop(target["masks"],
                                    top=top,
                                    left=left,
                                    height=out_h,
                                    width=out_w)
        return img_c, tgt_c

    def _pad_to(self, img: torch.Tensor, target: dict, out_h: int, out_w: int):
        """Pad to (out_h, out_w) with replicate (image) and constant 0 (masks)."""
        _, h, w = img.shape
        pad_h = max(0, out_h - h)
        pad_w = max(0, out_w - w)
        if pad_h == 0 and pad_w == 0:
            return img, target

        padding = [0, 0, pad_w, pad_h]  # [left, top, right, bottom]
        img = F.pad(img, padding, padding_mode="edge")
        if "masks" in target and target["masks"] is not None:
            target = dict(target)
            target["masks"] = F.pad(target["masks"], padding)
        return img, target

    def _random_crop_to_non_empty(self,
                                  img: torch.Tensor,
                                  target: dict,
                                  out_h: int,
                                  out_w: int,
                                  max_tries: int = 20):
        """Random-crop to (out_h, out_w), preferring at least one non-empty mask."""
        rcrop = T.RandomCrop((out_h, out_w))
        for _ in range(max_tries):
            img_c, tgt_c = rcrop(img, target)
            if "masks" in tgt_c and tgt_c["masks"] is not None:
                mask_sums = tgt_c["masks"].sum(dim=[-2, -1])
                if (mask_sums > 0).any():
                    non_empty = mask_sums > 0
                    tgt_c["masks"] = tgt_c["masks"][non_empty]
                    if "labels" in tgt_c and tgt_c["labels"] is not None:
                        tgt_c["labels"] = tgt_c["labels"][non_empty]
                    return img_c, tgt_c
            else:
                return img_c, tgt_c
        # fallback (accept whatever)
        return rcrop(img, target)

    # ---------- main ----------
    def forward(self, img: torch.Tensor, target: dict):
        # 1) color jitter

        if self.disable_color_jitter is False:
            img = self.color_jitter(img)

        # 2) flip
        img, target = self.random_horizontal_flip(img, target)

        # 3) scale jitter around native (square target for stability)
        _, h0, w0 = img.shape
        target_size = max(h0, w0)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        _, h, w = img.shape
        Hc, Wc = self.img_size

        # A) if both dims >= img_size → random crop to img_size
        if h >= Hc and w >= Wc:
            return self._random_crop_to_non_empty(img, target, Hc, Wc)

        # B) otherwise: crop only exceeding dims (independently)
        img, target = self._crop_dims_down_to_max(img, target, Hc, Wc)
        _, h, w = img.shape  # refresh

        # C) pad each dim to next multiple of lcm_align, then cap at img_size
        out_h = min(_next_multiple(h, self.lcm_align), Hc)
        out_w = min(_next_multiple(w, self.lcm_align), Wc)
        img, target = self._pad_to(img, target, out_h, out_w)

        # No final forced crop — max size is img_size; smaller allowed
        return img, target


class AIGradingTransforms(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        max_brightness_delta=32,   # kept for API compatibility, not used now
        max_contrast_factor=0.5,  # kept for API compatibility, not used now
        saturation_factor=0.5,    # interpreted as +/- around 1.0
        max_hue_delta=36,
        rotation_range_deg: float = 90.0,
        shift_range_frac: float = 0.2,
    ):
        """
        PyTorch equivalent of the Keras ImageDataGenerator + random_adjust_saturation:

        - Rotation up to ±rotation_range_deg
        - Translation up to shift_range_frac * H/W
        - Zoom via ScaleJitter(scale_range)
        - Color jitter: random hue in [-max_hue_delta, max_hue_delta] degrees
                        random saturation in [1 - saturation_factor, 1 + saturation_factor]
        """
        super().__init__()
        from torchvision.transforms import InterpolationMode

        self.img_size = img_size

        # color jitter params, assuming img in [0,1]
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0  # TF uses hue in [−0.5,0.5]

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_range = scale_range
        self.random_crop = T.RandomCrop(img_size)

        # geometric: rotation + translation (zoom handled by ScaleJitter below)
        self.random_affine = T.RandomAffine(
            degrees=rotation_range_deg,
            translate=(shift_range_frac, shift_range_frac),
            scale=(1.0, 1.0),  # no extra zoom here, we use ScaleJitter
            interpolation=InterpolationMode.BILINEAR,
            fill=0,  # background for images; masks will get nearest + 0 fill
        )

    # -------------------- color jitter: TF-style random_adjust_saturation -----

    def random_saturation_and_hue_tf(self, img: torch.Tensor) -> torch.Tensor:
        """
        Rough equivalent of TF's random_adjust_saturation:

            delta_hue ~ U[-max_delta_hue, max_delta_hue]
            saturation_factor ~ U[min_sat, max_sat]

        where:
            max_delta_hue = self.max_hue_delta (in [-1,1])
            min_sat = 1 - self.max_saturation_factor
            max_sat = 1 + self.max_saturation_factor
        """
        # adjust hue
        delta_hue = torch.empty(1).uniform_(
            -self.max_hue_delta,
            self.max_hue_delta,
        ).item()
        img = F.adjust_hue(img, delta_hue)

        # adjust saturation
        sat_min = max(0.0, 1.0 - self.max_saturation_factor)
        sat_max = 1.0 + self.max_saturation_factor
        saturation_factor = torch.empty(1).uniform_(sat_min, sat_max).item()
        img = F.adjust_saturation(img, saturation_factor)

        # clip back to [0,1] (TF clips after hue and saturation)
        img = img.clamp(0.0, 1.0)

        return img

    # ------------------------------ pad & crop (unchanged) --------------------

    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def crop(self, img, target: dict):
        img_crop, target_crop = self.random_crop(img, target)

        mask_sums = target_crop["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            # recurse until we get a crop with at least one positive mask
            return self.crop(img, target)

        target_crop["masks"] = target_crop["masks"][non_empty_mask]
        target_crop["labels"] = target_crop["labels"][non_empty_mask]

        return img_crop, target_crop

    # -------------------------------- forward ---------------------------------

    def forward(self, img, target: dict):
        # img: (C,H,W), float in [0,1]
        # target: {"masks": (Q,H,W), "labels": (Q,)}

        img = self.random_saturation_and_hue_tf(img)

        img, target = self.random_horizontal_flip(img, target)


        c, h, w = img.shape
        target_size = max(h, w)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        img, target = self.pad(img, target)

        img, target = self.random_affine(img, target)

        return self.crop(img, target)
