from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

# ---- unified origin ----------------------------------------------------------


@dataclass(frozen=True)
class Origin:
    """Per-crop metadata for stitching."""
    img_idx: int
    top: int
    left: int
    valid_h: int
    valid_w: int


# ---- base interface ----------------------------------------------------------


class Tiler(ABC):
    """Interface for tiling/stitching image batches."""

    @abstractmethod
    def window(
        self, imgs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[Origin], List[Tuple[int, int]]]:
        """
        Split images into fixed-size crops.

        Args:
            imgs: sequence of tensors shaped [C, H, W]

        Returns:
            crops: Tensor [N, C, tile, tile]
            origins: per-crop origin records
            img_sizes: per-image (H, W)
        """
        raise NotImplementedError

    @abstractmethod
    def stitch(
        self,
        crop_logits: torch.Tensor,
        origins: Sequence[Origin],
        img_sizes: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """
        Merge per-crop logits back to full-size images.

        Args:
            crop_logits: [N, C, tile, tile]
            origins: origins returned by window()
            img_sizes: per-image (H, W)

        Returns:
            List of tensors [C, H, W], one per input image.
        """
        raise NotImplementedError


# ---- Fixed tiles with border padding -----------------------------------------


class FixedTileTiler(Tiler):
    """
    Slide a fixed-size window over the ORIGINAL image with stride S.
    Border crops are zero-padded to tile_size so the model always sees
    [C, tile, tile]. Stitching averages overlapping predictions uniformly.
    """

    def __init__(self,
                 tile_size: int,
                 stride: int,
                 weighted_blend: bool = False) -> None:
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.weighted_blend = bool(weighted_blend)

    @torch.compiler.disable  # type: ignore[attr-defined]
    def window(
        self, imgs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[Origin], List[Tuple[int, int]]]:
        crops: List[torch.Tensor] = []
        origins: List[Origin] = []
        img_sizes: List[Tuple[int, int]] = []

        for i, img in enumerate(imgs):
            if img.ndim != 3:
                raise ValueError("Each image must be [C, H, W].")
            _, H, W = img.shape
            img_sizes.append((H, W))

            def starts(L: int) -> List[int]:
                if L <= self.tile_size:
                    return [0]
                s = list(range(0, max(L - self.tile_size, 0) + 1, self.stride))
                if s[-1] != L - self.tile_size:
                    s.append(L - self.tile_size)
                return s

            for top in starts(H):
                for left in starts(W):
                    crop = img[:, top:top + self.tile_size,
                               left:left + self.tile_size]
                    pad_h = self.tile_size - crop.shape[1]
                    pad_w = self.tile_size - crop.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        crop = F.pad(crop, (0, pad_w, 0, pad_h),
                                     mode="constant",
                                     value=0)

                    crops.append(crop)
                    valid_h = min(self.tile_size, H - top)
                    valid_w = min(self.tile_size, W - left)
                    origins.append(Origin(i, top, left, valid_h, valid_w))

        if not crops:
            raise ValueError("No crops were generated.")
        return torch.stack(crops, dim=0), origins, img_sizes

    @torch.compiler.disable  # type: ignore[attr-defined]
    def stitch(
        self,
        crop_logits: torch.Tensor,
        origins: Sequence[Origin],
        img_sizes: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        if crop_logits.ndim != 4:
            raise ValueError("crop_logits must be [N, C, tile, tile].")
        device = crop_logits.device
        C = int(crop_logits.shape[1])

        sum_logits: List[torch.Tensor] = []
        hit_count: List[torch.Tensor] = []
        for H, W in img_sizes:
            sum_logits.append(
                torch.zeros((C, H, W), device=device, dtype=crop_logits.dtype))
            hit_count.append(
                torch.zeros((1, H, W), device=device, dtype=crop_logits.dtype))

        for n, o in enumerate(origins):
            sl = crop_logits[n, :, :o.valid_h, :o.valid_w]
            sum_logits[o.img_idx][:, o.top:o.top + o.valid_h,
                                  o.left:o.left + o.valid_w] += sl
            hit_count[o.img_idx][:, o.top:o.top + o.valid_h,
                                 o.left:o.left + o.valid_w] += 1

        return [
            sums / torch.clamp(counts, min=1.0)
            for sums, counts in zip(sum_logits, hit_count)
        ]


# ---- Grid-padded tiler with optional Hann blending ---------------------------


class GridPadTiler(Tiler):
    """
    Pad H,W to a grid so that (H_pad - tile) % stride == 0 and (W_pad - tile) % stride == 0.
    Tiling is done over the PADDED canvas; stitching accumulates on that canvas,
    then a center-crop returns the original size. If weighted_blend=True, a 2D
    Hann window weights overlaps (seam suppression).
    """

    def __init__(
        self,
        tile: int,
        stride: int,
        weighted_blend: bool = False,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ) -> None:
        if not (1 <= stride <= tile):
            raise ValueError("stride must be in [1, tile].")
        self.tile = int(tile)
        self.stride = int(stride)
        self.weighted_blend = bool(weighted_blend)
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)

    # ---- helpers ----
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def _padded_len(self, L: int) -> int:
        T, S = self.tile, self.stride
        if L <= T:
            return T
        k = self._ceil_div(L - T, S)
        return k * S + T

    def _get_padding(
        self, img_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        H, W = img_size
        H_pad = self._padded_len(H)
        W_pad = self._padded_len(W)
        pad_h = H_pad - H
        pad_w = W_pad - W
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return H_pad, W_pad, pad_h, pad_w, pad_top, pad_bot, pad_left, pad_right

    @torch.compiler.disable  # type: ignore[attr-defined]
    def window(
        self, imgs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[Origin], List[Tuple[int, int]]]:
        crops: List[torch.Tensor] = []
        origins: List[Origin] = []
        img_sizes: List[Tuple[int, int]] = []

        T, S = self.tile, self.stride

        for i, img in enumerate(imgs):
            if img.ndim != 3:
                raise ValueError("Each image must be [C, H, W].")
            _, H, W = img.shape
            img_sizes.append((H, W))

            H_pad, W_pad, pad_h, pad_w, pad_top, pad_bot, pad_left, pad_right = self._get_padding(
                (H, W))
            if pad_h or pad_w:
                img_padded = F.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bot),
                    mode=self.pad_mode,
                    value=self.pad_value
                    if self.pad_mode == "constant" else 0.0,
                )
            else:
                img_padded = img

            for top in range(0, H_pad - T + 1, S):
                for left in range(0, W_pad - T + 1, S):
                    crops.append(img_padded[:, top:top + T, left:left + T])
                    origins.append(Origin(i, top, left, T,
                                          T))  # full tiles on padded grid

        if not crops:
            raise ValueError("No crops were generated.")
        return torch.stack(crops, dim=0), origins, img_sizes

    @torch.compiler.disable  # type: ignore[attr-defined]
    def stitch(
        self,
        crop_logits: torch.Tensor,
        origins: Sequence[Origin],
        img_sizes: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        if crop_logits.ndim != 4:
            raise ValueError("crop_logits must be [N, C, T, T].")

        crop_logits = crop_logits.float()
        device = crop_logits.device
        C = int(crop_logits.shape[1])

        sums_list: List[torch.Tensor] = []
        counts_list: List[torch.Tensor] = []
        for (H, W) in img_sizes:
            H_pad, W_pad, *_ = self._get_padding((H, W))
            sums_list.append(
                torch.zeros((C, H_pad, W_pad),
                            device=device,
                            dtype=torch.float32))
            counts_list.append(
                torch.zeros((C, H_pad, W_pad),
                            device=device,
                            dtype=torch.float32))

        w_tile: torch.Tensor | None = None
        if self.weighted_blend:
            T = self.tile
            wy = torch.hann_window(T,
                                   periodic=False,
                                   device=device,
                                   dtype=torch.float32).clamp_min(1e-6)
            wx = torch.hann_window(T,
                                   periodic=False,
                                   device=device,
                                   dtype=torch.float32).clamp_min(1e-6)
            w_tile = (wy[:, None] * wx[None, :])[None]  # [1, T, T]

        for n, o in enumerate(origins):
            sl = crop_logits[n, :, :o.valid_h, :o.valid_w]  # [C, h, w]
            if w_tile is None:
                sums_list[o.img_idx][:, o.top:o.top + o.valid_h,
                                     o.left:o.left + o.valid_w] += sl
                counts_list[o.img_idx][:, o.top:o.top + o.valid_h,
                                       o.left:o.left + o.valid_w] += 1.0
            else:
                w = w_tile[:, :o.valid_h, :o.valid_w]  # [1, h, w]
                sums_list[o.img_idx][:, o.top:o.top + o.valid_h,
                                     o.left:o.left + o.valid_w] += sl * w
                counts_list[o.img_idx][:, o.top:o.top + o.valid_h,
                                       o.left:o.left + o.valid_w] += w

        outs: List[torch.Tensor] = []
        for i, (sums, counts) in enumerate(zip(sums_list, counts_list)):
            H, W = img_sizes[i]
            sums = CenterCrop((H, W))(sums)
            counts = CenterCrop((H, W))(counts)
            denom = torch.where(counts > 0, counts, torch.ones_like(counts))
            outs.append(sums / denom)
        return outs
