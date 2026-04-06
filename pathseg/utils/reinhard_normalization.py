from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


def _compute_lab_stats_from_rgb_uint8(
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std in Lab space from an RGB uint8 image.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image of shape [H, W, 3], dtype uint8.

    Returns
    -------
    mean : np.ndarray
        Shape [3], float32.
    std : np.ndarray
        Shape [3], float32.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape [H, W, 3], got {image_rgb.shape}"
        )
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got dtype={image_rgb.dtype}")

    image_float = image_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2Lab)
    mean = lab.mean(axis=(0, 1)).astype(np.float32)
    std = lab.std(axis=(0, 1)).astype(np.float32)
    return mean, std


def reinhard_normalize_rgb(
    image_rgb: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    std_threshold: float = 5.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Apply Reinhard stain normalization to an RGB uint8 image.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image of shape [H, W, 3], dtype uint8.
    target_mean : np.ndarray
        Target Lab mean, shape [3].
    target_std : np.ndarray
        Target Lab std, shape [3].
    std_threshold : float
        If sum(source_std) <= std_threshold, skip normalization
        to mimic the original code behavior.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized RGB uint8 image of shape [H, W, 3].
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape [H, W, 3], got {image_rgb.shape}"
        )
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got dtype={image_rgb.dtype}")

    image_float = image_rgb.astype(np.float32) / 255.0

    if not np.any(image_float):
        return image_rgb.copy()

    source_lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2Lab)
    source_mean = source_lab.mean(axis=(0, 1)).astype(np.float32)
    source_std = source_lab.std(axis=(0, 1)).astype(np.float32)

    # Preserve original behavior from the old code
    if float(source_std.sum()) <= std_threshold:
        return image_rgb.copy()

    norm_lab = source_lab.copy()
    norm_lab[..., 0] = (
        (norm_lab[..., 0] - source_mean[0]) * (target_std[0] / max(source_std[0], eps))
    ) + target_mean[0]
    norm_lab[..., 1] = (
        (norm_lab[..., 1] - source_mean[1]) * (target_std[1] / max(source_std[1], eps))
    ) + target_mean[1]
    norm_lab[..., 2] = (
        (norm_lab[..., 2] - source_mean[2]) * (target_std[2] / max(source_std[2], eps))
    ) + target_mean[2]

    norm_rgb = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2RGB)
    norm_rgb = np.clip(norm_rgb * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return norm_rgb


class ReinhardNormalize:
    """
    Callable transform for RGB images using Reinhard normalization.

    Accepts:
      - PIL.Image
      - np.ndarray [H, W, 3] uint8
      - torch.Tensor [3, H, W] uint8 or float in [0,255] / [0,1]

    Returns:
      - same style as a PIL->numpy->torch preprocessing flow:
        a torch.Tensor [3, H, W] float32 in [0, 1] if `return_tensor=True`
        otherwise a PIL.Image
    """

    def __init__(
        self,
        target_image_path: str | Path,
        return_tensor: bool = True,
        std_threshold: float = 5.0,
    ):
        target_image_path = Path(target_image_path)
        if not target_image_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        target_rgb = np.array(
            Image.open(target_image_path).convert("RGB"), dtype=np.uint8
        )
        self.target_mean, self.target_std = _compute_lab_stats_from_rgb_uint8(
            target_rgb
        )
        self.return_tensor = return_tensor
        self.std_threshold = std_threshold

    def __call__(self, image):
        image_rgb = self._to_numpy_uint8_rgb(image)
        image_rgb = reinhard_normalize_rgb(
            image_rgb=image_rgb,
            target_mean=self.target_mean,
            target_std=self.target_std,
            std_threshold=self.std_threshold,
        )

        if self.return_tensor:
            return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        return Image.fromarray(image_rgb)

    @staticmethod
    def _to_numpy_uint8_rgb(image) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"), dtype=np.uint8)

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected numpy image [H, W, 3], got {image.shape}")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return image

        if torch.is_tensor(image):
            if image.ndim != 3:
                raise ValueError(f"Expected tensor [3, H, W], got {tuple(image.shape)}")
            if image.shape[0] != 3:
                raise ValueError(f"Expected tensor [3, H, W], got {tuple(image.shape)}")

            x = image.detach().cpu()
            if x.dtype.is_floating_point:
                if x.max() <= 1.0:
                    x = x * 255.0
                x = x.clamp(0, 255).round().to(torch.uint8)
            else:
                x = x.to(torch.uint8)

            return x.permute(1, 2, 0).numpy()

        raise TypeError(f"Unsupported image type: {type(image)}")
