"""Minimal corruption functions used by the Histopath-C dataset generator.

The original repository vendors a much larger image-corruption module.  This
version keeps only the six corruptions used by the project and passes an
explicit NumPy random generator, so generation is reproducible without
modifying global random state.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import color

RGBArray = NDArray[np.uint8]
CorruptionFunction = Callable[[RGBArray, int, np.random.Generator], RGBArray]


def _validate_image(image: NDArray[np.generic]) -> RGBArray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if min(array.shape[:2]) < 32:
        raise ValueError("image width and height must be at least 32 pixels")
    return array


def _validate_severity(severity: int) -> int:
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be an integer in [1, 5]")
    return severity


def _to_float(image: RGBArray) -> NDArray[np.float32]:
    return image.astype(np.float32) / 255.0


def _to_uint8(image: NDArray[np.floating]) -> RGBArray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _disk_kernel(radius: int, antialias_sigma: float) -> NDArray[np.float32]:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.GaussianBlur(kernel, (3, 3), sigmaX=antialias_sigma)


def gaussian_noise(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    scales = (0.08, 0.12, 0.18, 0.26, 0.38)
    image = _to_float(_validate_image(image))
    scale = scales[_validate_severity(severity) - 1]
    noisy = image + rng.normal(0.0, scale, size=image.shape)
    return _to_uint8(np.clip(noisy, 0.0, 1.0))


def shot_noise(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    rates = (60.0, 25.0, 12.0, 5.0, 3.0)
    image = _to_float(_validate_image(image))
    rate = rates[_validate_severity(severity) - 1]
    noisy = rng.poisson(image * rate) / rate
    return _to_uint8(np.clip(noisy, 0.0, 1.0))


def defocus_blur(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    del rng  # The operation itself is deterministic.
    parameters = ((3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5))
    radius, antialias_sigma = parameters[_validate_severity(severity) - 1]
    image = _to_float(_validate_image(image))
    kernel = _disk_kernel(radius, antialias_sigma)
    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return _to_uint8(np.clip(blurred, 0.0, 1.0))


def motion_blur(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    """Apply a randomly oriented line-motion kernel.

    This is intentionally simpler than the shift-and-accumulate implementation
    in the original repository, while retaining the same qualitative effect.
    """

    kernel_sizes = (9, 13, 17, 21, 27)
    size = kernel_sizes[_validate_severity(severity) - 1]
    image = _validate_image(image)

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    angle = float(rng.uniform(-45.0, 45.0))
    rotation = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size), flags=cv2.INTER_LINEAR)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        raise RuntimeError("motion-blur kernel is empty")
    kernel /= kernel_sum

    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def brightness(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    del rng
    offsets = (0.1, 0.2, 0.3, 0.4, 0.5)
    image = _to_float(_validate_image(image))
    offset = offsets[_validate_severity(severity) - 1]

    hsv = color.rgb2hsv(image)
    hsv[..., 2] = np.clip(hsv[..., 2] + offset, 0.0, 1.0)
    return _to_uint8(color.hsv2rgb(hsv))


def contrast(
    image: RGBArray,
    severity: int,
    rng: np.random.Generator,
) -> RGBArray:
    del rng
    factors = (0.4, 0.3, 0.2, 0.1, 0.05)
    image = _to_float(_validate_image(image))
    factor = factors[_validate_severity(severity) - 1]
    mean = image.mean(axis=(0, 1), keepdims=True)
    adjusted = (image - mean) * factor + mean
    return _to_uint8(np.clip(adjusted, 0.0, 1.0))


CORRUPTIONS: dict[str, CorruptionFunction] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "contrast": contrast,
}


def get_corruption_names() -> tuple[str, ...]:
    return tuple(CORRUPTIONS)


def corrupt(
    image: RGBArray,
    *,
    severity: int,
    corruption_name: str,
    rng: np.random.Generator,
) -> RGBArray:
    try:
        corruption = CORRUPTIONS[corruption_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown corruption {corruption_name!r}; choose from {get_corruption_names()}"
        ) from exc
    return corruption(image, severity, rng)
