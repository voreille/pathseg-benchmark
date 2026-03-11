from __future__ import annotations

import numpy as np
import torch
from skimage.color import hed2rgb, rgb2hed


def _check_image_tensor(img: torch.Tensor) -> None:
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected image of shape (C, H, W), got {tuple(img.shape)}")
    if img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {img.shape[0]}")


def _sample_uniform(low: float, high: float) -> float:
    return torch.empty(1).uniform_(low, high).item()


def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image tensor in [0, 1] to HSV.
    Input/output shape: (3, H, W)
    H in [0, 1), S in [0, 1], V in [0, 1]
    """
    _check_image_tensor(img)

    r, g, b = img[0], img[1], img[2]

    maxc, _ = img.max(dim=0)
    minc, _ = img.min(dim=0)
    deltac = maxc - minc

    v = maxc

    s = torch.zeros_like(maxc)
    nonzero_v = maxc > 0
    s[nonzero_v] = deltac[nonzero_v] / maxc[nonzero_v]

    h = torch.zeros_like(maxc)

    nonzero_delta = deltac > 0
    rc = torch.zeros_like(r)
    gc = torch.zeros_like(g)
    bc = torch.zeros_like(b)

    rc[nonzero_delta] = (maxc[nonzero_delta] - r[nonzero_delta]) / deltac[nonzero_delta]
    gc[nonzero_delta] = (maxc[nonzero_delta] - g[nonzero_delta]) / deltac[nonzero_delta]
    bc[nonzero_delta] = (maxc[nonzero_delta] - b[nonzero_delta]) / deltac[nonzero_delta]

    mask_r = nonzero_delta & (r == maxc)
    mask_g = nonzero_delta & (g == maxc)
    mask_b = nonzero_delta & (b == maxc)

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = 2.0 + (rc - bc)[mask_g]
    h[mask_b] = 4.0 + (gc - rc)[mask_b]

    h = (h / 6.0) % 1.0

    return torch.stack((h, s, v), dim=0)


def hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV image tensor to RGB.
    Input/output shape: (3, H, W)
    H in [0, 1), S in [0, 1], V in [0, 1]
    """
    _check_image_tensor(img)

    h, s, v = img[0], img[1], img[2]

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    out = torch.empty_like(img)

    for k in range(6):
        mask = i == k
        if k == 0:
            out[0][mask], out[1][mask], out[2][mask] = v[mask], t[mask], p[mask]
        elif k == 1:
            out[0][mask], out[1][mask], out[2][mask] = q[mask], v[mask], p[mask]
        elif k == 2:
            out[0][mask], out[1][mask], out[2][mask] = p[mask], v[mask], t[mask]
        elif k == 3:
            out[0][mask], out[1][mask], out[2][mask] = p[mask], q[mask], v[mask]
        elif k == 4:
            out[0][mask], out[1][mask], out[2][mask] = t[mask], p[mask], v[mask]
        else:
            out[0][mask], out[1][mask], out[2][mask] = v[mask], p[mask], q[mask]

    return out.clamp(0.0, 1.0)


def random_hsv_shift(
    img: torch.Tensor,
    hue_shift_limit: float = 0.05,
    sat_scale_limit: float = 0.15,
    val_scale_limit: float = 0.15,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Pathology-friendly HSV augmentation.

    Parameters
    ----------
    img:
        RGB tensor of shape (3, H, W), expected in [0, 1].
    hue_shift_limit:
        Additive hue shift sampled in [-limit, +limit].
    sat_scale_limit:
        Multiplicative saturation factor sampled in [1-limit, 1+limit].
    val_scale_limit:
        Multiplicative value factor sampled in [1-limit, 1+limit].
    p:
        Probability of applying the augmentation.
    """
    _check_image_tensor(img)

    if torch.rand(()) >= p:
        return img

    hsv = rgb_to_hsv(img).clone()

    hue_shift = _sample_uniform(-hue_shift_limit, hue_shift_limit)
    sat_scale = _sample_uniform(1.0 - sat_scale_limit, 1.0 + sat_scale_limit)
    val_scale = _sample_uniform(1.0 - val_scale_limit, 1.0 + val_scale_limit)

    hsv[0] = (hsv[0] + hue_shift) % 1.0
    hsv[1] = (hsv[1] * sat_scale).clamp(0.0, 1.0)
    hsv[2] = (hsv[2] * val_scale).clamp(0.0, 1.0)

    return hsv_to_rgb(hsv)


def random_hed_shift(
    img: torch.Tensor,
    h_shift_limit: float = 0.03,
    e_shift_limit: float = 0.03,
    d_shift_limit: float = 0.0,
    p: float = 0.3,
) -> torch.Tensor:
    """
    Apply mild HED perturbation to an RGB tensor image.

    Parameters
    ----------
    img:
        torch.Tensor of shape (3, H, W), float in [0, 1].
    h_shift_limit:
        Additive shift sampled in [-limit, +limit] for hematoxylin channel.
    e_shift_limit:
        Additive shift sampled in [-limit, +limit] for eosin channel.
    d_shift_limit:
        Additive shift sampled in [-limit, +limit] for 3rd HED channel.
    p:
        Probability of applying the transform.
    """
    _check_image_tensor(img)

    if torch.rand(()) >= p:
        return img

    device = img.device
    dtype = img.dtype

    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0.0, 1.0)

    hed = rgb2hed(img_np).copy()
    hed[..., 0] += _sample_uniform(-h_shift_limit, h_shift_limit)
    hed[..., 1] += _sample_uniform(-e_shift_limit, e_shift_limit)
    hed[..., 2] += _sample_uniform(-d_shift_limit, d_shift_limit)

    out = hed2rgb(hed)
    out = np.clip(out, 0.0, 1.0)

    return torch.from_numpy(out).permute(2, 0, 1).to(device=device, dtype=dtype)
