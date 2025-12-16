import torch
from torch import Tensor

# ------------------------
# Helpers
# ------------------------


def _to_chlast_float(img: Tensor) -> tuple[Tensor, bool]:
    """
    Accepts (C,H,W) or (H,W,3) or (B,C,H,W) or (B,H,W,3).
    Returns (B,H,W,3) float32 on the same device, and a flag whether batch dim was added.
    """
    if img.dtype == torch.uint8:
        img = img.float()
    device = img.device

    # add batch if missing
    added_batch = False
    if img.ndim == 3:
        added_batch = True
        img = img.unsqueeze(0)  # (1, ..., ..., ...)

    # now img is 4D
    if img.shape[1] == 3 and img.ndim == 4:  # (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1).contiguous()
    elif img.shape[-1] == 3:  # already (B, H, W, C)
        pass
    else:
        raise ValueError(
            "Expected channels to be size 3 in first or last dim.")

    # if likely 0..1, scale to 0..255 (heuristic but standard)
    if img.max() <= 1.5:
        img = img * 255.0

    return img.to(device=device, dtype=torch.float32), added_batch


def _unit_norm_cols(M: Tensor, eps: float = 1e-12) -> Tensor:
    return M / (M.norm(dim=0, keepdim=True) + eps)


# ------------------------
# Macenko: estimate HE (torch)
# ------------------------


@torch.no_grad()
def estimate_stain_matrix(
    img: Tensor,
    Io: float = 240.0,
    alpha_percentile: float = 1.0,
    beta: float = 0.15,
) -> Tensor:
    """
    Estimate (3,2) stain OD matrix HE from a single image tensor.

    Args:
        img: Tensor shaped (C,H,W) | (H,W,3) | (1,C,H,W) | (1,H,W,3).
             dtype float in [0,1] or uint8. Any device.
        Io:  transmitted light intensity (≈ background white).
        alpha_percentile: percentile for angular trimming (outlier-robust).
        beta: OD threshold to drop background/transparent pixels.

    Returns:
        HE: (3, 2) tensor (unit-norm columns), same device as img.
    """
    device = img.device
    x, added_batch = _to_chlast_float(img)  # (B,H,W,3)
    if x.shape[0] != 1:
        raise ValueError(
            "Use estimate_stain_matrix_batch_torch for batched input.")

    x = x[0]  # (H,W,3)
    H, W, _ = x.shape
    I = x.view(-1, 3)  # (N,3)

    # Optical density
    OD = -(I + 1.0).clamp_min(1e-6).log() + torch.log(
        torch.tensor(Io, device=device))

    # Remove background/transparent
    mask = ~(OD < beta).any(dim=1)
    OD_t = OD[mask]  # (M,3)
    if OD_t.numel() == 0:
        raise ValueError(
            "No tissue pixels after OD thresholding. Adjust beta or check image."
        )

    # Covariance and eigendecomposition
    # cov = (X^T X) / (n-1)
    X = (OD_t - OD_t.mean(dim=0, keepdim=True))
    cov = X.T @ X / max(1, (X.shape[0] - 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]  # (3,3), columns are PCs

    # Project onto top-2 PCs
    OD_proj = OD_t @ eigvecs[:, :2]  # (M,2)
    phi = torch.atan2(OD_proj[:, 1], OD_proj[:, 0])

    # Robust extremes via percentiles
    min_phi = torch.quantile(phi, alpha_percentile / 100.0)
    max_phi = torch.quantile(phi, 1.0 - alpha_percentile / 100.0)

    vmin = eigvecs[:, :2] @ torch.stack(
        [torch.cos(min_phi), torch.sin(min_phi)])  # (3,)
    vmax = eigvecs[:, :2] @ torch.stack(
        [torch.cos(max_phi), torch.sin(max_phi)])  # (3,)

    # Heuristic ordering: H first, E second
    if vmin[0] > vmax[0]:
        HE = torch.stack([vmin, vmax], dim=1)
    else:
        HE = torch.stack([vmax, vmin], dim=1)

    return _unit_norm_cols(HE).to(device)


@torch.no_grad()
def estimate_stain_matrix_batch(
    imgs: Tensor,
    Io: float = 240.0,
    alpha_percentile: float = 1.0,
    beta: float = 0.15,
) -> Tensor:
    """
    Batched wrapper: per-image HE estimation.

    Args:
        imgs: (B,C,H,W) | (B,H,W,3), float [0,1] or uint8.

    Returns:
        HE_batch: (B, 3, 2)
    """
    x, _ = _to_chlast_float(imgs)  # (B,H,W,3)
    HE_list = []
    for b in range(x.shape[0]):
        HE = estimate_stain_matrix(
            x[b],
            Io=Io,
            alpha_percentile=alpha_percentile,
            beta=beta,
        )
        HE_list.append(HE)
    return torch.stack(HE_list, dim=0)  # (B,3,2)


# ------------------------
# Normalize + unmix (torch)
# ------------------------
@torch.no_grad()
def normalize_and_unmix(
    img: Tensor,
    HE: Tensor | None = None,
    Io: float = 240.0,
    HE_ref: Tensor | None = None,
    maxC_ref: Tensor | None = None,
    alpha_percentile: float = 1.0,
    beta: float = 0.15,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    If HE is None, estimate it from `img` (Macenko), else use the provided HE.
    Returns (Inorm, H_only, E_only, HE_used)
    """
    device = img.device
    x, _ = _to_chlast_float(img)  # (1,H,W,3) expected; raises if batch>1
    if x.shape[0] != 1:
        raise ValueError(
            "Use normalize_and_unmix_batch_torch for batched input.")
    x = x[0]
    Hh, Ww, _ = x.shape
    I = x.view(-1, 3)

    # ---- get or estimate HE ----
    if HE is None:
        HE = estimate_stain_matrix(x,
                                   Io=Io,
                                   alpha_percentile=alpha_percentile,
                                   beta=beta)
    else:
        # ensure (3,2), float, device
        HE = HE.to(device=device, dtype=torch.float32)

    # ---- defaults for HE_ref / maxC_ref ----
    if HE_ref is None:
        HE_ref = torch.tensor(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]],
            device=device,
            dtype=torch.float32)
        HE_ref = _unit_norm_cols(HE_ref)
    if maxC_ref is None:
        maxC_ref = torch.tensor([1.9705, 1.0308],
                                device=device,
                                dtype=torch.float32)

    # ---- OD + concentrations ----
    OD = -(I + 1.0).clamp_min(1e-6).log() + torch.log(
        torch.tensor(Io, device=device))
    HE_pinv = torch.linalg.pinv(HE)  # (2,3)
    C = (HE_pinv @ OD.T)  # (2,N)

    # ---- scaling and reconstruction ----
    maxC = torch.stack(
        [torch.quantile(C[0], 0.99),
         torch.quantile(C[1], 0.99)])
    scale = (maxC / maxC_ref).clamp_min(1e-6)
    C_scaled = C / scale[:, None]

    OD_norm = (HE_ref @ C_scaled).T
    I_norm = (torch.tensor(Io, device=device) * torch.exp(-OD_norm)).clamp(
        0, 255.0)
    I_norm = I_norm.view(Hh, Ww, 3).to(torch.uint8)

    H_only_OD = (HE_ref[:, [0]] @ C_scaled[[0], :]).T
    E_only_OD = (HE_ref[:, [1]] @ C_scaled[[1], :]).T
    H_only = (torch.tensor(Io, device=device) * torch.exp(-H_only_OD)).clamp(
        0, 255.0)
    E_only = (torch.tensor(Io, device=device) * torch.exp(-E_only_OD)).clamp(
        0, 255.0)
    H_only = H_only.view(Hh, Ww, 3).to(torch.uint8)
    E_only = E_only.view(Hh, Ww, 3).to(torch.uint8)

    return I_norm, H_only, E_only, HE


@torch.no_grad()
def normalize_and_unmix_batch(
    imgs: Tensor,
    HE_batch: Tensor | None = None,
    Io: float = 240.0,
    HE_ref: Tensor | None = None,
    maxC_ref: Tensor | None = None,
    alpha_percentile: float = 1.0,
    beta: float = 0.15,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    If HE_batch is None, estimate per-image HE internally.
    Returns (Inorm_b, H_b, E_b, HE_used_b) with shapes (B,H,W,3) and (B,3,2)
    """
    x, _ = _to_chlast_float(imgs)  # (B,H,W,3)
    B, Hh, Ww, _ = x.shape
    outs_Inorm, outs_H, outs_E, outs_HE = [], [], [], []

    for b in range(B):
        HE_b = None if HE_batch is None else HE_batch[b]
        Inorm, Himg, Eimg, HE_used = normalize_and_unmix(
            x[b],
            HE=HE_b,
            Io=Io,
            HE_ref=HE_ref,
            maxC_ref=maxC_ref,
            alpha_percentile=alpha_percentile,
            beta=beta,
        )
        outs_Inorm.append(Inorm)
        outs_H.append(Himg)
        outs_E.append(Eimg)
        outs_HE.append(HE_used)

    return (torch.stack(outs_Inorm, 0), torch.stack(outs_H, 0),
            torch.stack(outs_E, 0), torch.stack(outs_HE, 0))
