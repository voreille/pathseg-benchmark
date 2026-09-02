from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def ablate_latents(
    latents: torch.Tensor,
    latent_ids: int | Sequence[int],
) -> torch.Tensor:
    """Return a copy of ``latents`` with selected dimensions set to zero."""

    if isinstance(latent_ids, int):
        latent_ids = (latent_ids,)
    else:
        latent_ids = tuple(dict.fromkeys(int(value) for value in latent_ids))

    ablated = latents.clone()
    ablated[..., list(latent_ids)] = 0
    return ablated


def decode_latents(
    segmenter: nn.Module,
    *,
    latents: torch.Tensor,
    feature_maps: Sequence[torch.Tensor],
    task: str | None = None,
):
    """Decode modified SAE latents without requiring a model refactor."""

    sae = getattr(segmenter, "sae", None)
    if sae is None or not callable(getattr(sae, "decode", None)):
        raise TypeError("segmenter.sae must expose decode(latents).")
    reconstructed_tokens = sae.decode(latents)
    return decode_reconstructed_tokens(
        segmenter,
        reconstructed_tokens=reconstructed_tokens,
        feature_maps=feature_maps,
        task=task,
    )


def decode_reconstructed_tokens(
    segmenter: nn.Module,
    *,
    reconstructed_tokens: torch.Tensor,
    feature_maps: Sequence[torch.Tensor],
    task: str | None = None,
):
    """Decode already-reconstructed tokens through the semantic head."""

    if not feature_maps:
        raise ValueError("feature_maps cannot be empty.")
    if not callable(getattr(segmenter, "decode", None)):
        raise TypeError("segmenter must expose decode(feature_maps, task=...).")

    reference_map = feature_maps[-1]
    if reconstructed_tokens.ndim != 3:
        raise ValueError("SAE reconstruction must have shape [B, N, D].")
    if not torch.is_tensor(reference_map) or reference_map.ndim != 4:
        raise ValueError("The final feature map must have shape [B,D,H,W].")

    batch_size, num_tokens, embed_dim = reconstructed_tokens.shape
    height, width = reference_map.shape[-2:]
    if batch_size != reference_map.shape[0]:
        raise ValueError("Reconstruction and feature-map batch sizes differ.")
    if embed_dim != reference_map.shape[1]:
        raise ValueError("Reconstruction and feature-map dimensions differ.")
    if num_tokens != height * width:
        raise ValueError(
            f"Received {num_tokens} reconstructed tokens for a {height}x{width} map."
        )
    reconstructed_map = reconstructed_tokens.transpose(1, 2).reshape(
        batch_size,
        embed_dim,
        height,
        width,
    )
    reconstructed_feature_maps = (
        *feature_maps[:-1],
        reconstructed_map,
    )
    return segmenter.decode(reconstructed_feature_maps, task=task)


def ablate_reconstructed_tokens(
    sae: nn.Module,
    *,
    latents: torch.Tensor,
    reconstructed_tokens: torch.Tensor,
    latent_ids: int | Sequence[int],
) -> torch.Tensor:
    """Remove selected latent decoder contributions from a reconstruction.

    This is exactly equivalent to decoding a cloned latent tensor after setting
    the selected coordinates to zero, but avoids repeating the full SAE decoder
    matrix multiplication for every intervention.
    """

    if isinstance(latent_ids, int):
        latent_ids = (latent_ids,)
    else:
        latent_ids = tuple(dict.fromkeys(int(value) for value in latent_ids))
    if not latent_ids:
        raise ValueError("At least one latent ID is required.")
    if latents.ndim != 3 or reconstructed_tokens.ndim != 3:
        raise ValueError("latents and reconstructed_tokens must be three-dimensional.")
    if latents.shape[:2] != reconstructed_tokens.shape[:2]:
        raise ValueError("Latent and reconstructed-token grids do not match.")

    decoder = getattr(sae, "decoder", None)
    weight = getattr(decoder, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise TypeError("sae.decoder must expose a two-dimensional weight.")
    if latents.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"Received {latents.shape[-1]} latents for decoder with "
            f"{weight.shape[1]} columns."
        )

    ids = torch.as_tensor(latent_ids, dtype=torch.long, device=latents.device)
    if torch.any(ids < 0) or torch.any(ids >= latents.shape[-1]):
        raise IndexError("At least one latent ID is out of bounds.")
    selected_latents = latents.index_select(-1, ids)
    selected_columns = weight.index_select(1, ids).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    contribution = torch.einsum(
        "bnk,dk->bnd",
        selected_latents,
        selected_columns,
    )
    return reconstructed_tokens - contribution.to(reconstructed_tokens.dtype)


def linear_logit_delta(
    latents: torch.Tensor,
    alignment: torch.Tensor,
    latent_ids: int | Sequence[int],
) -> torch.Tensor:
    """Exact pre-interpolation logit change caused by zero-ablation.

    ``latents`` is ``[B, N, L]`` and ``alignment`` is ``[C, L]``.  The return
    value has shape ``[B, N, C]``.
    """

    if isinstance(latent_ids, int):
        latent_ids = (latent_ids,)
    else:
        latent_ids = tuple(dict.fromkeys(int(value) for value in latent_ids))
    ids = torch.as_tensor(latent_ids, device=latents.device)
    if alignment.device != latents.device or alignment.dtype != latents.dtype:
        alignment = alignment.to(
            device=latents.device,
            dtype=latents.dtype,
        )
    selected_latents = latents.index_select(-1, ids)
    selected_alignment = alignment.index_select(-1, ids)
    return -torch.einsum("bnk,ck->bnc", selected_latents, selected_alignment)
