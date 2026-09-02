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
        latent_ids = tuple(int(value) for value in latent_ids)

    ablated = latents.clone()
    ablated[..., list(latent_ids)] = 0
    return ablated


def decode_latents(
    segmenter: nn.Module,
    *,
    latents: torch.Tensor,
    feature_maps: Sequence[torch.Tensor],
    task: str | None = None,
) -> torch.Tensor:
    """Decode modified SAE latents without requiring a model refactor."""

    if not feature_maps:
        raise ValueError("feature_maps cannot be empty.")
    sae = getattr(segmenter, "sae", None)
    if sae is None or not callable(getattr(sae, "decode", None)):
        raise TypeError("segmenter.sae must expose decode(latents).")
    if not callable(getattr(segmenter, "decode", None)):
        raise TypeError("segmenter must expose decode(feature_maps, task=...).")

    reference_map = feature_maps[-1]
    reconstructed_tokens = sae.decode(latents)
    if reconstructed_tokens.ndim != 3:
        raise ValueError("SAE reconstruction must have shape [B, N, D].")

    batch_size, num_tokens, embed_dim = reconstructed_tokens.shape
    height, width = reference_map.shape[-2:]
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
    ids = torch.as_tensor(tuple(latent_ids), device=latents.device)
    if alignment.device != latents.device:
        alignment = alignment.to(latents.device)
    selected_latents = latents.index_select(-1, ids)
    selected_alignment = alignment.index_select(-1, ids)
    return -torch.einsum("bnk,ck->bnc", selected_latents, selected_alignment)
