from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from pathseg.models.semantic_segmenter import FeatureMaps, SemanticSegmenter


def feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert BxDxHxW feature maps into row-major BxNxD patch tokens."""
    if feature_map.ndim != 4:
        raise ValueError(f"Expected BxDxHxW, got {tuple(feature_map.shape)}.")

    return feature_map.flatten(2).transpose(1, 2)


def tokens_to_feature_map(
    tokens: torch.Tensor,
    reference_map: torch.Tensor,
) -> torch.Tensor:
    """Restore BxNxD tokens to the spatial shape of a reference feature map."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected BxNxD, got {tuple(tokens.shape)}.")

    batch_size, num_tokens, embed_dim = tokens.shape
    ref_batch, ref_dim, height, width = reference_map.shape

    if batch_size != ref_batch or embed_dim != ref_dim:
        raise ValueError(
            "Token and feature-map batch/channel dimensions disagree: "
            f"tokens={tuple(tokens.shape)}, map={tuple(reference_map.shape)}."
        )

    if num_tokens != height * width:
        raise ValueError(f"Got {num_tokens} tokens for a {height}x{width} feature map.")

    return tokens.transpose(1, 2).reshape(
        batch_size,
        embed_dim,
        height,
        width,
    )


class SAESemanticSegmenter(nn.Module):
    """Wrap a semantic segmenter with an SAE on its final feature map.

    The wrapped segmenter retains its concrete architecture and owns the
    encoder and decoder.

    The SAE must expose::

        forward_with_aux(tokens) -> {
            "reconstructed_tokens": Tensor[B, N, D],
            "latents": Tensor[B, N, L],
            ...
        }
    """

    def __init__(
        self,
        segmenter: SemanticSegmenter,
        sae: nn.Module,
    ) -> None:
        super().__init__()

        self.segmenter = segmenter
        self.sae = sae

    @property
    def encoder(self) -> nn.Module:
        """Expose the wrapped encoder for training/freezing logic."""
        return self.segmenter.encoder

    @property
    def decoder(self) -> nn.Module:
        """Expose the wrapped decoder for training/freezing logic."""
        return self.segmenter.decoder

    def encode(self, imgs: torch.Tensor) -> FeatureMaps:
        return self.segmenter.encode(imgs)

    def decode(
        self,
        feature_maps: FeatureMaps,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.segmenter.decode(feature_maps, task=task)

    def ensure_input_resolution(
        self,
        logits_by_task: dict[str, torch.Tensor],
        input_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        return self.segmenter.ensure_input_resolution(
            logits_by_task,
            input_size,
        )

    def forward_sae(self, imgs: torch.Tensor) -> dict[str, Any]:
        feature_maps = self.encode(imgs)

        final_map = feature_maps[-1]
        tokens = feature_map_to_tokens(final_map)

        sae_output = self.sae.forward_with_aux(tokens)

        if not isinstance(sae_output, Mapping):
            raise TypeError(
                "sae.forward_with_aux() must return a mapping, "
                f"got {type(sae_output).__name__}."
            )

        if "reconstructed_tokens" not in sae_output:
            raise KeyError("SAE output must contain 'reconstructed_tokens'.")

        if "latents" not in sae_output:
            raise KeyError("SAE output must contain 'latents'.")

        reconstructed_tokens = sae_output["reconstructed_tokens"]

        reconstructed_map = tokens_to_feature_map(
            reconstructed_tokens,
            final_map,
        )

        reconstructed_feature_maps = (
            *feature_maps[:-1],
            reconstructed_map,
        )

        return {
            **dict(sae_output),
            "tokens": tokens,
            "feature_maps": feature_maps,
            "reconstructed_feature_maps": reconstructed_feature_maps,
        }

    def forward_with_aux(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
        *,
        include_original_logits: bool = False,
    ) -> dict[str, Any]:
        output = self.forward_sae(imgs)

        input_size = tuple(imgs.shape[-2:])

        output["logits"] = self.ensure_input_resolution(
            self.decode(
                output["reconstructed_feature_maps"],
                task=task,
            ),
            input_size,
        )

        if include_original_logits:
            output["original_logits"] = self.ensure_input_resolution(
                self.decode(
                    output["feature_maps"],
                    task=task,
                ),
                input_size,
            )

        return output

    def semantic_logits(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_with_aux(
            imgs,
            task=task,
        )["logits"]

    def forward(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.semantic_logits(imgs, task=task)
