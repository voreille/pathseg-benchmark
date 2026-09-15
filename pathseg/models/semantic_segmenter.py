from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureMaps = tuple[torch.Tensor, ...]
SemanticLogits = dict[str, torch.Tensor]


class SemanticSegmenter(nn.Module):
    """Compose a feature-map encoder and a semantic decoder.

    By default, decoder logits must already have the input image resolution.
    Setting ``upsample_logits=True`` explicitly authorizes interpolation from a
    smaller decoder grid to the actual input size.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        upsample_logits: bool = False,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsample_logits = bool(upsample_logits)
        self.interpolation_mode = interpolation_mode

        if self.interpolation_mode not in {"bilinear", "bicubic"}:
            raise ValueError(
                "Semantic logits support only bilinear or bicubic "
                f"interpolation, got {self.interpolation_mode!r}."
            )

    def encode(self, imgs: torch.Tensor) -> FeatureMaps:
        feature_maps = self.encoder.forward_feature_maps(imgs)

        if not isinstance(feature_maps, tuple):
            raise TypeError(
                "encoder.forward_feature_maps() must explicitly return a "
                "tuple of BxCxHxW tensors, even for one level; got "
                f"{type(feature_maps).__name__}."
            )

        if not feature_maps:
            raise ValueError("The encoder returned no feature maps.")
        if any(
            not torch.is_tensor(feature_map) or feature_map.ndim != 4
            for feature_map in feature_maps
        ):
            descriptions = [
                tuple(feature_map.shape)
                if torch.is_tensor(feature_map)
                else type(feature_map).__name__
                for feature_map in feature_maps
            ]
            raise ValueError(
                f"All feature maps must be BxCxHxW tensors; got {descriptions}."
            )
        if any(feature_map.shape[0] != imgs.shape[0] for feature_map in feature_maps):
            shapes = [tuple(feature_map.shape) for feature_map in feature_maps]
            raise ValueError(
                f"Feature-map batch sizes must match input batch {imgs.shape[0]}; "
                f"got {shapes}."
            )

        return feature_maps

    def decode(
        self,
        feature_maps: FeatureMaps,
        task: str | None = None,
    ) -> SemanticLogits:
        logits = self.decoder(feature_maps, task=task)
        if not isinstance(logits, dict):
            raise TypeError(
                "A semantic decoder must return dict[str, Tensor], "
                f"got {type(logits).__name__}."
            )
        return logits

    def ensure_input_resolution(
        self,
        logits_by_task: SemanticLogits,
        input_size: tuple[int, int],
    ) -> SemanticLogits:
        output: SemanticLogits = {}

        for task, logits in logits_by_task.items():
            if logits.ndim != 4:
                raise ValueError(
                    f"Task {task!r} logits must be BxCxHxW, "
                    f"got {tuple(logits.shape)}."
                )

            logits_size = tuple(logits.shape[-2:])
            if logits_size == input_size:
                output[task] = logits
                continue

            if not self.upsample_logits:
                raise ValueError(
                    f"Task {task!r} produced logits of size {logits_size}, "
                    f"but the input size is {input_size}. Either fix the "
                    "decoder or explicitly construct SemanticSegmenter with "
                    "upsample_logits=True."
                )

            if logits_size[0] > input_size[0] or logits_size[1] > input_size[1]:
                raise ValueError(
                    f"upsample_logits=True cannot downsample task {task!r} "
                    f"from {logits_size} to {input_size}."
                )

            output[task] = F.interpolate(
                logits,
                size=input_size,
                mode=self.interpolation_mode,
                align_corners=False,
            )

        return output

    def semantic_logits(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> SemanticLogits:
        logits = self.decode(self.encode(imgs), task=task)
        return self.ensure_input_resolution(logits, tuple(imgs.shape[-2:]))

    def forward(
        self,
        imgs: torch.Tensor,
        task: str | None = None,
    ) -> SemanticLogits:
        return self.semantic_logits(imgs, task=task)
