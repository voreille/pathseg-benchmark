from __future__ import annotations

import importlib
from typing import Any, Mapping

import torch.nn as nn

from pathseg.models.decoders import build_semantic_decoder
from pathseg.models.sae_semantic_segmenter import SAESemanticSegmenter
from pathseg.models.semantic_segmenter import SemanticSegmenter


def build_module(
    class_path: str,
    init_args: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate an nn.Module from an import path and serializable arguments."""
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            f"Expected a fully qualified class path, got {class_path!r}."
        )

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls(**dict(init_args or {}))

    if not isinstance(instance, nn.Module):
        raise TypeError(f"{class_path!r} did not construct an nn.Module.")

    return instance


def infer_final_feature_channels(encoder: nn.Module) -> int:
    """Infer the channel count exposed by encoder.forward_feature_maps()."""
    for attribute in ("out_channels", "embed_dim", "num_features"):
        value = getattr(encoder, attribute, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            value = value[-1]
        return int(value)

    raise AttributeError(
        "Cannot infer encoder channels. Define encoder.out_channels, "
        "encoder.embed_dim, or encoder.num_features."
    )


def build_semantic_segmenter(
    *,
    encoder_class_path: str,
    encoder_init_args: dict[str, Any] | None,
    decoder_name: str,
    decoder_init_args: dict[str, Any] | None,
    num_classes_by_task: Mapping[str, int],
    upsample_logits: bool,
    interpolation_mode: str,
) -> SemanticSegmenter:
    encoder = build_module(encoder_class_path, encoder_init_args)
    decoder = build_semantic_decoder(
        decoder_name,
        in_channels=infer_final_feature_channels(encoder),
        num_classes_by_task=num_classes_by_task,
        decoder_init_args=decoder_init_args,
    )
    return SemanticSegmenter(
        encoder=encoder,
        decoder=decoder,
        upsample_logits=upsample_logits,
        interpolation_mode=interpolation_mode,
    )


def build_sae_semantic_segmenter(
    *,
    encoder_class_path: str,
    encoder_init_args: dict[str, Any] | None,
    decoder_name: str,
    decoder_init_args: dict[str, Any] | None,
    num_classes_by_task: Mapping[str, int],
    sae_class_path: str,
    sae_init_args: dict[str, Any] | None,
    upsample_logits: bool,
    interpolation_mode: str,
) -> SAESemanticSegmenter:
    encoder = build_module(encoder_class_path, encoder_init_args)
    decoder = build_semantic_decoder(
        decoder_name,
        in_channels=infer_final_feature_channels(encoder),
        num_classes_by_task=num_classes_by_task,
        decoder_init_args=decoder_init_args,
    )
    sae = build_module(sae_class_path, sae_init_args)
    return SAESemanticSegmenter(
        encoder=encoder,
        decoder=decoder,
        sae=sae,
        upsample_logits=upsample_logits,
        interpolation_mode=interpolation_mode,
    )
