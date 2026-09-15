from typing import Any, Mapping

import torch.nn as nn

from pathseg.models.decoders.linear import LinearSemanticDecoder


def build_semantic_decoder(
    decoder_name: str,
    *,
    in_channels: int,
    num_classes_by_task: Mapping[str, int],
    decoder_init_args: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a decoder whose output is a mapping from task name to logits."""
    init_args = dict(decoder_init_args or {})

    if decoder_name == "linear":
        return LinearSemanticDecoder(
            in_channels=in_channels,
            num_classes_by_task=num_classes_by_task,
            **init_args,
        )

    raise ValueError(f"Unknown semantic decoder: {decoder_name!r}")


__all__ = ["LinearSemanticDecoder", "build_semantic_decoder"]

