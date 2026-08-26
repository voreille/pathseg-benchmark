from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from pathseg.models.linear_decoder import LinearDecoder
from pathseg.training.tiler import GridPadTiler, Tiler


def build_two_tasks_decoder(
    decoder_name: str,
    *,
    num_classes_a: int,
    num_classes_b: int,
    img_size: tuple[int, int],
    decoder_init_args: dict[str, Any] | None = None,
) -> nn.Module:
    init_args = dict(decoder_init_args or {})

    if decoder_name == "two_tasks_linear_decoder":
        from pathseg.models.multitask_decoder import TwoHeadsLinearDecoder

        return TwoHeadsLinearDecoder(
            num_classes_a=num_classes_a,
            num_classes_b=num_classes_b,
            img_size=img_size,
            **init_args,
        )

    raise ValueError(
        f"Unknown decoder_name={decoder_name!r}. Available decoders: linear."
    )


def build_decoder(
    decoder_name: str,
    *,
    num_classes: int,
    img_size: tuple[int, int],
    decoder_init_args: dict[str, Any] | None = None,
) -> nn.Module:
    init_args = dict(decoder_init_args or {})

    if decoder_name == "linear_decoder":
        return LinearDecoder(
            num_classes=num_classes,
            img_size=img_size,
            **init_args,
        )
    raise ValueError(
        f"Unknown decoder_name={decoder_name!r}. Available decoders: linear."
    )


def build_tiler(
    tiler_name: Optional[str],
    *,
    tiler_init_args: dict[str, Any] | None = None,
) -> Optional[Tiler]:
    if tiler_name is None or tiler_name == "none":
        return None

    init_args = dict(tiler_init_args or {})

    if tiler_name == "grid_pad_tiler":
        return GridPadTiler(**init_args)

    raise ValueError(
        f"Unknown tiler_name={tiler_name!r}. Available tilers: grid_pad, none."
    )
