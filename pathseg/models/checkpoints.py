from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_lightning_state_dict(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Read and normalize the state dict from a trusted Lightning checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise TypeError("Expected a Lightning checkpoint containing 'state_dict'.")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Lightning checkpoint 'state_dict' must be a mapping.")

    return {
        ".".join(part for part in key.split(".") if part != "_orig_mod"): value
        for key, value in state_dict.items()
    }


def load_submodule_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    source_prefix: str,
    strict: bool = True,
):
    """Select one prefixed submodule from an already-read state dict."""
    prefix = source_prefix.rstrip(".") + "."
    submodule_state_dict = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    if not submodule_state_dict:
        raise KeyError(
            f"No checkpoint keys start with {prefix!r}. "
            f"Examples: {list(state_dict)[:8]}"
        )

    return module.load_state_dict(submodule_state_dict, strict=strict)


def load_checkpoint_submodule(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    source_prefix: str,
    strict: bool = True,
):
    """Load one submodule from a trusted Lightning checkpoint."""
    state_dict = load_lightning_state_dict(checkpoint_path)
    return load_submodule_state_dict(
        module,
        state_dict,
        source_prefix=source_prefix,
        strict=strict,
    )
