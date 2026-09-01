from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn


def load_checkpoint_submodule(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    source_prefix: str,
    strict: bool = True,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,  # Lightning checkpoint; trusted file
    )

    state_dict = checkpoint["state_dict"]
    prefix = source_prefix.rstrip(".") + "."

    submodule_state_dict = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    if not submodule_state_dict:
        raise KeyError(
            f"No keys beginning with {prefix!r}. Example keys: {list(state_dict)[:10]}"
        )

    return module.load_state_dict(submodule_state_dict, strict=strict)
