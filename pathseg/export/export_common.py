from __future__ import annotations

import importlib
from pathlib import Path

import yaml
from lightning.pytorch import LightningModule


def load_training_module(
    *,
    config_path: Path,
    ckpt_path: Path,
) -> LightningModule:
    """Load the concrete LightningModule specified by the experiment config."""
    with config_path.open() as file:
        config = yaml.safe_load(file)

    try:
        class_path = config["model"]["class_path"]
    except KeyError as error:
        raise ValueError(f"{config_path} does not define model.class_path.") from error

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    if not isinstance(model_cls, type) or not issubclass(
        model_cls,
        LightningModule,
    ):
        raise TypeError(f"{class_path!r} is not a LightningModule subclass.")

    return model_cls.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
    )
