from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pathseg.sae.analysis.runner import analyze_sae


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config {path} must contain a mapping.")
    return dict(value)


def _import_object(class_path: str):
    module_name, separator, attribute = class_path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid class path: {class_path!r}.")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return dict(value)


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise TypeError(f"Missing required configuration value {path}.{key}.")
    return mapping[key]


def _shared_semantic_value(
    name: str,
    model_init_args: Mapping[str, Any],
    data_init_args: Mapping[str, Any],
) -> Any:
    if name in model_init_args and model_init_args[name] is not None:
        return deepcopy(model_init_args[name])
    if name in data_init_args and data_init_args[name] is not None:
        return deepcopy(data_init_args[name])
    raise TypeError(
        f"The semantic config defines neither model.init_args.{name} nor "
        f"data.init_args.{name}."
    )


def _resolve_semantic_config_path(
    value: str | Path,
    *,
    sae_config_path: Path,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Match the SAE training launcher: relative paths are interpreted from
        # the process working directory, not from the YAML file's directory.
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Semantic config referenced by "
            f"{sae_config_path} was not found: {path}"
        )
    return path


def _compose_analysis_config(
    sae_config: Mapping[str, Any],
    semantic_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose the model and data sections needed for SAE analysis.

    This mirrors the training launcher, except initialization checkpoints are
    deliberately omitted: the completed SAE-run checkpoint supplies all
    encoder, semantic-decoder, and SAE parameters.
    """

    sae = _as_mapping(
        _require(sae_config, "sae", "SAE config"),
        "SAE config.sae",
    )
    semantic_model = _as_mapping(
        _require(semantic_config, "model", "semantic config"),
        "semantic config.model",
    )
    semantic_model_args = _as_mapping(
        _require(semantic_model, "init_args", "semantic config.model"),
        "semantic config.model.init_args",
    )
    semantic_data = deepcopy(
        _as_mapping(
            _require(semantic_config, "data", "semantic config"),
            "semantic config.data",
        )
    )
    semantic_data_args = _as_mapping(
        _require(semantic_data, "init_args", "semantic config.data"),
        "semantic config.data.init_args",
    )

    model_init_args: dict[str, Any] = {
        "encoder_class_path": _require(
            semantic_model_args,
            "encoder_class_path",
            "semantic config.model.init_args",
        ),
        "decoder_name": _require(
            semantic_model_args,
            "decoder_name",
            "semantic config.model.init_args",
        ),
        "tasks": deepcopy(
            _require(
                semantic_model_args,
                "tasks",
                "semantic config.model.init_args",
            )
        ),
        "eval_task_names": deepcopy(
            _require(
                semantic_model_args,
                "eval_task_names",
                "semantic config.model.init_args",
            )
        ),
        "ignore_idx": _shared_semantic_value(
            "ignore_idx",
            semantic_model_args,
            semantic_data_args,
        ),
        "img_size": _shared_semantic_value(
            "img_size",
            semantic_model_args,
            semantic_data_args,
        ),
        "sae_class_path": _require(sae, "class_path", "SAE config.sae"),
        "sae_init_args": deepcopy(sae.get("init_args") or {}),
        "lr": float(sae.get("lr", 3e-4)),
        "weight_decay": float(sae.get("weight_decay", 0.0)),
        "poly_lr_decay_power": float(sae.get("poly_lr_decay_power", 0.9)),
        "normalize_decoder": bool(sae.get("normalize_decoder", True)),
    }

    inherited_optional_args = (
        "encoder_init_args",
        "decoder_init_args",
        "tiler_name",
        "tiler_init_args",
        "upsample_logits",
        "interpolation_mode",
    )
    for name in inherited_optional_args:
        if name in semantic_model_args:
            model_init_args[name] = deepcopy(semantic_model_args[name])

    model_config = {
        "class_path": sae_config.get(
            "training_class_path",
            "pathseg.training.sae_semantic.TopKSAESemanticTraining",
        ),
        "init_args": model_init_args,
    }
    return model_config, semantic_data


def _resolve_run_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve either a minimal SAE config or an expanded Lightning config."""

    config = _load_yaml(config_path)
    model_config = config.get("model")
    data_config = config.get("data")

    if isinstance(model_config, Mapping) and isinstance(data_config, Mapping):
        return dict(model_config), dict(data_config)

    semantic_run = _as_mapping(
        _require(config, "semantic_run", "SAE config"),
        "SAE config.semantic_run",
    )
    semantic_config_path = _resolve_semantic_config_path(
        _require(semantic_run, "config_path", "SAE config.semantic_run"),
        sae_config_path=config_path,
    )
    return _compose_analysis_config(
        config,
        _load_yaml(semantic_config_path),
    )


def _instantiate_data_module(data_config: Mapping[str, Any]):
    """Instantiate a LightningCLI subclass config without constructing a Trainer."""

    from lightning.pytorch import LightningDataModule
    from lightning.pytorch.cli import LightningArgumentParser

    parser = LightningArgumentParser()
    parser.add_subclass_arguments(
        LightningDataModule,
        "data",
        required=True,
    )
    parsed = parser.parse_object({"data": dict(data_config)})
    return parser.instantiate_classes(parsed).data


def build_analysis_objects(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
):
    """Build the current SAE module and load its learned network parameters."""

    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SAE config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    model_config, data_config = _resolve_run_config(config_path)
    training_class = _import_object(str(model_config["class_path"]))
    model_init_args = deepcopy(model_config.get("init_args") or {})
    if not isinstance(model_init_args, dict):
        raise TypeError("model.init_args must be a mapping.")

    # A full SAE checkpoint contains encoder, semantic decoder, and SAE weights.
    # Avoid opening initialization checkpoints referenced by the new config;
    # they would be overwritten immediately by the full state dict anyway.
    for name in (
        "semantic_init_checkpoint_path",
        "sae_init_checkpoint_path",
    ):
        if name in model_init_args:
            model_init_args[name] = None

    module = training_class(**model_init_args)

    from pathseg.models.checkpoints import load_checkpoint_submodule

    # Keep the current config authoritative for task specs, losses, metrics,
    # and other Lightning state.  Restore only the learned semantic network:
    # encoder, task decoder, and SAE.
    load_checkpoint_submodule(
        module.network,
        str(checkpoint_path),
        source_prefix="network",
        strict=True,
    )
    module.freeze()
    data_module = _instantiate_data_module(data_config)
    return module, data_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a trained semantic TopK SAE on all configured validation "
            "datasets."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help=(
            "Minimal SAE config with semantic_run/sae sections, or an already "
            "resolved config with model/data sections. The config supplies "
            "constructor arguments; the checkpoint supplies learned weights."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Full TopKSAESemanticTraining checkpoint, normally last.ckpt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for analysis.pt, CSV tables, JSON, and JSONL outputs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device; defaults to CUDA when available.",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=("32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"),
    )
    parser.add_argument("--top-latents-per-class", type=int, default=10)
    parser.add_argument("--global-top-latents", type=int, default=32)
    parser.add_argument("--max-selected-latents", type=int, default=128)
    parser.add_argument("--top-examples-per-latent", type=int, default=16)
    parser.add_argument("--max-examples-per-image", type=int, default=2)
    parser.add_argument(
        "--max-batches-per-loader",
        type=int,
        default=None,
        help="Limit each validation loader for a quick smoke test.",
    )
    parser.add_argument(
        "--no-window-inputs",
        action="store_true",
        help="Analyze dataset samples directly instead of validation crops.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    module, data_module = build_analysis_objects(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    analyze_sae(
        lightning_module=module,
        data_module=data_module,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        top_latents_per_class=args.top_latents_per_class,
        global_top_latents=args.global_top_latents,
        max_selected_latents=args.max_selected_latents,
        top_examples_per_latent=args.top_examples_per_latent,
        max_examples_per_image=args.max_examples_per_image,
        max_batches_per_loader=args.max_batches_per_loader,
        window_inputs=not args.no_window_inputs,
    )


if __name__ == "__main__":
    main()
