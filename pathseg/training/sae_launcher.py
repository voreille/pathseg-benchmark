from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class SAEConfigError(ValueError):
    """Raised when a composed SAE run configuration is incomplete."""


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SAEConfigError(f"{path} must be a mapping.")
    return dict(value)


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise SAEConfigError(f"Missing required configuration value {path}.{key}.")
    return mapping[key]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return _as_mapping(value, str(path))


def _resolve_path(value: str | Path, cwd: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _shared_model_value(
    name: str,
    semantic_model_args: Mapping[str, Any],
    semantic_data_args: Mapping[str, Any],
) -> Any:
    """Resolve values that may be linked from data into the semantic model."""
    if name in semantic_model_args and semantic_model_args[name] is not None:
        return deepcopy(semantic_model_args[name])
    if name in semantic_data_args and semantic_data_args[name] is not None:
        return deepcopy(semantic_data_args[name])
    raise SAEConfigError(
        f"The semantic run defines neither model.init_args.{name} nor "
        f"data.init_args.{name}."
    )


def compose_sae_cli_config(
    sae_config: Mapping[str, Any],
    semantic_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a minimal SAE config into a regular LightningCLI fit config.

    The semantic run is the source of truth for the encoder, semantic decoder,
    task definitions, tiler, output-resolution policy, and datamodule. The SAE
    config owns only the SAE, its optimizer settings, and the new Trainer run.
    """
    sae_config = _as_mapping(sae_config, "sae config")
    semantic_config = _as_mapping(semantic_config, "semantic config")

    semantic_run = _as_mapping(
        _require(sae_config, "semantic_run", "sae config"),
        "semantic_run",
    )
    sae = _as_mapping(_require(sae_config, "sae", "sae config"), "sae")
    trainer = _as_mapping(sae_config.get("trainer", {}), "trainer")

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

    model_args: dict[str, Any] = {
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
        "ignore_idx": _shared_model_value(
            "ignore_idx",
            semantic_model_args,
            semantic_data_args,
        ),
        "img_size": _shared_model_value(
            "img_size",
            semantic_model_args,
            semantic_data_args,
        ),
        "sae_class_path": _require(sae, "class_path", "sae"),
        "sae_init_args": deepcopy(sae.get("init_args") or {}),
        "semantic_init_checkpoint_path": _require(
            semantic_run,
            "checkpoint_path",
            "semantic_run",
        ),
        "semantic_encoder_prefix": semantic_run.get(
            "encoder_prefix",
            "network.encoder",
        ),
        "semantic_decoder_prefix": semantic_run.get(
            "decoder_prefix",
            "network.decoder",
        ),
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
            model_args[name] = deepcopy(semantic_model_args[name])

    if sae.get("checkpoint_path") is not None:
        model_args["sae_init_checkpoint_path"] = sae["checkpoint_path"]
        model_args["sae_prefix"] = sae.get(
            "checkpoint_prefix",
            "network.sae",
        )

    resolved: dict[str, Any] = {
        "seed_everything": deepcopy(
            sae_config.get(
                "seed_everything",
                semantic_config.get("seed_everything", 0),
            )
        ),
        "trainer": deepcopy(trainer),
        "model": {
            "class_path": sae_config.get(
                "training_class_path",
                "pathseg.training.sae_semantic.TopKSAESemanticTraining",
            ),
            "init_args": model_args,
        },
        "data": semantic_data,
    }

    # This is an SAE-run resume checkpoint, distinct from both semantic and
    # optional SAE initialization checkpoints.
    if sae_config.get("ckpt_path") is not None:
        resolved["ckpt_path"] = sae_config["ckpt_path"]
    if "weights_only" in sae_config:
        resolved["weights_only"] = sae_config["weights_only"]

    return resolved


def load_composed_sae_config(
    sae_config_path: str | Path,
    *,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Read the two configs, resolve checkpoint paths, and compose the run."""
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    sae_config_path = _resolve_path(sae_config_path, working_directory)
    if not sae_config_path.is_file():
        raise FileNotFoundError(f"SAE config not found: {sae_config_path}")

    sae_config = _load_yaml(sae_config_path)
    semantic_run = _as_mapping(
        _require(sae_config, "semantic_run", "sae config"),
        "semantic_run",
    )

    semantic_config_path = _resolve_path(
        _require(semantic_run, "config_path", "semantic_run"),
        working_directory,
    )
    semantic_checkpoint_path = _resolve_path(
        _require(semantic_run, "checkpoint_path", "semantic_run"),
        working_directory,
    )
    if not semantic_config_path.is_file():
        raise FileNotFoundError(
            f"Semantic run config not found: {semantic_config_path}"
        )
    if not semantic_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Semantic checkpoint not found: {semantic_checkpoint_path}"
        )

    semantic_run["config_path"] = str(semantic_config_path)
    semantic_run["checkpoint_path"] = str(semantic_checkpoint_path)
    sae_config["semantic_run"] = semantic_run

    sae = _as_mapping(_require(sae_config, "sae", "sae config"), "sae")
    if sae.get("checkpoint_path") is not None:
        sae_checkpoint_path = _resolve_path(
            sae["checkpoint_path"],
            working_directory,
        )
        if not sae_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAE initialization checkpoint not found: {sae_checkpoint_path}"
            )
        sae["checkpoint_path"] = str(sae_checkpoint_path)
        sae_config["sae"] = sae

    if sae_config.get("ckpt_path") is not None:
        resume_checkpoint_path = _resolve_path(
            sae_config["ckpt_path"],
            working_directory,
        )
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAE resume checkpoint not found: {resume_checkpoint_path}"
            )
        sae_config["ckpt_path"] = str(resume_checkpoint_path)

    return compose_sae_cli_config(
        sae_config,
        _load_yaml(semantic_config_path),
    )


def launch_sae(
    config_path: str | Path,
    lightning_args: Sequence[str] = (),
) -> None:
    """Launch a composed SAE fit using an ordinary LightningCLI config."""
    resolved_config = load_composed_sae_config(config_path)

    # Import Lightning only for execution. This keeps config composition easy
    # to test without constructing a Trainer or importing torch.
    from lightning.pytorch import LightningDataModule, LightningModule
    from lightning.pytorch.cli import LightningCLI

    with tempfile.TemporaryDirectory(prefix="pathseg-sae-") as directory:
        resolved_path = Path(directory) / "resolved_sae_config.yaml"
        with resolved_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(resolved_config, stream, sort_keys=False)

        LightningCLI(
            LightningModule,
            LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            auto_configure_optimizers=False,
            args=[
                "fit",
                "--config",
                str(resolved_path),
                *lightning_args,
            ],
            save_config_kwargs={
                "overwrite": True,
            },
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an SAE from a saved semantic LightningCLI run without "
            "duplicating its model or data configuration."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Minimal SAE YAML configuration.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print the generated LightningCLI config and exit.",
    )
    parsed, lightning_args = parser.parse_known_args(argv)
    if lightning_args[:1] == ["--"]:
        lightning_args = lightning_args[1:]

    if parsed.print_resolved_config:
        resolved = load_composed_sae_config(parsed.config)
        yaml.safe_dump(resolved, sys.stdout, sort_keys=False)
        return

    launch_sae(parsed.config, lightning_args)


if __name__ == "__main__":
    main()
