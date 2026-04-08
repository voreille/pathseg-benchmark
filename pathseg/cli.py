from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from gitignore_parser import parse_gitignore
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.training.lightning_module import LightningModule


def _has_logger(trainer) -> bool:
    logger = getattr(trainer, "logger", None)
    if logger is None or logger is False:
        return False
    if isinstance(logger, (list, tuple)) and len(logger) == 0:
        return False
    return True


def _get_logger_experiment(trainer) -> Any | None:
    logger = getattr(trainer, "logger", None)
    if logger is None or logger is False:
        return None
    return getattr(logger, "experiment", None)


def _get_run_id(trainer) -> str | None:
    experiment = _get_logger_experiment(trainer)
    if experiment is None:
        return None

    run_id = getattr(experiment, "id", None)
    if isinstance(run_id, str) and run_id:
        return run_id
    return None


def _get_checkpoint_dir(trainer) -> Path:
    cached = getattr(trainer, "_pathseg_ckpt_dir", None)
    if cached is not None:
        return Path(cached)

    root = Path(trainer.default_root_dir) / "checkpoints"
    run_id = _get_run_id(trainer) or datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = root / run_id
    trainer._pathseg_ckpt_dir = str(ckpt_dir)
    return ckpt_dir


def _configure_checkpoint_callbacks(trainer) -> None:
    ckpt_dir = _get_checkpoint_dir(trainer)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = str(ckpt_dir)


def _get_single_ckpt(run_dir: Path) -> Path:
    ckpts = list(run_dir.glob("*.ckpt"))

    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoint found in {run_dir}")

    if len(ckpts) > 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {run_dir}, found {len(ckpts)}:\n"
            + "\n".join(str(p) for p in ckpts)
        )

    return ckpts[0]


def _find_latest_valid_run(checkpoints_root: Path) -> tuple[str, Path, Path]:
    if not checkpoints_root.exists():
        raise RuntimeError(f"Checkpoint root does not exist: {checkpoints_root}")

    candidates: list[tuple[float, str, Path, Path]] = []

    for run_dir in checkpoints_root.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue

        try:
            ckpt_path = _get_single_ckpt(run_dir)
        except RuntimeError:
            continue

        mtime = ckpt_path.stat().st_mtime
        candidates.append((mtime, run_dir.name, run_dir, ckpt_path))

    if not candidates:
        raise RuntimeError(
            f"No valid run found in {checkpoints_root}. "
            "A valid run must contain config.yaml and exactly one .ckpt."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, run_id, run_dir, ckpt_path = candidates[0]
    return run_id, run_dir, ckpt_path


def _log_code_to_wandb(trainer) -> None:
    experiment = _get_logger_experiment(trainer)
    if experiment is None or not hasattr(experiment, "log_code"):
        return

    is_gitignored = parse_gitignore(".gitignore")

    def include_fn(path: str) -> bool:
        return path.endswith(".py") or path.endswith(".yaml")

    experiment.log_code(".", include_fn=include_fn, exclude_fn=is_gitignored)


def _extract_cli_value(argv: list[str], flag: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _remove_flag_and_value(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                skip_next = True
            continue
        if arg.startswith(flag + "="):
            continue
        out.append(arg)
    return out


def _inject_resume_args(argv: list[str]) -> list[str]:
    """
    Resolve --resume_latest / --resume_run_id BEFORE LightningCLI validation.

    This function injects:
      --config <saved_config>
      --ckpt_path <saved_ckpt>
      --trainer.logger.init_args.id <run_id>
      --trainer.logger.init_args.resume must
    """
    if len(argv) < 2:
        return argv

    if "fit" not in argv:
        return argv

    resume_latest = _has_flag(argv, "--resume_latest")
    resume_run_id = _extract_cli_value(argv, "--resume_run_id")

    if not resume_latest and resume_run_id is None:
        return argv

    if resume_latest and resume_run_id is not None:
        raise RuntimeError("Use either --resume_latest or --resume_run_id, not both.")

    runs_dir = _extract_cli_value(argv, "--runs_dir") or "runs"
    checkpoints_root = Path(runs_dir) / "checkpoints"

    if resume_latest:
        run_id, run_dir, ckpt_path = _find_latest_valid_run(checkpoints_root)
    else:
        run_id = resume_run_id
        assert run_id is not None
        run_dir = checkpoints_root / run_id
        if not run_dir.exists():
            raise RuntimeError(f"Run directory does not exist: {run_dir}")
        ckpt_path = _get_single_ckpt(run_dir)

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"Missing config.yaml in {run_dir}")

    cleaned = list(argv)
    cleaned = _remove_flag_and_value(cleaned, "--resume_latest")
    cleaned = _remove_flag_and_value(cleaned, "--resume_run_id")

    # Only inject if user did not already specify them
    if _extract_cli_value(cleaned, "--config") is None and _extract_cli_value(cleaned, "-c") is None:
        cleaned.extend(["--config", str(config_path)])

    if _extract_cli_value(cleaned, "--ckpt_path") is None:
        cleaned.extend(["--ckpt_path", str(ckpt_path)])

    cleaned.extend(["--trainer.logger.init_args.id", run_id])
    cleaned.extend(["--trainer.logger.init_args.resume", "must"])

    logging.info("Resolved resume: run_id=%s", run_id)
    logging.info("Resolved config: %s", config_path)
    logging.info("Resolved ckpt: %s", ckpt_path)

    return cleaned


class SaveConfigToCheckpointDir(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage: str) -> None:
        ckpt_dir = _get_checkpoint_dir(trainer)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        config_path = ckpt_dir / self.config_filename
        config_str = self.parser.dump(self.config, skip_none=False)

        with config_path.open("w", encoding="utf-8") as f:
            f.write(config_str)


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]

        sys.argv = _inject_resume_args(sys.argv)

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            "--runs_dir",
            type=str,
            default="runs",
            help="Root directory for logs and checkpoints.",
        )
        parser.link_arguments("runs_dir", "trainer.default_root_dir")
        parser.link_arguments("runs_dir", "trainer.logger.init_args.save_dir")

        parser.add_argument("--no_compile", action="store_true")

        parser.add_argument(
            "--resume_latest",
            action="store_true",
            help="Resume the latest valid run found in <runs_dir>/checkpoints.",
        )
        parser.add_argument(
            "--resume_run_id",
            type=str,
            default=None,
            help="Resume a specific run id from <runs_dir>/checkpoints/<run_id>.",
        )

        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.num_metrics",
            "model.init_args.num_metrics",
        )
        parser.link_arguments(
            "data.init_args.ignore_idx",
            "model.init_args.ignore_idx",
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.img_size",
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.img_size",
        )

    def fit(self, model, **kwargs) -> None:
        cfg = self.config[self.config["subcommand"]]  # type: ignore[index]

        _configure_checkpoint_callbacks(self.trainer)
        _log_code_to_wandb(self.trainer)

        if not cfg.get("no_compile", False):
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)  # type: ignore[misc]


def main() -> None:
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=SaveConfigToCheckpointDir,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": False,
            "save_to_log_dir": False,
        },
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=2),
                ModelCheckpoint(),
            ],
            "devices": 1,
            "gradient_clip_val": 1,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    main()