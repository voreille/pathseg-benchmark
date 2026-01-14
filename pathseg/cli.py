import logging
from datetime import datetime
from pathlib import Path

import torch
from gitignore_parser import parse_gitignore
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.training.lightning_module import LightningModule


def _has_logger(trainer) -> bool:
    lg = getattr(trainer, "logger", None)
    if lg is False or lg is None:
        return False
    if isinstance(lg, (list, tuple)) and len(lg) == 0:
        return False
    return True


def _run_dir(trainer) -> Path:
    log_dir = None
    if _has_logger(trainer):
        log_dir = getattr(trainer, "log_dir", None)

    if isinstance(log_dir, str) and log_dir:
        return Path(log_dir)

    # fallback for logger=False / debug
    return Path(trainer.default_root_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")


def _configure_checkpoint_dir(trainer) -> None:
    run_dir = _run_dir(trainer)
    run_id = None
    if trainer.logger is not None:
        if hasattr(trainer.logger, "experiment"):
            experiment = getattr(trainer.logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "id"):
                run_id = experiment.id
    ckpt_dir = (
        run_dir
        / "checkpoints"
        / (run_id if run_id is not None else datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = str(ckpt_dir)


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.suppress_errors = True  # type: ignore
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # Run outputs
        parser.add_argument(
            "--runs_dir",
            type=str,
            default="runs",
            help="Directory where run outputs (checkpoints, logs) are stored",
        )
        parser.link_arguments("runs_dir", "trainer.default_root_dir")
        parser.link_arguments("runs_dir", "trainer.logger.init_args.save_dir")

        # Misc
        parser.add_argument("--no_compile", action="store_true")

        # Keep model/data wiring explicit
        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.num_metrics", "model.init_args.num_metrics"
        )
        parser.link_arguments("data.init_args.ignore_idx", "model.init_args.ignore_idx")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.img_size",
        )

    def fit(self, model, **kwargs):
        logger = getattr(self.trainer, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None

        if experiment is not None and hasattr(experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")

            def include_fn(path: str) -> bool:
                return path.endswith(".py") or path.endswith(".yaml")

            experiment.log_code(".", include_fn=include_fn, exclude_fn=is_gitignored)

        _configure_checkpoint_dir(self.trainer)

        cfg = self.config[self.config["subcommand"]]  # type: ignore
        if not cfg.get("no_compile", False):
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)  # type: ignore


def main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
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
            "accumulate_grad_batches": 16,
            "gradient_clip_val": 1,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    main()
