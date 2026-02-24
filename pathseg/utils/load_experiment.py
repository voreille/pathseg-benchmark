from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch

from pathseg.cli import LightningCLI, LightningModule, LightningDataModule


def _overrides_to_cli_args(overrides: Dict[str, Any]) -> list[str]:
    args: list[str] = []
    for k, v in overrides.items():
        if isinstance(v, bool):
            v = str(v).lower()
        args.append(f"--{k}={v}")
    return args


def load_experiment(
    *,
    config_path: str,
    ckpt_path: Optional[str] = None,
    stage: Optional[str] = None,  # default None so nothing calls setup implicitly
    overrides: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    eval_mode: bool = True,
) -> Dict[str, Any]:
    """
    Rebuild from your custom LightningCLI (so link_arguments apply),
    then (optionally) load checkpoint weights WITHOUT running any loop.

    Key point for your design: we pass CLI-instantiated objects (network/tiler)
    back into load_from_checkpoint so the model is reconstructible.
    """
    cli_args = ["--config", config_path]
    if overrides:
        cli_args += _overrides_to_cli_args(overrides)

    cli = LightningCLI(
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
            "callbacks": [],
            "devices": 1,
            "gradient_clip_val": 1,
            "gradient_clip_algorithm": "norm",
        },
        run=False,
        args=cli_args,
    )

    trainer = cli.trainer
    dm = cli.datamodule

    # You control setup (so it won't be overridden by validate/test/predict)
    if stage is not None and hasattr(dm, "setup"):
        dm.setup(stage)

    # Build model (unweighted) from CLI first (this builds network correctly via links)
    model = cli.model

    if ckpt_path is not None:
        init_args = dict(cli.config["model"]["init_args"])

        # Ensure object-type args are the instantiated ones (NOT dicts)
        # This is the crucial part for your current design.
        init_args["network"] = model.network
        if hasattr(model, "tiler"):
            init_args["tiler"] = model.tiler

        model = type(model).load_from_checkpoint(
            ckpt_path,
            **init_args,
        )

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()

    return {"config": cli.config, "trainer": trainer, "datamodule": dm, "model": model}
