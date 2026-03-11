from pathlib import Path

import torch

from pathseg.utils.load_experiment import load_experiment

import click


def export_torchscript(
    config_path: str,
    ckpt_path: str,
    out_path: str,
    device="cuda",
):
    exp = load_experiment(
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
        eval_mode=True,
    )
    net = exp["model"].network.to(device).eval()
    dm = exp["datamodule"]
    dm.setup(stage="fit")
    example = next(iter(dm.train_dataloader()))[0].to(device)  # e
    example = example / 255.0

    # Prefer scripting; fall back to tracing if needed
    try:
        scripted = torch.jit.script(net)
    except Exception:
        with torch.inference_mode():
            scripted = torch.jit.trace(net, example, strict=False)
            scripted = torch.jit.freeze(scripted)

    scripted.save(out_path)  # e.g. "network_scripted.pt"


@click.command()
@click.option("--ckpt-path", default="World", help="Path to the checkpoint.")
@click.option("--config-path", default="World", help="Path to the configuration file.")
@click.option(
    "--out-path",
    default="network_scripted.pt",
    help="Path to save the TorchScript model.",
)
@click.option(
    "--device",
    default="cuda",
    help="Device to use for loading the model and example (e.g., 'cuda' or 'cpu').",
)
def main(ckpt_path, config_path, out_path, device):
    """Simple CLI program to greet someone"""
    export_torchscript(
        config_path=config_path,
        ckpt_path=ckpt_path,
        out_path=out_path,
        device=device,
    )


if __name__ == "__main__":
    main()
