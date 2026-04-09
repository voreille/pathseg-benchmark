from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click
import torch
import yaml

from pathseg.utils.load_experiment import load_experiment


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _get_cfg_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "fit" in cfg and isinstance(cfg["fit"], dict):
        return cfg["fit"]
    return cfg


def import_from_string(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the loading config YAML.",
)
@click.option(
    "--ckpt-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Lightning checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where weights.pt and model.yaml will be written.",
)
@click.option(
    "--yaml-name",
    type=str,
    default="model.yaml",
    show_default=True,
    help="Name of the exported YAML file.",
)
@click.option(
    "--weights-name",
    type=str,
    default="weights.pt",
    show_default=True,
    help="Name of the exported weights file.",
)
def main(
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    yaml_name: str,
    weights_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_experiment(
        config_path=str(config_path.resolve()),
        ckpt_path=str(ckpt_path.resolve()),
        stage="fit",
        device=None,
        eval_mode=True,
    )

    cfg = bundle["config"]
    pl_model = bundle["model"]

    if not hasattr(pl_model, "network"):
        raise AttributeError("Loaded Lightning model has no attribute 'network'.")

    network = pl_model.network.cpu().eval()

    cfg_section = _get_cfg_section(cfg)

    try:
        network_cfg = cfg_section["model"]["init_args"]["network"]
    except KeyError as e:
        raise KeyError(
            "Could not find cfg['model']['init_args']['network'] in the config."
        ) from e

    if "class_path" not in network_cfg:
        raise KeyError("network config has no 'class_path'.")
    if "init_args" not in network_cfg:
        raise KeyError("network config has no 'init_args'.")

    class_path = str(network_cfg["class_path"])
    init_args = _to_builtin(network_cfg["init_args"])

    weights_path = output_dir / weights_name
    yaml_path = output_dir / yaml_name

    torch.save(network.state_dict(), weights_path)

    export_cfg = {
        "model": {
            "factory": class_path,
            "init_args": init_args,
        }
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(export_cfg, f, sort_keys=False)

    click.echo(f"[OK] saved weights -> {weights_path}")
    click.echo(f"[OK] saved model yaml -> {yaml_path}")

    click.echo("\nExample reload:")
    click.echo("```python")
    click.echo("import torch, yaml")
    click.echo("from your_module import import_from_string")
    click.echo(f"cfg = yaml.safe_load(open(r'{yaml_path}', 'r'))")
    click.echo("cls = import_from_string(cfg['model']['factory'])")
    click.echo("model = cls(**cfg['model']['init_args'])")
    click.echo(f"state_dict = torch.load(r'{weights_path}', map_location='cpu')")
    click.echo("model.load_state_dict(state_dict)")
    click.echo("model.eval()")
    click.echo("```")


if __name__ == "__main__":
    main()
