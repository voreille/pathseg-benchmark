from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from lightning.pytorch import LightningModule


class InferenceModel(nn.Module):
    """Inference-only view of a trained segmentation network.

    The training-time `task` argument is intentionally not exposed.
    All semantic heads are returned.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network(x)


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


def assert_outputs_close(
    eager_output,
    exported_output,
) -> None:
    if isinstance(eager_output, Mapping):
        if set(eager_output) != set(exported_output):
            raise RuntimeError(
                "Exported model returned different output heads: "
                f"eager={sorted(eager_output)}, "
                f"exported={sorted(exported_output)}."
            )

        for task in eager_output:
            torch.testing.assert_close(
                exported_output[task],
                eager_output[task],
                rtol=1e-4,
                atol=1e-5,
            )
        return

    torch.testing.assert_close(
        exported_output,
        eager_output,
        rtol=1e-4,
        atol=1e-5,
    )


def export_model(
    *,
    config_path: Path,
    ckpt_path: Path,
    out_path: Path,
    device: str = "cuda",
    max_batch_size: int = 32,
) -> None:
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1.")

    training_module = load_training_module(
        config_path=config_path,
        ckpt_path=ckpt_path,
    )

    device_obj = torch.device(device)

    network = training_module.network.to(device_obj).eval()
    network.requires_grad_(False)

    model = InferenceModel(network).eval()

    height, width = training_module.img_size

    # Batch 2 avoids specializing the example itself to the special batch=1 case.
    example_batch_size = min(2, max_batch_size)
    example = torch.rand(
        example_batch_size,
        3,
        height,
        width,
        dtype=torch.float32,
        device=device_obj,
    )

    dynamic_shapes = None
    if max_batch_size > 1:
        batch = torch.export.Dim(
            "batch",
            min=1,
            max=max_batch_size,
        )
        dynamic_shapes = {
            "x": {0: batch},
        }

    with torch.inference_mode():
        exported = torch.export.export(
            model,
            args=(example,),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

        # Sanity check before serialization.
        eager_output = model(example)
        exported_output = exported.module()(example)
        assert_outputs_close(
            eager_output,
            exported_output,
        )

    torch.export.save(
        exported,
        out_path,
    )


@click.command()
@click.option(
    "--ckpt-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Path to the Lightning checkpoint.",
)
@click.option(
    "--config-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Path to the experiment configuration.",
)
@click.option(
    "--out-path",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    default=Path("model.pt2"),
    show_default=True,
    help="Path to save the exported PyTorch model.",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    help="Device used during export, e.g. 'cuda' or 'cpu'.",
)
@click.option(
    "--max-batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Maximum supported inference batch size.",
)
def main(
    ckpt_path: Path,
    config_path: Path,
    out_path: Path,
    device: str,
    max_batch_size: int,
) -> None:
    """Export a trained network using torch.export."""
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_model(
        config_path=config_path,
        ckpt_path=ckpt_path,
        out_path=out_path,
        device=device,
        max_batch_size=max_batch_size,
    )

    click.echo(f"Saved exported model to {out_path}")


if __name__ == "__main__":
    main()
