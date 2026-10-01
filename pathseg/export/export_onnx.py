from __future__ import annotations

from pathlib import Path

import click
import onnxruntime as ort
import torch
from torch import nn

# Import this after moving the shared function to a common module.
from pathseg.export.export_common import load_training_module


class ONNXInferenceModel(nn.Module):
    """Inference wrapper with explicitly ordered tensor outputs."""

    def __init__(
        self,
        network: nn.Module,
        output_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.network = network
        self.output_names = output_names

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.network(x)
        return tuple(outputs[name] for name in self.output_names)


def assert_onnx_outputs_close(
    *,
    model: ONNXInferenceModel,
    example: torch.Tensor,
    out_path: Path,
    output_names: tuple[str, ...],
) -> None:
    with torch.inference_mode():
        eager_outputs = model(example)

    session = ort.InferenceSession(
        str(out_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_outputs = session.run(
        list(output_names),
        {
            input_name: example.detach().cpu().numpy(),
        },
    )

    if len(onnx_outputs) != len(eager_outputs):
        raise RuntimeError(
            f"ONNX returned {len(onnx_outputs)} outputs, expected {len(eager_outputs)}."
        )

    for name, eager_output, onnx_output in zip(
        output_names,
        eager_outputs,
        onnx_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            torch.from_numpy(onnx_output),
            eager_output.detach().cpu(),
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message, name=name: (
                f"ONNX output {name!r} differs from eager output:\n{message}"
            ),
        )


def export_model(
    *,
    config_path: Path,
    ckpt_path: Path,
    out_path: Path,
    device: str = "cpu",
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

    output_names = tuple(network.num_classes_by_task)

    if not output_names:
        raise ValueError("The network does not expose any semantic heads.")

    model = ONNXInferenceModel(
        network=network,
        output_names=output_names,
    ).eval()

    height, width = training_module.img_size

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
        torch.onnx.export(
            model,
            args=(example,),
            f=str(out_path),
            input_names=["input"],
            output_names=list(output_names),
            opset_version=18,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )

    assert_onnx_outputs_close(
        model=model,
        example=example,
        out_path=out_path,
        output_names=output_names,
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
)
@click.option(
    "--config-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    default=Path("model.onnx"),
    show_default=True,
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device used while capturing the model.",
)
@click.option(
    "--max-batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
)
def main(
    ckpt_path: Path,
    config_path: Path,
    out_path: Path,
    device: str,
    max_batch_size: int,
) -> None:
    """Export a trained segmentation network to ONNX."""
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_model(
        config_path=config_path,
        ckpt_path=ckpt_path,
        out_path=out_path,
        device=device,
        max_batch_size=max_batch_size,
    )

    click.echo(f"Saved ONNX model to {out_path}")


if __name__ == "__main__":
    main()
