"""Which Med-NCA level limits the segmentation? Level ablations at evaluation.

Runs the unchanged validation pipeline (tiler, stitching, IoU/F1) on a trained
checkpoint several times. Each time, ``MedNCAEncoder.forward_state`` is swapped for
a different composition of the encoder stages. Nothing is retrained, and the
checkpoint and the model code are unchanged.

Modes:
- ``full``: the model as trained (coarse → upscale → fine). Reference.
- ``coarse_only``: the coarse level's output channels, nearest-upscaled to the
  tile. What the coarse level alone predicts, at 4× lower resolution.
- ``fine_only``: the fine level from a fresh seed on the full image, with no
  coarse state. How much the fine level depends on the coarse level's context.
  Out of distribution for the fine level, so only the size of the drop matters.
- ``coarse_antialias``: like ``full``, but the coarse input is downscaled with
  antialiasing. Also out of distribution: a gain without retraining points at
  aliasing; no gain is inconclusive.

How to read the result, per class:
- ``coarse_only`` ≈ ``full``, both low: the fine level adds little, and the limit
  is what the coarse level can see or represent (context, aliasing, capacity).
- ``coarse_only`` clearly above ``full``: the fine level degrades the coarse
  prediction.
- ``fine_only`` ≈ ``full``: the coarse level contributes little context.

Example (sandbox: ``--num-workers 0``; one validation pass per mode):

    CUDA_VISIBLE_DEVICES=1 python experiments/mednca/diagnose_levels.py \\
        -c configs/mednca/ignite_mednca_upstream.yaml \\
        --ckpt "runs/checkpoints/7ro3wqzo/epoch=26-step=40000.migrated.ckpt"
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from pathseg.cli import LightningCLI
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.training.lightning_module import LightningModule

MODES = ("full", "coarse_only", "fine_only", "coarse_antialias")


def coarse_only(self, imgs: torch.Tensor) -> torch.Tensor:
    small = self.downscale(imgs)
    state = self.run_level("coarse", self.init_state(small), small)
    return self.upscale_state(state, imgs.shape[-2:])


def fine_only(self, imgs: torch.Tensor) -> torch.Tensor:
    return self.run_level("fine", self.init_state(imgs), imgs)


def coarse_antialias(self, imgs: torch.Tensor) -> torch.Tensor:
    size = (imgs.shape[-2] // self.scale_factor, imgs.shape[-1] // self.scale_factor)
    small = F.interpolate(
        imgs, size=size, mode="bilinear", align_corners=False, antialias=True
    )
    state = self.run_level("coarse", self.init_state(small), small)
    state = self.upscale_state(state, imgs.shape[-2:])
    return self.run_level("fine", state, imgs)


OVERRIDES = {
    "coarse_only": coarse_only,
    "fine_only": fine_only,
    "coarse_antialias": coarse_antialias,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Write results as JSON.")
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]  # LightningCLI must not see this script's arguments
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=[
            "-c", args.config,
            "--trainer.logger=false",
            "--trainer.enable_checkpointing=false",
            f"--data.num_workers={args.num_workers}",
        ],
    )
    model, datamodule, trainer = cli.model, cli.datamodule, cli.trainer
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    encoder = model.network.encoder
    original = encoder.forward_state

    results: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        override = OVERRIDES.get(mode)
        encoder.forward_state = (
            original if override is None else types.MethodType(override, encoder)
        )
        torch.manual_seed(args.seed)
        metrics = trainer.validate(model, datamodule=datamodule, verbose=False)[0]
        results[mode] = {
            k: float(v) for k, v in metrics.items() if "_iou_" in k or k.endswith("miou")
        }
        print(f"{mode}: mIoU {next(v for k, v in results[mode].items() if k.endswith('miou')):.4f}")

    keys = sorted(
        next(iter(results.values())),
        key=lambda k: (not k.endswith("miou"), int(k.rsplit("_", 1)[-1]) if k[-1].isdigit() else -1),
    )
    print("\n" + " | ".join(["metric", *args.modes]))
    for key in keys:
        print(" | ".join([key, *(f"{results[m][key]:.3f}" for m in args.modes)]))
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
