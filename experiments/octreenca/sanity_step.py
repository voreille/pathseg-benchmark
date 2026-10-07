"""OctreeNCA sanity checks on one fixed real training batch (CLAUDE.md checks 4 and 5).

Builds the module and data from a config through LightningCLI, draws one
training batch (augmented, as in training) and runs the module's own
``training_step`` (bf16 autocast, the module's optimizer) on it:

- ``--iters N``: throughput (s/iter, it/s) and peak GPU memory of a training
  step (forward + loss + backward + optimizer step). Data loading and
  Lightning overhead are excluded. The first ``--warmup`` steps (compilation
  with ``compile_step``) are not timed.
- ``--overfit N``: N steps on the same batch; prints loss and pixel accuracy.
  ``overfit_batches=1`` does not fix the batch with this datamodule
  (WeightedRandomSampler + random augmentations), hence the direct loop.

Example (on a free GPU):

    CUDA_VISIBLE_DEVICES=0 python experiments/octreenca/sanity_step.py \
        -c configs/octreenca/ignite_octreenca_upstream.yaml --iters 10
    CUDA_VISIBLE_DEVICES=0 python experiments/octreenca/sanity_step.py \
        -c configs/octreenca/ignite_octreenca_upstream.yaml --overfit 1500 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import torch
from lightning.pytorch.utilities import move_data_to_device

from pathseg.cli import LightningCLI
from pathseg.datasets.lightning_data_module import LightningDataModule
from pathseg.training.lightning_module import LightningModule


def build(args):
    sys.argv = [sys.argv[0]]  # LightningCLI must not see this script's arguments
    overrides = [
        "-c", args.config,
        "--trainer.logger=false",
        "--trainer.enable_checkpointing=false",
        "--data.num_workers=0",
        f"--data.batch_size={args.batch_size}",
        f"--model.init_args.network.init_args.compile_step={str(args.compile_step).lower()}",
    ]
    if args.every is not None:
        overrides.append(
            f"--model.init_args.network.init_args.grad_checkpointing_every={args.every}"
        )
    if args.loss_name is not None:
        overrides.append(f"--model.init_args.tasks.ignite.loss_name={args.loss_name}")
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=overrides,
    )
    return cli.model, cli.datamodule


def train_step(model, optimizer, batch) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def pixel_accuracy(model, batch) -> float:
    imgs, targets, _tasks, _ids = model.unpack_batch(batch)
    target = torch.stack(model.to_per_pixel_targets_semantic(targets, model.ignore_idx))
    with torch.autocast("cuda", torch.bfloat16):
        logits = model.network(imgs / 255.0, task="ignite")["ignite"]
    valid = target != model.ignore_idx
    return float((logits.argmax(1) == target)[valid].float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--compile-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--every", type=int, default=None, help="grad_checkpointing_every")
    parser.add_argument("--loss-name", default=None, help="override tasks.ignite.loss_name")
    parser.add_argument("--iters", type=int, default=0, help="timed training steps")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--overfit", type=int, default=0, help="steps on one fixed batch")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    torch.manual_seed(args.seed)
    model, datamodule = build(args)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    device = torch.device("cuda")
    model = model.to(device).train()
    batch = move_data_to_device(batch, device)
    optimizer = model.configure_optimizers()["optimizer"]
    network = model.network
    print(
        f"batch={args.batch_size} tile={tuple(batch[0].shape[-2:])} "
        f"compile_step={network.encoder.compile_step} "
        f"every={network.encoder.grad_checkpointing_every} "
        f"loss={model.task_specs['ignite'].loss_name} "
        f"params={sum(p.numel() for p in network.parameters())}"
    )

    if args.iters:
        for _ in range(args.warmup):
            train_step(model, optimizer, batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iters):
            train_step(model, optimizer, batch)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / args.iters
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{seconds:.3f} s/iter, {1 / seconds:.2f} it/s, peak {peak:.1f} GB")

    if args.overfit:
        print(f"step 0: pixel acc {pixel_accuracy(model, batch):.3f}")
        for step in range(1, args.overfit + 1):
            loss = train_step(model, optimizer, batch)
            if step % args.log_every == 0 or step == args.overfit:
                print(f"step {step}: loss {loss:.4f} pixel acc {pixel_accuracy(model, batch):.3f}")


if __name__ == "__main__":
    main()
