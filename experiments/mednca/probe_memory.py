"""Peak GPU memory of one Med-NCA training step (CLAUDE.md sanity check 5).

Runs forward + cross-entropy backward under the trainer precision for each
(batch size, grad_checkpointing_every) pair and prints peak allocated memory
and step time. Example:

    python experiments/mednca/probe_memory.py --tile 448 --batch-sizes 4 8 16 \
        --every 0 2 4 8 16
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from pathseg.models.architectures.med_nca import MedNCASegmenter


def probe(args, batch_size: int, every: int | None) -> str:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    network = MedNCASegmenter(
        {"ignite": args.num_classes},
        channel_n=args.channel_n,
        steps=args.steps,
        grad_checkpointing_every=every,
    ).cuda()
    network.train()
    imgs = torch.rand(batch_size, 3, args.tile, args.tile, device="cuda")
    targets = torch.randint(
        0, args.num_classes, (batch_size, args.tile, args.tile), device="cuda"
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            logits = network(imgs)["ignite"]
            loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        return "OOM"
    finally:
        del network
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return f"{peak_gb:7.2f} GB  {elapsed:6.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=int, default=448)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--every", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16], help="0 = off"
    )
    parser.add_argument("--channel-n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = parser.parse_args()

    print(
        f"{torch.cuda.get_device_name()} | tile={args.tile} channel_n={args.channel_n} "
        f"steps={args.steps} bf16={args.bf16}"
    )
    for batch_size in args.batch_sizes:
        for every in args.every:
            result = probe(args, batch_size, every or None)
            print(f"batch={batch_size:3d} every={every or 'off':>3}  {result}", flush=True)


if __name__ == "__main__":
    main()
