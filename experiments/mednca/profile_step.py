"""Profile one Med-NCA training step (see "Current focus: training throughput" in CLAUDE.md).

A step is what ``SemanticTraining`` runs: forward of the network on ``imgs / 255``
under autocast, ``cross_entropy_dice`` on the task logits, backward, and AdamW.
Data loading and Lightning overhead are excluded. The script reports:

1. Throughput: s/iter, it/s, analytic TFLOP/iter and achieved TFLOP/s, peak memory.
2. Device time per kernel category (conv, GEMM, copy/cast, layout transform, ...).
3. Top CUDA kernels and top aten ops by self device time.

Example (on a free GPU):

    CUDA_VISIBLE_DEVICES=1 python experiments/mednca/profile_step.py \
        --tile 448 --batch-size 16 --every 8 --trace runs/mednca_step.json
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from pathseg.models.architectures.med_nca import MedNCASegmenter
from pathseg.training.histo_loss import CrossEntropyDiceLoss

# First matching pattern wins, so specific names come before generic ones.
KERNEL_CATEGORIES: list[tuple[str, str]] = [
    ("layout transform", r"nchwToNhwc|nhwcToNchw|transpose|permute"),
    ("reflect pad", r"reflection_pad"),
    ("conv", r"conv|cudnn|fprop|dgrad|wgrad|implicit_gemm|xmma"),
    ("gemm", r"gemm|cutlass|ampere_.*_(nn|nt|tn|tt)|sm\d+_.*(nn|nt|tn|tt)"),
    ("cat", r"CatArray|cat_"),
    ("cast", r"bfloat16_copy|float_copy|half_copy"),
    ("copy (layout)", r"direct_copy"),
    ("rand (fire mask)", r"distribution|philox|uniform|random"),
    ("reduce", r"reduce|sum|norm"),
    ("optimizer", r"multi_tensor|adam|foreach"),
    ("elementwise", r"elementwise|vectorized|unrolled|pointwise|threshold|relu"),
]


def categorize(kernel_name: str) -> str:
    for category, pattern in KERNEL_CATEGORIES:
        if re.search(pattern, kernel_name, flags=re.IGNORECASE):
            return category
    return "other"


def analytic_tflop(args, checkpointing: bool) -> float:
    """FLOPs of one training step: (forward + backward [+ recompute]) of both levels."""
    c, h = args.channel_n, args.hidden_size
    # p0 + p1 (3x3 convs), fc0 (3c -> h), fc1 (h -> c); 2 FLOPs per MAC.
    per_pixel_step = 2 * (2 * c * c * 9 + 3 * c * h + h * c)
    coarse = args.tile // args.scale_factor
    pixels = args.tile**2 + coarse**2
    forward = args.batch_size * args.steps * pixels * per_pixel_step
    # Backward ~2x forward; checkpointing re-runs the forward once.
    return forward * (4 if checkpointing else 3) / 1e12


def make_batch(args, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(0)
    imgs = torch.randint(
        0, 256, (args.batch_size, 3, args.tile, args.tile), device=device, generator=generator
    ).float()
    targets = torch.randint(
        0, args.num_classes, (args.batch_size, args.tile, args.tile), device=device,
        generator=generator,
    )
    return imgs, targets


def train_step(network, criterion, optimizer, imgs, targets, args, device) -> None:
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
        logits = network(imgs / 255.0, task="ignite")["ignite"]
        loss = criterion(logits.float(), targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_category_table(events, total_us: float) -> None:
    by_category: dict[str, float] = defaultdict(float)
    for event in events:
        by_category[categorize(event.key)] += event.self_device_time_total
    print("\nDevice time by kernel category")
    print(f"{'category':<20}{'ms':>12}{'share':>9}")
    for category, us in sorted(by_category.items(), key=lambda item: -item[1]):
        print(f"{category:<20}{us / 1e3:>12.1f}{100 * us / total_us:>8.1f}%")


def print_top_kernels(events, total_us: float, limit: int) -> None:
    print(f"\nTop {limit} CUDA kernels by device time")
    print(f"{'ms':>10}{'share':>8}{'calls':>9}  {'category':<18}kernel")
    for event in sorted(events, key=lambda e: -e.self_device_time_total)[:limit]:
        us = event.self_device_time_total
        name = event.key if len(event.key) <= 110 else event.key[:107] + "..."
        print(
            f"{us / 1e3:>10.1f}{100 * us / total_us:>7.1f}%{event.count:>9}  "
            f"{categorize(event.key):<18}{name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tile", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--every", type=int, default=8, help="grad_checkpointing_every; 0 = off")
    parser.add_argument("--channel-n", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3, help="timed iterations")
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--trace", type=str, default=None, help="chrome trace output path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    # Same as pathseg.cli.
    torch.set_float32_matmul_precision("medium")

    checkpointing = args.every > 0
    network = MedNCASegmenter(
        {"ignite": args.num_classes},
        channel_n=args.channel_n,
        hidden_size=args.hidden_size,
        steps=args.steps,
        scale_factor=args.scale_factor,
        grad_checkpointing_every=args.every or None,
    ).to(device)
    network.train()
    criterion = CrossEntropyDiceLoss(ignore_index=255).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0.05)
    imgs, targets = make_batch(args, device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"{name} | tile={args.tile} batch={args.batch_size} channel_n={args.channel_n} "
        f"hidden={args.hidden_size} steps={args.steps} every={args.every or 'off'} "
        f"bf16={args.bf16} cudnn.benchmark={args.cudnn_benchmark}"
    )

    for _ in range(args.warmup):
        train_step(network, criterion, optimizer, imgs, targets, args, device)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        train_step(network, criterion, optimizer, imgs, targets, args, device)
    synchronize(device)
    seconds = (time.perf_counter() - start) / args.iters

    tflop = analytic_tflop(args, checkpointing)
    print(
        f"\n{seconds:.3f} s/iter | {1 / seconds:.3f} it/s | {tflop:.1f} TFLOP/iter "
        f"(analytic) | {tflop / seconds:.1f} TFLOP/s achieved"
    )
    if device.type == "cuda":
        print(f"peak memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.1f} GB")

    if not args.profile:
        return

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        train_step(network, criterion, optimizer, imgs, targets, args, device)
        synchronize(device)

    averages = prof.key_averages()
    if device.type == "cuda":
        kernels = [e for e in averages if e.device_type == DeviceType.CUDA]
        total_us = sum(e.self_device_time_total for e in kernels)
        print(f"\nProfiled step: {total_us / 1e6:.3f} s of device time")
        print_category_table(kernels, total_us)
        print_top_kernels(kernels, total_us, args.row_limit)
        sort_by = "self_device_time_total"
    else:
        sort_by = "self_cpu_time_total"

    print(f"\nTop {args.row_limit} ops by {sort_by}")
    print(averages.table(sort_by=sort_by, row_limit=args.row_limit, max_name_column_width=60))

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\nChrome trace written to {args.trace}")


if __name__ == "__main__":
    main()
