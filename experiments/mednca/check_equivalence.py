"""Check that the current Med-NCA computes the same function as a reference commit.

Throughput work must not change the computed function (see CLAUDE.md, "Training
throughput"). This script loads ``pathseg/models/architectures/med_nca.py`` from a
reference commit (default ``071dfa1``, the last version before the optimizations),
builds both networks with the same weights, and compares one training forward +
backward on GPU under the same seed (same fire masks):

- fp32 and bf16 autocast, with and without gradient checkpointing;
- max |output difference| and max relative gradient difference per parameter;
- the noise floor: the reference's own bf16 vs fp32 difference.

Expected: in fp32 with ``--no-tf32``, about 1e-7. In bf16, differences at or below
the noise floor. Without ``--no-tf32``, fp32 convs use TF32 on Ampere and differ
around 1e-4.

Example (on a free GPU):

    CUDA_VISIBLE_DEVICES=1 python experiments/mednca/check_equivalence.py --no-tf32
    CUDA_VISIBLE_DEVICES=1 python experiments/mednca/check_equivalence.py --compile-step
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pathseg.models.architectures import med_nca as current

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = "pathseg/models/architectures/med_nca.py"


def load_reference(commit: str):
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{MODULE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "med_nca_reference.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("med_nca_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(module, args, *, bf16: bool, every: int | None, compile_step: bool = False):
    kwargs = {"compile_step": True} if compile_step else {}
    network = module.MedNCASegmenter(
        {"ignite": args.num_classes},
        steps=args.steps,
        grad_checkpointing_every=every,
        **kwargs,
    )
    network = network.cuda().train()
    # Same random weights for both (fc1 is zero-initialised otherwise).
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    imgs = torch.rand(
        2, 3, args.height, args.width, generator=torch.Generator().manual_seed(2)
    ).cuda()

    torch.manual_seed(0)
    with torch.autocast("cuda", torch.bfloat16, enabled=bf16):
        out = network(imgs)["ignite"].float()
    out.square().mean().backward()
    grads = {name: p.grad for name, p in network.named_parameters()}
    return out.detach(), grads


def compare(a, b) -> str:
    (out_a, grads_a), (out_b, grads_b) = a, b
    rel = max(
        ((grads_b[n] - grads_a[n]).norm() / grads_a[n].norm()).item() for n in grads_a
    )
    return (
        f"out max|d|={(out_b - out_a).abs().max():.2e} "
        f"(max|out|={out_a.abs().max():.2f})  grad max rel={rel:.2e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="071dfa1", help="reference commit")
    parser.add_argument("--no-tf32", action="store_true", help="true fp32 convs")
    parser.add_argument(
        "--compile-step", action="store_true", help="current network with compile_step"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--num-classes", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required.")
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    reference = load_reference(args.ref)
    print(
        f"reference={args.ref} tf32={'off' if args.no_tf32 else 'on'} "
        f"compile_step={args.compile_step} steps={args.steps} "
        f"input=2x3x{args.height}x{args.width}"
    )

    for bf16 in (False, True):
        for every in (None, 4):
            expected = run(reference, args, bf16=bf16, every=every)
            actual = run(
                current, args, bf16=bf16, every=every, compile_step=args.compile_step
            )
            print(f"bf16={bf16!s:<5} every={every!s:<4}: {compare(expected, actual)}")

    fp32 = run(reference, args, bf16=False, every=None)
    bf16 = run(reference, args, bf16=True, every=None)
    print(f"noise floor (reference bf16 vs fp32): {compare(fp32, bf16)}")


if __name__ == "__main__":
    main()
