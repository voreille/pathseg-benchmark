"""Write a decoded ``.npy`` copy of a dataset's images and masks.

``pathseg.datasets.dataset.Dataset`` loads ``.npy`` files as the same pixels as the
PNGs (images H×W×3 uint8 as from ``convert("RGB")``, masks H×W uint8 as from
``convert("L")``), without decoding. That saves ~65 ms per IGNITE ROI in each
DataLoader worker. Point ``images_subdir`` / ``masks_subdir`` at the new folders.

    uv run python -m pathseg.preprocessing.npy_cache data/fss/ignite \
        --subdirs images:images_npy masks_semantic:masks_semantic_npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def write_cache(src: Path, dst: Path, mode: str) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        out = dst / f"{path.stem}.npy"
        if out.exists():
            continue
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(Image.open(path).convert(mode)))
        tmp.rename(out)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["images:images_npy", "masks_semantic:masks_semantic_npy"],
        help="src:dst pairs under root; a src starting with 'mask' is read as 'L'.",
    )
    args = parser.parse_args()
    for pair in args.subdirs:
        src, dst = pair.split(":")
        mode = "L" if src.startswith("mask") else "RGB"
        n = write_cache(args.root / src, args.root / dst, mode)
        print(f"{src} -> {dst} ({mode}): {n} new files")


if __name__ == "__main__":
    main()
