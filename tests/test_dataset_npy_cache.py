from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from pathseg.datasets.dataset import Dataset
from pathseg.datasets.transforms import CustomTransforms
from pathseg.preprocessing.npy_cache import write_cache


def make_roi(root, image_id: str, size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    h, w = size
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
        root / "images" / f"{image_id}.png"
    )
    mask = rng.choice(np.array([0, 1, 5, 255], dtype=np.uint8), size=(h, w))
    Image.fromarray(mask).save(root / "masks" / f"{image_id}.png")


def test_npy_cache_gives_identical_samples(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    ids = ["a", "b"]
    make_roi(tmp_path, "a", (70, 90), seed=0)
    make_roi(tmp_path, "b", (50, 40), seed=1)
    assert write_cache(tmp_path / "images", tmp_path / "images_npy", "RGB") == 2
    assert write_cache(tmp_path / "masks", tmp_path / "masks_npy", "L") == 2
    assert write_cache(tmp_path / "images", tmp_path / "images_npy", "RGB") == 0

    transforms = CustomTransforms(img_size=(48, 48), scale_range=(0.7, 1.4))
    for tf in (None, transforms):
        png = Dataset(ids, tmp_path / "images", tmp_path / "masks", transforms=tf)
        npy = Dataset(ids, tmp_path / "images_npy", tmp_path / "masks_npy", transforms=tf)
        assert npy.data_df["mask_path"].map(lambda p: p.suffix).eq(".npy").all()
        for i in range(len(ids)):
            torch.manual_seed(i)
            image_a, target_a = png[i]
            torch.manual_seed(i)
            image_b, target_b = npy[i]
            assert type(image_a) is type(image_b)
            assert torch.equal(image_a, image_b)
            assert torch.equal(target_a["masks"], target_b["masks"])
            assert torch.equal(target_a["labels"], target_b["labels"])
