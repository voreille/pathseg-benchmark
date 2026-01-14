from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# -----------------------------
# Helpers: IO
# -----------------------------
def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_ids(path: Path) -> np.ndarray:
    """
    Loads a label mask as integer IDs.
    Assumes your preprocessing saved masks as 'L' (uint8) or 'P' palette (indices).
    """
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im, dtype=np.int32)
        if im.mode == "L":
            return np.array(im, dtype=np.int32)
        raise ValueError(f"Unsupported mask mode {im.mode} for {path}")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGB to float tensor [C,H,W] in [0,1].
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,C]
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    return torch.from_numpy(arr)


# -----------------------------
# Episode specification
# -----------------------------
@dataclass(frozen=True)
class EpisodeSpec:
    ways: int  # N
    shots: int  # K
    queries: int  # Q
    crop_size: int = 448  # in pixels
    seed: Optional[int] = None


@dataclass(frozen=True)
class SampleRef:
    dataset_id: str
    sample_id: str
    image_relpath: str
    mask_relpath: str
    class_id: int


@dataclass(frozen=True)
class EpisodeRefs:
    dataset_id: str
    class_ids: List[int]  # length N (dataset-local IDs)
    support: List[SampleRef]  # length N*K
    query: List[SampleRef]  # length N*Q


# -----------------------------
# Crop logic (no bbox needed)
# -----------------------------
def sample_crop_center_on_class(
    mask: np.ndarray,
    class_id: int,
    crop_size: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Choose a crop top-left (y0,x0) such that crop likely contains pixels of `class_id`.
    Strategy: sample a pixel coordinate from that class, then center crop around it.

    Falls back to uniform crop if class pixels not found.
    """
    H, W = mask.shape
    cs = crop_size
    if H < cs or W < cs:
        # caller should handle resizing/padding; keep simple: clamp later
        pass

    ys, xs = np.where(mask == class_id)
    if ys.size == 0:
        # fallback: random crop
        y0 = rng.randint(0, max(0, H - cs))
        x0 = rng.randint(0, max(0, W - cs))
        return y0, x0

    idx = rng.randrange(ys.size)
    cy, cx = int(ys[idx]), int(xs[idx])

    y0 = cy - cs // 2
    x0 = cx - cs // 2

    y0 = max(0, min(y0, max(0, H - cs)))
    x0 = max(0, min(x0, max(0, W - cs)))
    return y0, x0


def crop_pair(
    img: Image.Image,
    mask: np.ndarray,
    y0: int,
    x0: int,
    crop_size: int,
) -> Tuple[Image.Image, np.ndarray]:
    cs = crop_size
    img_c = img.crop((x0, y0, x0 + cs, y0 + cs))
    mask_c = mask[y0 : y0 + cs, x0 : x0 + cs]
    return img_c, mask_c


# -----------------------------
# Transforms
# -----------------------------
class PairedTransform:
    """
    Applies class-conditioned random crop and optional augmentation to an (image, mask) pair.
    You can plug your own augmentation function that takes (PIL_img, np_mask) and returns same.
    """

    def __init__(
        self,
        crop_size: int = 448,
        augmentation: Optional[
            Callable[
                [Image.Image, np.ndarray, random.Random], Tuple[Image.Image, np.ndarray]
            ]
        ] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.crop_size = int(crop_size)
        self.augmentation = augmentation
        self._base_seed = seed

    def __call__(
        self,
        img: Image.Image,
        mask: np.ndarray,
        *,
        class_id: int,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If images can be smaller than crop, you should decide: pad or resize.
        # Minimal approach: raise early (forces you to preprocess consistently).
        H, W = mask.shape
        if H < self.crop_size or W < self.crop_size:
            raise ValueError(
                f"Image/mask smaller than crop_size={self.crop_size}: got {(H, W)}"
            )

        y0, x0 = sample_crop_center_on_class(mask, class_id, self.crop_size, rng)
        img_c, mask_c = crop_pair(img, mask, y0, x0, self.crop_size)

        if self.augmentation is not None:
            img_c, mask_c = self.augmentation(img_c, mask_c, rng)

        x = pil_to_tensor(img_c)  # [C,H,W], float
        y = torch.from_numpy(mask_c.astype(np.int64))  # [H,W], long
        return x, y


# -----------------------------
# Episode sampling
# -----------------------------
class EpisodeSampler:
    """
    Samples episodes from a global class_index (long-form table).
    Assumes class_index contains at least:
      dataset_id, dataset_class_id, sample_id, image_relpath, mask_relpath
    Optionally you can filter by split upstream.
    """

    def __init__(
        self,
        class_index: pd.DataFrame,
        *,
        datasets: Optional[Sequence[str]] = None,
        seed: int = 0,
    ) -> None:
        required = {
            "dataset_id",
            "dataset_class_id",
            "sample_id",
            "image_relpath",
            "mask_relpath",
        }
        missing = required - set(class_index.columns)
        if missing:
            raise ValueError(f"class_index is missing columns: {missing}")

        self.df = class_index.copy()
        self.df["dataset_class_id"] = self.df["dataset_class_id"].astype(int)

        if datasets is not None:
            self.df = self.df[self.df["dataset_id"].isin(list(datasets))]

        # Build fast lookup: (dataset_id, class_id) -> rows
        self._by_dataset: Dict[str, pd.DataFrame] = {}
        self._by_dataset_class: Dict[Tuple[str, int], pd.DataFrame] = {}

        for ds_id, sub in self.df.groupby("dataset_id"):
            self._by_dataset[ds_id] = sub
            for cid, sub2 in sub.groupby("dataset_class_id"):
                self._by_dataset_class[(ds_id, int(cid))] = sub2

        self.dataset_ids = sorted(self._by_dataset.keys())
        if not self.dataset_ids:
            raise ValueError("No datasets available in class_index after filtering.")

        self.rng = random.Random(seed)

    def sample_episode(self, spec: EpisodeSpec) -> EpisodeRefs:
        rng = self.rng if spec.seed is None else random.Random(spec.seed)

        dataset_id = rng.choice(self.dataset_ids)
        ds_df = self._by_dataset[dataset_id]

        available_classes = sorted(ds_df["dataset_class_id"].unique().tolist())
        if len(available_classes) < spec.ways:
            raise ValueError(
                f"Dataset {dataset_id} has {len(available_classes)} classes, but ways={spec.ways}"
            )

        class_ids = rng.sample(available_classes, k=spec.ways)

        support: List[SampleRef] = []
        query: List[SampleRef] = []

        for cid in class_ids:
            pool = self._by_dataset_class[(dataset_id, int(cid))]

            # Need at least shots+queries candidates for this class
            needed = spec.shots + spec.queries
            if len(pool) < needed:
                raise ValueError(
                    f"Not enough candidates for dataset={dataset_id} class={cid}: "
                    f"have {len(pool)}, need {needed}. "
                    "Consider lowering min_area or allowing sampling with replacement."
                )

            # sample without replacement
            chosen = pool.sample(n=needed, random_state=rng.randrange(10**9))
            chosen = chosen.reset_index(drop=True)

            for i in range(spec.shots):
                r = chosen.iloc[i]
                support.append(
                    SampleRef(
                        dataset_id=dataset_id,
                        sample_id=str(r["sample_id"]),
                        image_relpath=str(r["image_relpath"]),
                        mask_relpath=str(r["mask_relpath"]),
                        class_id=int(cid),
                    )
                )
            for i in range(spec.shots, needed):
                r = chosen.iloc[i]
                query.append(
                    SampleRef(
                        dataset_id=dataset_id,
                        sample_id=str(r["sample_id"]),
                        image_relpath=str(r["image_relpath"]),
                        mask_relpath=str(r["mask_relpath"]),
                        class_id=int(cid),
                    )
                )

        return EpisodeRefs(
            dataset_id=dataset_id,
            class_ids=[int(x) for x in class_ids],
            support=support,
            query=query,
        )


# -----------------------------
# EpisodeDataset (PyTorch)
# -----------------------------
class EpisodeDataset(Dataset):
    """
    Each __getitem__ returns one episode.
    __len__ is episodes_per_epoch (a training hyperparameter).
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        spec: EpisodeSpec,
        episodes_per_epoch: int,
        class_index_parquet: Path,
        split_csv: Optional[Path] = None,
        split: Optional[str] = None,  # "train"/"val"/"test"
        transform: Optional[PairedTransform] = None,
        seed: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.episodes_per_epoch = int(episodes_per_epoch)
        if self.episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be > 0")

        df = pd.read_parquet(class_index_parquet)

        # Optional split filtering: expects split_csv with columns [dataset_id, sample_id, split]
        if split_csv is not None and split is not None:
            sp = pd.read_csv(split_csv)
            required = {"dataset_id", "sample_id", "split"}
            miss = required - set(sp.columns)
            if miss:
                raise ValueError(f"split_csv missing columns: {miss}")
            sp = sp[sp["split"] == split][["dataset_id", "sample_id"]].copy()
            df = df.merge(sp, on=["dataset_id", "sample_id"], how="inner")

        self.sampler = EpisodeSampler(df, seed=seed)

        self.transform = transform or PairedTransform(
            crop_size=spec.crop_size, seed=seed
        )

        # For deterministic episode generation per index i (optional but useful)
        self._seed = seed

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Make episode sampling deterministic per idx if you want reproducibility
        episode_seed = (self._seed * 10**6) + int(idx)
        episode = self.sampler.sample_episode(
            EpisodeSpec(**{**self.spec.__dict__, "seed": episode_seed})
        )

        rng = random.Random(episode_seed)

        support_x: List[torch.Tensor] = []
        support_y: List[torch.Tensor] = []
        query_x: List[torch.Tensor] = []
        query_y: List[torch.Tensor] = []

        # Load + transform support
        for ref in episode.support:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(img, mask, class_id=ref.class_id, rng=rng)
            support_x.append(x)
            support_y.append(y)

        # Load + transform query
        for ref in episode.query:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(img, mask, class_id=ref.class_id, rng=rng)
            query_x.append(x)
            query_y.append(y)

        # Stack to fixed tensors
        support_images = torch.stack(support_x, dim=0)  # [N*K, C, H, W]
        support_masks = torch.stack(support_y, dim=0)  # [N*K, H, W]
        query_images = torch.stack(query_x, dim=0)  # [N*Q, C, H, W]
        query_masks = torch.stack(query_y, dim=0)  # [N*Q, H, W]

        return {
            "dataset_id": episode.dataset_id,
            "class_ids": torch.tensor(episode.class_ids, dtype=torch.long),  # [N]
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_images,
            "query_masks": query_masks,
        }


# -----------------------------
# Optional collate
# -----------------------------
def episode_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If all episodes have fixed shapes/counts, default collate works.
    This version is explicit and keeps dataset_id as a list[str].
    """
    out: Dict[str, Any] = {}
    out["dataset_id"] = [b["dataset_id"] for b in batch]
    out["class_ids"] = torch.stack([b["class_ids"] for b in batch], dim=0)  # [B,N]
    out["support_images"] = torch.stack(
        [b["support_images"] for b in batch], dim=0
    )  # [B,NS,C,H,W]
    out["support_masks"] = torch.stack(
        [b["support_masks"] for b in batch], dim=0
    )  # [B,NS,H,W]
    out["query_images"] = torch.stack(
        [b["query_images"] for b in batch], dim=0
    )  # [B,NQ,C,H,W]
    out["query_masks"] = torch.stack(
        [b["query_masks"] for b in batch], dim=0
    )  # [B,NQ,H,W]
    return out
