from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from pathseg.ood.residual_pca_scorer import FullPCA
from pathseg.training.linear_semantic import LinearSemantic
from pathseg.models.linear_decoder import LinearDecoder

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TileRecord:
    patient_id: str
    sample_id: str
    image_path: Path
    mask_path: Path
    top: int
    left: int
    tile_size: tuple[int, int] = (448, 448)


def normalize_hw(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value

    if len(value) != 2:
        raise ValueError(f"Expected an image size with two dimensions, got {value!r}.")

    return int(value[0]), int(value[1])


def normalize_fold(fold: str) -> str:
    """Normalize ``all``, ``0`` or ``fold0`` to a canonical fold label."""
    value = str(fold).strip().lower()

    if value == "all":
        return "all"

    if value.startswith("fold"):
        value = value[4:]

    if not value.isdigit():
        raise ValueError(
            f"Invalid fold {fold!r}. Expected 'all', an integer, or 'foldN'."
        )

    return f"fold{int(value)}"


def select_fit_metadata(
    metadata: pd.DataFrame,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training rows used for fitting and those held out for validation."""
    required_columns = {"split", "validation_fold", "sample_id"}
    missing = required_columns.difference(metadata.columns)

    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}.")

    selected_fold = normalize_fold(fold)

    is_train = metadata["split"].astype("string").str.strip().str.lower().eq("train")
    training = metadata.loc[is_train].copy()

    if training.empty:
        raise ValueError("The metadata contains no rows with split='train'.")

    training_folds = (
        training["validation_fold"].astype("string").str.strip().str.lower()
    )

    invalid_folds = ~training_folds.str.fullmatch(r"fold\d+", na=False)
    if invalid_folds.any():
        invalid_samples = training.loc[invalid_folds, "sample_id"].astype(str)
        preview = ", ".join(invalid_samples.head(10))
        suffix = "..." if len(invalid_samples) > 10 else ""

        raise ValueError(
            "Every training row must have validation_fold formatted as "
            f"'foldN'. Invalid samples: {preview}{suffix}"
        )

    if selected_fold == "all":
        fit = training
        held_out = training.iloc[:0]
    else:
        available_folds = sorted(training_folds.unique())

        if selected_fold not in available_folds:
            raise ValueError(
                f"Requested validation fold {selected_fold!r} is not present. "
                f"Available folds: {available_folds}."
            )

        is_held_out = training_folds.eq(selected_fold)
        fit = training.loc[~is_held_out]
        held_out = training.loc[is_held_out]

        if fit.empty:
            raise ValueError(
                f"No training rows remain after excluding {selected_fold}."
            )

    return fit.reset_index(drop=True), held_out.reset_index(drop=True)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image).copy()

    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional semantic mask at {path}, "
            f"got shape {mask.shape}."
        )

    return mask


def centered_tile_aligned_crop(
    image_size: int,
    tile_size: int,
) -> tuple[int, int]:
    """
    Return the start and size of the largest centred tile-aligned crop.

    The crop size is the largest multiple of ``tile_size`` that fits inside
    the image. Any remainder is removed as evenly as possible from both sides.
    """
    if image_size < tile_size:
        raise ValueError(
            f"Image dimension {image_size} is smaller than tile size {tile_size}."
        )

    crop_size = (image_size // tile_size) * tile_size
    crop_start = (image_size - crop_size) // 2

    return crop_start, crop_size


class TileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Deterministic non-overlapping tiles from context images.

    For each image:

    1. Centre-crop the image to the largest height and width that are exact
       multiples of the model input dimensions.
    2. Split that crop into deterministic non-overlapping model-sized tiles.
    3. Skip tiles containing no annotated pixels.

    The crop does not depend on the annotation bounding box. The full image
    tile is returned to the encoder, including unannotated context, while the
    mask determines which tokens contribute to PCA.
    """

    REQUIRED_COLUMNS = {
        "patient_id",
        "sample_id",
        "image_relpath",
        "mask_relpath",
    }

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: Path,
        tile_size: tuple[int, int],
        ignore_index: int = 255,
    ) -> None:
        missing_columns = self.REQUIRED_COLUMNS - set(metadata.columns)

        if missing_columns:
            raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}.")

        self.data_root = data_root
        self.tile_size = tile_size
        self.ignore_index = ignore_index

        metadata = metadata.copy()
        metadata["patient_id"] = metadata["patient_id"].astype(str)
        metadata["sample_id"] = metadata["sample_id"].astype(str)

        if metadata.empty:
            raise ValueError("No metadata rows were provided to the dataset.")

        self.records = self._build_index(metadata)

        if not self.records:
            raise RuntimeError("No valid context tiles were indexed.")

    def _build_index(
        self,
        metadata: pd.DataFrame,
    ) -> list[TileRecord]:
        tile_h, tile_w = self.tile_size
        records: list[TileRecord] = []

        for row in metadata.itertuples(index=False):
            patient_id = str(row.patient_id)
            sample_id = str(row.sample_id)

            image_path = self.data_root / str(row.image_relpath)
            mask_path = self.data_root / str(row.mask_relpath)

            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")

            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            mask = read_mask(mask_path)
            mask_height, mask_width = mask.shape

            if (image_height, image_width) != (mask_height, mask_width):
                raise ValueError(
                    f"Image and mask dimensions differ for {sample_id}: "
                    f"image={(image_height, image_width)}, "
                    f"mask={(mask_height, mask_width)}."
                )

            valid = mask != self.ignore_index

            if not valid.any():
                continue

            try:
                crop_top, crop_height = centered_tile_aligned_crop(
                    image_size=image_height,
                    tile_size=tile_h,
                )
                crop_left, crop_width = centered_tile_aligned_crop(
                    image_size=image_width,
                    tile_size=tile_w,
                )
            except ValueError as error:
                raise ValueError(
                    f"Could not construct a tile-aligned centre crop for "
                    f"sample {sample_id!r}."
                ) from error

            crop_bottom = crop_top + crop_height
            crop_right = crop_left + crop_width

            for top in range(crop_top, crop_bottom, tile_h):
                for left in range(crop_left, crop_right, tile_w):
                    tile_valid = valid[
                        top : top + tile_h,
                        left : left + tile_w,
                    ]

                    # No token from this tile could contribute to PCA.
                    if not tile_valid.any():
                        continue

                    records.append(
                        TileRecord(
                            patient_id=patient_id,
                            sample_id=sample_id,
                            image_path=image_path,
                            mask_path=mask_path,
                            top=top,
                            left=left,
                            tile_size=self.tile_size,
                        )
                    )

        return records

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        tile_h, tile_w = self.tile_size

        box = (
            record.left,
            record.top,
            record.left + tile_w,
            record.top + tile_h,
        )

        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            image_tile = np.asarray(image.crop(box)).copy()

        with Image.open(record.mask_path) as mask:
            mask_tile = np.asarray(mask.crop(box)).copy()

        if mask_tile.ndim == 3 and mask_tile.shape[-1] == 1:
            mask_tile = mask_tile[..., 0]

        if image_tile.shape[:2] != (tile_h, tile_w):
            raise RuntimeError(
                f"Unexpected image tile shape {image_tile.shape} "
                f"for {record.sample_id}."
            )

        if mask_tile.shape != (tile_h, tile_w):
            raise RuntimeError(
                f"Unexpected mask tile shape {mask_tile.shape} for {record.sample_id}."
            )

        image_tensor = torch.from_numpy(image_tile).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_tile)

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.records)


class FixedTilesPerPatientSampler(Sampler[int]):
    """
    Sample exactly `tiles_per_patient` indexed tiles per patient.

    Images are first sampled uniformly within each patient, then one tile is
    sampled uniformly from the selected image. This prevents patients or
    images with larger annotated regions from dominating the PCA fit.
    """

    def __init__(
        self,
        dataset: TileDataset,
        tiles_per_patient: int,
        seed: int = 0,
    ) -> None:
        if tiles_per_patient <= 0:
            raise ValueError("tiles_per_patient must be positive.")

        patient_images: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for index, record in enumerate(dataset.records):
            patient_images[record.patient_id][record.sample_id].append(index)

        if not patient_images:
            raise RuntimeError("The dataset contains no indexed patients.")

        rng = np.random.default_rng(seed)
        sampled_indices: list[int] = []

        for patient_id in sorted(patient_images):
            image_tiles = patient_images[patient_id]
            sample_ids = sorted(image_tiles)

            sampled_images = rng.choice(
                sample_ids,
                size=tiles_per_patient,
                replace=True,
            )

            for sample_id in sampled_images:
                tile_index = rng.choice(image_tiles[str(sample_id)])
                sampled_indices.append(int(tile_index))

        rng.shuffle(sampled_indices)

        self.indices = sampled_indices
        self.num_patients = len(patient_images)
        self.tiles_per_patient = tiles_per_patient

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_loader(
    dataset: TileDataset,
    sampler: Sampler[int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def masks_to_token_mask(
    masks: torch.Tensor,
    grid_size: tuple[int, int],
    ignore_index: int,
    min_valid_fraction: float,
) -> torch.Tensor:
    """
    Convert semantic masks [B,H,W] to valid-token masks [B,N].

    A token is retained when at least `min_valid_fraction` of its corresponding
    pixels have a mask value different from `ignore_index`.
    """
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape [B,H,W], got {tuple(masks.shape)}."
        )

    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    valid_pixels = (masks != ignore_index).float().unsqueeze(1)

    valid_fraction = F.adaptive_avg_pool2d(
        valid_pixels,
        output_size=grid_size,
    )

    return valid_fraction.flatten(1) >= min_valid_fraction


def get_feature_dim(network: torch.nn.Module) -> int:
    for obj in (network, getattr(network, "encoder", None)):
        if obj is None:
            continue

        for attribute in ("embed_dim", "feature_dim"):
            value = getattr(obj, attribute, None)

            if value is not None:
                return int(value)

    raise AttributeError(
        "Could not determine the encoder feature dimension. "
        "Expected `network.embed_dim`, `network.feature_dim`, "
        "`network.encoder.embed_dim`, or `network.encoder.feature_dim`."
    )


def get_grid_size(network: torch.nn.Module) -> tuple[int, int]:
    grid_size = getattr(network, "grid_size", None)

    if grid_size is None:
        encoder = getattr(network, "encoder", None)
        grid_size = getattr(encoder, "grid_size", None)

    if grid_size is None:
        raise AttributeError("Could not determine the encoder token grid size.")

    return normalize_hw(grid_size)


@torch.inference_mode()
def iter_valid_tokens(
    network: LinearDecoder,
    dataloader: DataLoader,
    grid_size: tuple[int, int],
    device: torch.device,
    ignore_index: int,
    min_valid_fraction: float,
) -> Iterator[torch.Tensor]:
    """
    Extract valid spatial tokens as CPU matrices shaped [M,D].

    The network receives complete context tiles. Only tokens sufficiently
    covered by annotated pixels contribute to PCA.
    """
    network.eval()

    expected_tokens = grid_size[0] * grid_size[1]

    for images, masks in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )

        # The LightningModule normally performs this conversion before calling
        # the decoder. We call the network directly, so it must be done here.
        encoder_inputs = images.float().div_(255.0)

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            tokens = network.forward_feature_maps(encoder_inputs)  # B, D, H, W
            tokens = tokens.flatten(2).transpose(1, 2)  # B, N, D

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected encoder output [B,N,D], got {tuple(tokens.shape)}."
            )

        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Encoder returned {tokens.shape[1]} spatial tokens, but "
                f"grid_size={grid_size} implies {expected_tokens}."
            )

        valid_token_mask = masks_to_token_mask(
            masks=masks,
            grid_size=grid_size,
            ignore_index=ignore_index,
            min_valid_fraction=min_valid_fraction,
        ).to(
            tokens.device,
            non_blocking=True,
        )

        valid_tokens = tokens[valid_token_mask]

        if valid_tokens.numel() > 0:
            yield valid_tokens.detach().float().cpu()


def first_component_at_threshold(
    cumulative_variance: np.ndarray,
    threshold: float,
) -> int:
    """
    Return the smallest one-based component count whose cumulative explained
    variance reaches the requested threshold.
    """
    index = np.searchsorted(
        cumulative_variance,
        threshold,
        side="left",
    )

    return min(int(index) + 1, len(cumulative_variance))


def estimate_spectrum_elbow(
    eigenvalues: np.ndarray,
    max_components: int,
) -> int | None:
    """
    Estimate a heuristic elbow using the maximum distance from the line
    joining the first and last points of the log eigenspectrum.

    The returned component number is one-based.
    """
    max_components = min(max_components, len(eigenvalues))

    spectrum = eigenvalues[:max_components]
    spectrum = spectrum[np.isfinite(spectrum) & (spectrum > 0)]

    if len(spectrum) < 3:
        return None

    x = np.arange(len(spectrum), dtype=np.float64)
    y = np.log10(spectrum)

    x_range = x[-1] - x[0]
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        return None

    x_normalized = (x - x[0]) / x_range
    y_normalized = (y - y.min()) / y_range

    line = y_normalized[0] + (y_normalized[-1] - y_normalized[0]) * x_normalized

    distances = np.abs(y_normalized - line)

    return int(np.argmax(distances)) + 1


def assess_minka_dimension(
    spectrum: np.ndarray,
    rank: int,
    n_samples: int,
) -> float:
    """
    Minka's PCA dimensionality log-likelihood.

    This follows the probabilistic PCA dimensionality criterion also used for
    PCA MLE estimation. The spectrum must be sorted in descending order.
    """
    n_features = len(spectrum)
    epsilon = 1e-15

    if rank < 1 or rank >= n_features:
        return float("-inf")

    if spectrum[rank - 1] <= epsilon:
        return float("-inf")

    pu = -rank * math.log(2.0)

    for i in range(1, rank + 1):
        value = (n_features - i + 1) / 2.0

        pu += math.lgamma(value) - value * math.log(math.pi)

    pl = -0.5 * n_samples * float(np.log(spectrum[:rank]).sum())

    noise_variance = max(
        epsilon,
        float(spectrum[rank:].mean()),
    )

    pv = -0.5 * n_samples * (n_features - rank) * math.log(noise_variance)

    model_dimension = n_features * rank - rank * (rank + 1) / 2.0

    pp = 0.5 * (model_dimension + rank) * math.log(2.0 * math.pi)

    modeled_spectrum = spectrum.copy()
    modeled_spectrum[rank:] = noise_variance

    pa = 0.0

    for i in range(rank):
        eigenvalue_differences = spectrum[i] - spectrum[i + 1 :]

        inverse_differences = (
            1.0 / modeled_spectrum[i + 1 :] - 1.0 / modeled_spectrum[i]
        )

        terms = eigenvalue_differences * inverse_differences
        terms = np.maximum(terms, epsilon)

        pa += float(np.log(terms).sum())
        pa += (n_features - i - 1) * math.log(n_samples)

    return pu + pl + pv + pp - 0.5 * pa - 0.5 * rank * math.log(n_samples)


def estimate_minka_rank(
    eigenvalues: np.ndarray,
    n_samples: int,
    max_components: int,
) -> tuple[int | None, np.ndarray, np.ndarray]:
    """
    Estimate the Minka MLE rank up to `max_components`.

    Limiting the candidate range avoids an expensive search over all embedding
    dimensions. If the optimum equals the configured maximum, the true optimum
    may lie above the search range.
    """
    positive_spectrum = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]

    if len(positive_spectrum) < 2:
        return None, np.empty(0, dtype=int), np.empty(0)

    if n_samples <= len(positive_spectrum):
        # PCA MLE is not well behaved when there are not more observations
        # than feature dimensions.
        return None, np.empty(0, dtype=int), np.empty(0)

    max_rank = min(
        max_components,
        len(positive_spectrum) - 1,
    )

    if max_rank < 1:
        return None, np.empty(0, dtype=int), np.empty(0)

    ranks = np.arange(1, max_rank + 1)

    scores = np.asarray(
        [
            assess_minka_dimension(
                spectrum=positive_spectrum,
                rank=int(rank),
                n_samples=n_samples,
            )
            for rank in ranks
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(scores)

    if not finite.any():
        return None, ranks, scores

    best_index = int(np.nanargmax(np.where(finite, scores, np.nan)))

    return int(ranks[best_index]), ranks, scores


def save_pca_diagnostics(
    pca: FullPCA,
    diagnostics_dir: Path,
    max_plot_components: int,
    mle_max_components: int,
) -> dict:
    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eigenvalues = pca.explained_variance.detach().double().cpu().numpy()

    explained_variance_ratio = (
        pca.explained_variance_ratio.detach().double().cpu().numpy()
    )

    components = np.arange(
        1,
        len(eigenvalues) + 1,
    )

    cumulative_variance = np.cumsum(explained_variance_ratio)

    spectrum_table = pd.DataFrame(
        {
            "component": components,
            "explained_variance": eigenvalues,
            "explained_variance_ratio": explained_variance_ratio,
            "explained_variance_percent": (100.0 * explained_variance_ratio),
            "cumulative_explained_variance": cumulative_variance,
            "cumulative_explained_variance_percent": (100.0 * cumulative_variance),
        }
    )

    spectrum_table.to_csv(
        diagnostics_dir / "explained_variance.csv",
        index=False,
    )

    plot_count = min(
        max_plot_components,
        len(eigenvalues),
    )

    plotted_components = components[:plot_count]

    elbow_component = estimate_spectrum_elbow(
        eigenvalues=eigenvalues,
        max_components=max_plot_components,
    )

    mle_component = None
    mle_ranks = np.empty(0, dtype=int)
    mle_scores = np.empty(0)

    if mle_max_components > 0:
        mle_component, mle_ranks, mle_scores = estimate_minka_rank(
            eigenvalues=eigenvalues,
            n_samples=int(pca.n_fit_tokens),
            max_components=mle_max_components,
        )

    # Scree plot.
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        plotted_components,
        100.0 * explained_variance_ratio[:plot_count],
    )

    if elbow_component is not None:
        axis.axvline(
            elbow_component,
            linestyle="--",
            label=f"Heuristic elbow: {elbow_component}",
        )

    if mle_component is not None:
        axis.axvline(
            mle_component,
            linestyle=":",
            label=f"MLE rank: {mle_component}",
        )

    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance (%)")
    axis.set_title("PCA scree plot")
    axis.grid(alpha=0.3)

    if elbow_component is not None or mle_component is not None:
        axis.legend()

    figure.tight_layout()
    figure.savefig(
        diagnostics_dir / "scree_plot.png",
        dpi=200,
    )
    plt.close(figure)

    # Cumulative explained variance.
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        plotted_components,
        100.0 * cumulative_variance[:plot_count],
    )

    for threshold in (0.90, 0.95, 0.99):
        component = first_component_at_threshold(
            cumulative_variance,
            threshold,
        )

        axis.axhline(
            threshold * 100.0,
            linestyle="--",
            alpha=0.5,
        )

        if component <= plot_count:
            axis.axvline(
                component,
                linestyle=":",
                alpha=0.5,
            )

    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance (%)")
    axis.set_ylim(0.0, 101.0)
    axis.set_title("Cumulative explained variance")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        diagnostics_dir / "cumulative_variance.png",
        dpi=200,
    )
    plt.close(figure)

    # Log eigenspectrum, often clearer for a low-rank nuisance subspace.
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.semilogy(
        plotted_components,
        eigenvalues[:plot_count],
    )

    if elbow_component is not None:
        axis.axvline(
            elbow_component,
            linestyle="--",
            label=f"Heuristic elbow: {elbow_component}",
        )

    if mle_component is not None:
        axis.axvline(
            mle_component,
            linestyle=":",
            label=f"MLE rank: {mle_component}",
        )

    axis.set_xlabel("Principal component")
    axis.set_ylabel("Eigenvalue")
    axis.set_title("PCA eigenspectrum")
    axis.grid(alpha=0.3)

    if elbow_component is not None or mle_component is not None:
        axis.legend()

    figure.tight_layout()
    figure.savefig(
        diagnostics_dir / "log_spectrum.png",
        dpi=200,
    )
    plt.close(figure)

    # MLE likelihood curve.
    if len(mle_ranks) > 0:
        finite_scores = np.isfinite(mle_scores)

        relative_scores = np.full_like(
            mle_scores,
            np.nan,
        )

        if finite_scores.any():
            relative_scores[finite_scores] = (
                mle_scores[finite_scores] - mle_scores[finite_scores].max()
            )

        pd.DataFrame(
            {
                "rank": mle_ranks,
                "log_likelihood": mle_scores,
                "relative_log_likelihood": relative_scores,
            }
        ).to_csv(
            diagnostics_dir / "mle_scores.csv",
            index=False,
        )

        figure, axis = plt.subplots(figsize=(8, 5))

        axis.plot(
            mle_ranks,
            relative_scores,
        )

        if mle_component is not None:
            axis.axvline(
                mle_component,
                linestyle="--",
                label=f"MLE rank: {mle_component}",
            )
            axis.legend()

        axis.set_xlabel("Candidate PCA rank")
        axis.set_ylabel("Relative log-likelihood")
        axis.set_title("Minka PCA dimensionality estimate")
        axis.grid(alpha=0.3)

        figure.tight_layout()
        figure.savefig(
            diagnostics_dir / "mle_log_likelihood.png",
            dpi=200,
        )
        plt.close(figure)

    positive_ratios = explained_variance_ratio[explained_variance_ratio > 0]

    effective_rank = float(np.exp(-np.sum(positive_ratios * np.log(positive_ratios))))

    participation_ratio = float(1.0 / np.sum(positive_ratios**2))

    stable_rank = float(eigenvalues.sum() / eigenvalues[0])

    summary = {
        "n_components": int(len(eigenvalues)),
        "n_fit_tokens": int(pca.n_fit_tokens),
        "components_for_90_percent": first_component_at_threshold(
            cumulative_variance,
            0.90,
        ),
        "components_for_95_percent": first_component_at_threshold(
            cumulative_variance,
            0.95,
        ),
        "components_for_99_percent": first_component_at_threshold(
            cumulative_variance,
            0.99,
        ),
        "heuristic_elbow_component": elbow_component,
        "heuristic_elbow_search_limit": int(plot_count),
        "minka_mle_component": mle_component,
        "minka_mle_search_limit": (int(mle_ranks[-1]) if len(mle_ranks) > 0 else None),
        "minka_mle_reached_search_limit": (
            mle_component is not None
            and len(mle_ranks) > 0
            and mle_component == int(mle_ranks[-1])
        ),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "stable_rank": stable_rank,
    }

    with (diagnostics_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


@click.command()
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Self-reconstructing LinearSemantic checkpoint.",
)
@click.option(
    "--metadata",
    "metadata_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Dataset metadata CSV.",
)
@click.option(
    "--data-root",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Root directory for image_relpath and mask_relpath.",
)
@click.option(
    "--fold",
    type=str,
    default="all",
    show_default=True,
    help=(
        "Validation fold to exclude from PCA fitting: 'all', an integer, "
        "or 'foldN'. The default 'all' uses every row with split='train'. "
        "Rows with split='test' are never used."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Directory receiving pca.pt and the diagnostics subdirectory.",
)
@click.option(
    "--tiles-per-patient",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of indexed context tiles sampled per patient.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=8,
    show_default=True,
)
@click.option(
    "--prefetch-factor",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
)
@click.option(
    "--ignore-index",
    type=int,
    default=None,
    help=(
        "Mask value indicating unannotated pixels. "
        "Defaults to the checkpoint model's ignore_idx."
    ),
)
@click.option(
    "--min-valid-fraction",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="Minimum annotated fraction required to retain a token.",
)
@click.option(
    "--normalize-embeddings/--no-normalize-embeddings",
    default=False,
    show_default=True,
    help="L2-normalize embeddings before fitting PCA.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--device",
    default=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    show_default="cuda when available, otherwise cpu",
    help="Device used for encoder inference.",
)
@click.option(
    "--pca-device",
    default="cpu",
    show_default=True,
    help="Device used to accumulate and decompose the covariance matrix.",
)
@click.option(
    "--max-plot-components",
    type=click.IntRange(min=2),
    default=256,
    show_default=True,
    help="Maximum number of components shown in spectrum plots.",
)
@click.option(
    "--mle-max-components",
    type=click.IntRange(min=0),
    default=256,
    show_default=True,
    help=(
        "Maximum candidate rank evaluated with Minka MLE. "
        "Use 0 to disable MLE diagnostics."
    ),
)
def main(
    checkpoint_path: Path,
    metadata_path: Path,
    data_root: Path,
    fold: str,
    output_dir: Path,
    tiles_per_patient: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    ignore_index: int | None,
    min_valid_fraction: float,
    normalize_embeddings: bool,
    seed: int,
    device: str,
    pca_device: str,
    max_plot_components: int,
    mle_max_components: int,
) -> None:
    """
    Fit a full PCA on patient-balanced context-tile embeddings.

    By default, all rows with ``split == "train"`` are used. When a specific
    validation fold is requested, that fold is excluded. Test rows are never
    used.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    inference_device = torch.device(device)

    click.echo(f"Loading checkpoint: {checkpoint_path}")

    model = LinearSemantic.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    model = model.to(inference_device)
    model.eval()

    network = model.network
    if not isinstance(network, LinearDecoder):
        raise TypeError(
            f"Expected a LinearDecoder network, got {type(network).__name__}."
        )

    tile_size = normalize_hw(model.img_size)
    grid_size = get_grid_size(network)
    feature_dim = get_feature_dim(network)

    click.echo(f"Model input size: {tile_size}")
    click.echo(f"Token grid size: {grid_size}")
    click.echo(f"Embedding dimension: {feature_dim}")

    if ignore_index is None:
        ignore_index = int(model.ignore_idx)

    selected_fold = normalize_fold(fold)

    metadata = pd.read_csv(metadata_path)
    fit_metadata, held_out_metadata = select_fit_metadata(
        metadata=metadata,
        fold=fold,
    )

    n_fit_patients = fit_metadata["patient_id"].astype(str).nunique()
    n_held_out_patients = held_out_metadata["patient_id"].astype(str).nunique()

    click.echo(
        f"Using {len(fit_metadata):,} training rows from {n_fit_patients:,} patients."
    )

    if selected_fold == "all":
        click.echo("Using all validation folds; no training rows are held out.")
    else:
        click.echo(
            f"Holding out {len(held_out_metadata):,} rows assigned to "
            f"{selected_fold} ({n_held_out_patients:,} patients)."
        )
    click.echo(f"Ignore index: {ignore_index}")

    dataset = TileDataset(
        metadata=fit_metadata,
        data_root=data_root,
        tile_size=tile_size,
        ignore_index=ignore_index,
    )

    sampler = FixedTilesPerPatientSampler(
        dataset=dataset,
        tiles_per_patient=tiles_per_patient,
        seed=seed,
    )

    dataloader = build_loader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=inference_device.type == "cuda",
    )

    click.echo(
        f"Indexed {len(dataset):,} valid non-overlapping tiles from "
        f"{sampler.num_patients} patients."
    )
    click.echo(
        f"Sampling {len(sampler):,} tiles ({sampler.tiles_per_patient} per patient)."
    )

    pca = FullPCA(
        feature_dim=feature_dim,
        normalize_embeddings=normalize_embeddings,
    )

    pca.fit(
        iter_valid_tokens(
            network=network,
            dataloader=dataloader,
            grid_size=grid_size,
            device=inference_device,
            ignore_index=ignore_index,
            min_valid_fraction=min_valid_fraction,
        ),
        fit_device=pca_device,
        fit_dtype=torch.float64,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    fit_metadata.to_csv(
        diagnostics_dir / "fit_metadata.csv",
        index=False,
    )
    held_out_metadata.to_csv(
        diagnostics_dir / "held_out_metadata.csv",
        index=False,
    )

    if selected_fold == "all":
        fit_rule = "split == 'train'"
        held_out_rule = None
    else:
        fit_rule = "split == 'train' and validation_fold != selected_validation_fold"
        held_out_rule = (
            "split == 'train' and validation_fold == selected_validation_fold"
        )

    selection_summary = {
        "selected_validation_fold": selected_fold,
        "fit_rule": fit_rule,
        "held_out_rule": held_out_rule,
        "n_fit_rows": int(len(fit_metadata)),
        "n_fit_patients": int(n_fit_patients),
        "n_held_out_rows": int(len(held_out_metadata)),
        "n_held_out_patients": int(n_held_out_patients),
    }

    with (diagnostics_dir / "data_selection.json").open("w", encoding="utf-8") as file:
        json.dump(selection_summary, file, indent=2)

    click.echo("Computing PCA diagnostics.")

    diagnostics_summary = save_pca_diagnostics(
        pca=pca,
        diagnostics_dir=diagnostics_dir,
        max_plot_components=max_plot_components,
        mle_max_components=mle_max_components,
    )

    pca_path = output_dir / "pca.pt"

    torch.save(
        {
            "pca_state_dict": pca.state_dict(),
            "feature_dim": pca.feature_dim,
            "normalize_embeddings": pca.normalize_embeddings,
            "n_fit_tokens": pca.n_fit_tokens,
            "tile_size": tile_size,
            "grid_size": grid_size,
            "ignore_index": ignore_index,
            "min_valid_fraction": min_valid_fraction,
            "tiles_per_patient": tiles_per_patient,
            "n_indexed_tiles": len(dataset),
            "n_sampled_tiles": len(sampler),
            "n_patients": sampler.num_patients,
            "validation_fold": selected_fold,
            "fit_metadata_rows": len(fit_metadata),
            "held_out_metadata_rows": len(held_out_metadata),
            "seed": seed,
            "segmentation_checkpoint": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "data_root": str(data_root),
            "data_selection": selection_summary,
            "diagnostics": diagnostics_summary,
        },
        pca_path,
    )

    click.echo(f"Saved PCA fitted on {pca.n_fit_tokens:,} tokens to {pca_path}.")
    click.echo(f"Saved diagnostic plots and tables to {diagnostics_dir}.")

    click.echo(
        "Diagnostic estimates: "
        f"elbow={diagnostics_summary['heuristic_elbow_component']}, "
        f"MLE={diagnostics_summary['minka_mle_component']}, "
        f"90%={diagnostics_summary['components_for_90_percent']}, "
        f"95%={diagnostics_summary['components_for_95_percent']}."
    )


if __name__ == "__main__":
    main()
