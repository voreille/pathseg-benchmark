from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import click
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pathseg.models.linear_decoder import LinearDecoder
from pathseg.training.linear_semantic import LinearSemantic
from pathseg.training.tiler import GridPadTiler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class ScorerConfig:
    score_type: str
    n_major_components: int
    end_component: int
    normalize_embeddings: bool
    normalize_by_dimension: bool
    relative_ridge: float
    min_eigenvalue_ratio: float | None


@dataclass(frozen=True)
class ImageResult:
    image: str
    score_path: str
    preview_path: str | None
    height: int
    width: int
    score_min: float
    score_max: float
    score_mean: float
    score_std: float


def _load_torch_file(path: Path) -> dict[str, Any]:
    """Load a trusted project artifact while supporting older PyTorch versions."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}.")

    return payload


def _state_tensor(state_dict: dict[str, Any], name: str) -> torch.Tensor:
    """Retrieve a PCA tensor by exact key or an unambiguous dotted suffix."""
    if name in state_dict:
        value = state_dict[name]
    else:
        candidates = [
            key
            for key in state_dict
            if key.endswith(f".{name}") or key.endswith(name)
        ]

        if len(candidates) != 1:
            raise KeyError(
                f"Could not uniquely find PCA tensor {name!r}. "
                f"Candidate keys: {candidates}. Available keys: {sorted(state_dict)}"
            )

        value = state_dict[candidates[0]]

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Expected PCA state {name!r} to be a tensor, got {type(value).__name__}."
        )

    return value


def resolve_checkpoint_path(
    pca_path: Path,
    pca_payload: dict[str, Any],
    checkpoint_override: Path | None,
) -> Path:
    if checkpoint_override is not None:
        return checkpoint_override.resolve()

    stored = pca_payload.get("segmentation_checkpoint")
    if not stored:
        raise KeyError(
            "The PCA file does not contain `segmentation_checkpoint`. "
            "Pass --checkpoint explicitly."
        )

    stored_path = Path(str(stored)).expanduser()
    candidates = [stored_path]

    if not stored_path.is_absolute():
        candidates.extend(
            [
                pca_path.parent / stored_path,
                pca_path.parent.parent / stored_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "The checkpoint path stored in the PCA artifact does not exist. "
        "Pass --checkpoint to override it. Attempted:\n"
        f"{attempted}"
    )


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()

    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise click.ClickException(
            f"No supported images were found in {input_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    return images


def load_rgb_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image_array = np.asarray(image.convert("RGB")).copy()

    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


def select_pca_subspace(
    components: torch.Tensor,
    eigenvalues: torch.Tensor,
    n_major_components: int,
    end_component: int | None,
    min_eigenvalue_ratio: float | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if components.ndim != 2:
        raise ValueError(
            f"Expected PCA components [D,D], got {tuple(components.shape)}."
        )

    if eigenvalues.ndim != 1:
        raise ValueError(
            f"Expected PCA explained variance [D], got {tuple(eigenvalues.shape)}."
        )

    if components.shape[0] != eigenvalues.shape[0]:
        raise ValueError(
            "PCA components and explained variance have incompatible dimensions: "
            f"{tuple(components.shape)} versus {tuple(eigenvalues.shape)}."
        )

    feature_dim = int(components.shape[0])

    if not 0 <= n_major_components < feature_dim:
        raise ValueError(
            f"n_major_components must be in [0, {feature_dim - 1}], "
            f"got {n_major_components}."
        )

    selected_end = feature_dim if end_component is None else int(end_component)

    if not n_major_components < selected_end <= feature_dim:
        raise ValueError(
            "end_component must be larger than n_major_components and no larger "
            f"than {feature_dim}; got start={n_major_components}, end={selected_end}."
        )

    if min_eigenvalue_ratio is not None:
        if not 0.0 < min_eigenvalue_ratio <= 1.0:
            raise ValueError("min_eigenvalue_ratio must be in (0, 1].")

        maximum_eigenvalue = float(eigenvalues[0].item())
        threshold = maximum_eigenvalue * min_eigenvalue_ratio
        retained = int((eigenvalues >= threshold).sum().item())
        selected_end = min(selected_end, retained)

        if selected_end <= n_major_components:
            raise ValueError(
                "The eigenvalue-ratio threshold removes the complete residual "
                "subspace. Lower --min-eigenvalue-ratio or "
                "--n-major-components."
            )

    return (
        components[n_major_components:selected_end],
        eigenvalues[n_major_components:selected_end],
        selected_end,
    )


class ResidualPCAScorer:
    """Token-wise Euclidean or regularized Mahalanobis PCA residual score."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        score_type: str,
        n_major_components: int,
        end_component: int | None,
        min_eigenvalue_ratio: float | None,
        relative_ridge: float,
        normalize_embeddings: bool,
        normalize_by_dimension: bool,
        device: torch.device,
    ) -> None:
        if mean.ndim != 1:
            raise ValueError(f"Expected PCA mean [D], got {tuple(mean.shape)}.")

        if components.shape[1] != mean.shape[0]:
            raise ValueError(
                "PCA mean and components have incompatible feature dimensions: "
                f"{mean.shape[0]} and {components.shape[1]}."
            )

        selected_components, selected_eigenvalues, selected_end = select_pca_subspace(
            components=components.float(),
            eigenvalues=eigenvalues.float(),
            n_major_components=n_major_components,
            end_component=end_component,
            min_eigenvalue_ratio=min_eigenvalue_ratio,
        )

        normalized_score_type = score_type.lower()
        if normalized_score_type == "normal":
            normalized_score_type = "euclidean"

        if normalized_score_type not in {"euclidean", "mahalanobis"}:
            raise ValueError(
                "score_type must be `euclidean`, `normal`, or `mahalanobis`."
            )

        if relative_ridge < 0.0:
            raise ValueError("relative_ridge must be non-negative.")

        self.mean = mean.float().to(device)
        self.components = selected_components.to(device)
        self.eigenvalues = selected_eigenvalues.to(device)
        self.score_type = normalized_score_type
        self.normalize_embeddings = normalize_embeddings
        self.normalize_by_dimension = normalize_by_dimension
        self.n_major_components = n_major_components
        self.end_component = selected_end
        self.relative_ridge = relative_ridge
        self.device = device

        if self.score_type == "mahalanobis":
            reference_eigenvalue = eigenvalues.float().max().clamp_min(0.0)
            ridge = relative_ridge * reference_eigenvalue
            self.denominator = torch.sqrt(
                self.eigenvalues.clamp_min(0.0) + ridge.to(device)
            ).clamp_min(torch.finfo(torch.float32).eps)
        else:
            self.denominator = None

    @property
    def n_selected_components(self) -> int:
        return int(self.components.shape[0])

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens [B,N,D], got {tuple(tokens.shape)}."
            )

        x = tokens.float()

        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=-1)

        centered = x - self.mean
        coordinates = centered @ self.components.transpose(0, 1)

        if self.denominator is not None:
            coordinates = coordinates / self.denominator

        scores = torch.linalg.vector_norm(coordinates, dim=-1)

        if self.normalize_by_dimension:
            scores = scores / np.sqrt(self.n_selected_components)

        return scores


def save_score_preview(
    score_map: np.ndarray,
    path: Path,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    finite = score_map[np.isfinite(score_map)]

    if finite.size == 0:
        raise ValueError("Cannot create a preview from a map without finite values.")

    lower = float(np.percentile(finite, lower_percentile))
    upper = float(np.percentile(finite, upper_percentile))

    if upper <= lower:
        lower = float(finite.min())
        upper = float(finite.max())

    if upper <= lower:
        upper = lower + 1.0

    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(score_map, vmin=lower, vmax=upper, cmap="magma")
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="OOD score")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def score_image(
    image: torch.Tensor,
    network: LinearDecoder,
    tiler: GridPadTiler,
    scorer: ResidualPCAScorer,
    *,
    batch_size: int,
    inference_device: torch.device,
    stitch_device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    crops, origins, image_sizes = tiler.window([image])
    tile_score_maps: list[torch.Tensor] = []

    for start in range(0, len(crops), batch_size):
        crop_batch = crops[start : start + batch_size].to(
            inference_device,
            non_blocking=True,
        )
        encoder_inputs = crop_batch.float().div_(255.0)

        with torch.autocast(
            device_type=inference_device.type,
            enabled=use_amp and inference_device.type == "cuda",
        ):
            feature_maps = network.forward_feature_maps(encoder_inputs)

        if feature_maps.ndim != 4:
            raise RuntimeError(
                "Expected network.forward_feature_maps() to return [B,D,H,W], "
                f"got {tuple(feature_maps.shape)}."
            )

        batch_count, _, grid_h, grid_w = feature_maps.shape
        tokens = feature_maps.flatten(2).transpose(1, 2)
        token_scores = scorer(tokens)
        grid_scores = token_scores.reshape(batch_count, 1, grid_h, grid_w)

        pixel_scores = F.interpolate(
            grid_scores.float(),
            size=(tiler.tile, tiler.tile),
            mode="bilinear",
            align_corners=False,
        )

        tile_score_maps.append(pixel_scores.to(stitch_device))

    crop_scores = torch.cat(tile_score_maps, dim=0)
    stitched = tiler.stitch(crop_scores, origins, image_sizes)

    if len(stitched) != 1:
        raise RuntimeError(f"Expected one stitched map, got {len(stitched)}.")

    score_map = stitched[0]

    if score_map.shape[0] != 1:
        raise RuntimeError(
            f"Expected a one-channel score map, got {tuple(score_map.shape)}."
        )

    return score_map[0].cpu()


@click.command()
@click.option(
    "--pca",
    "pca_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="PCA artifact produced by the fitting CLI.",
)
@click.option(
    "--checkpoint",
    "checkpoint_override",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional checkpoint override. By default, the checkpoint path is read "
        "from the PCA artifact."
    ),
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing input images.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory receiving raw score maps, previews, and a manifest.",
)
@click.option(
    "--score-type",
    type=click.Choice(["euclidean", "normal", "mahalanobis"], case_sensitive=False),
    default="euclidean",
    show_default=True,
    help=(
        "Euclidean residual norm, or regularized Mahalanobis norm in the "
        "selected PCA subspace. `normal` is an alias for `euclidean`."
    ),
)
@click.option(
    "--n-major-components",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of leading PCA components excluded from the score.",
)
@click.option(
    "--end-component",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Exclusive end of the selected PCA component range. The default uses "
        "all remaining components."
    ),
)
@click.option(
    "--min-eigenvalue-ratio",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=None,
    help=(
        "Optional tail truncation: discard components whose eigenvalue is below "
        "this fraction of the largest eigenvalue."
    ),
)
@click.option(
    "--relative-ridge",
    type=click.FloatRange(min=0.0),
    default=1e-6,
    show_default=True,
    help=(
        "Mahalanobis ridge relative to the largest PCA eigenvalue. Ignored for "
        "Euclidean scoring."
    ),
)
@click.option(
    "--normalize-by-dimension/--no-normalize-by-dimension",
    default=False,
    show_default=True,
    help="Divide the score by sqrt(number of selected PCA components).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Number of image tiles encoded at once.",
)
@click.option(
    "--device",
    default=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    show_default="cuda when available, otherwise cpu",
    help="Device used for feature extraction and token scoring.",
)
@click.option(
    "--stitch-device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Device used to allocate and stitch full-resolution score maps.",
)
@click.option(
    "--amp/--no-amp",
    default=True,
    show_default=True,
    help="Use automatic mixed precision for network feature extraction on CUDA.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Search the input directory recursively.",
)
@click.option(
    "--save-preview/--no-save-preview",
    default=True,
    show_default=True,
    help="Save percentile-scaled PNG previews in addition to raw NPY maps.",
)
@click.option(
    "--preview-percentiles",
    type=(click.FloatRange(min=0.0, max=100.0), click.FloatRange(min=0.0, max=100.0)),
    default=(1.0, 99.0),
    show_default=True,
    help="Lower and upper percentiles used only to display PNG previews.",
)
def main(
    pca_path: Path,
    checkpoint_override: Path | None,
    input_dir: Path,
    output_dir: Path,
    score_type: str,
    n_major_components: int,
    end_component: int | None,
    min_eigenvalue_ratio: float | None,
    relative_ridge: float,
    normalize_by_dimension: bool,
    batch_size: int,
    device: str,
    stitch_device: str,
    amp: bool,
    recursive: bool,
    save_preview: bool,
    preview_percentiles: tuple[float, float],
) -> None:
    """Compute stitched residual-PCA OOD score maps for a directory of images."""
    lower_percentile, upper_percentile = preview_percentiles

    if lower_percentile >= upper_percentile:
        raise click.ClickException(
            "The lower preview percentile must be smaller than the upper percentile."
        )

    inference_device = torch.device(device)
    requested_stitch_device = torch.device(stitch_device)

    if requested_stitch_device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("--stitch-device cuda requested, but CUDA is unavailable.")

    click.echo(f"Loading PCA artifact: {pca_path}")
    pca_payload = _load_torch_file(pca_path)

    pca_state = pca_payload.get("pca_state_dict")
    if not isinstance(pca_state, dict):
        raise click.ClickException("The PCA artifact has no valid `pca_state_dict`.")

    mean = _state_tensor(pca_state, "mean")
    components = _state_tensor(pca_state, "components")
    eigenvalues = _state_tensor(pca_state, "explained_variance")

    feature_dim = int(pca_payload.get("feature_dim", mean.numel()))
    normalize_embeddings = bool(pca_payload.get("normalize_embeddings", False))

    if mean.numel() != feature_dim:
        raise click.ClickException(
            f"PCA feature_dim={feature_dim}, but mean has {mean.numel()} values."
        )

    checkpoint_path = resolve_checkpoint_path(
        pca_path=pca_path,
        pca_payload=pca_payload,
        checkpoint_override=checkpoint_override,
    )

    click.echo(f"Loading segmentation checkpoint: {checkpoint_path}")
    model = LinearSemantic.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    model = model.to(inference_device)
    model.eval()

    network = model.network
    if not isinstance(network, LinearDecoder):
        raise click.ClickException(
            f"Expected model.network to be LinearDecoder, got {type(network).__name__}."
        )

    tiler = model.tiler
    if not isinstance(tiler, GridPadTiler):
        raise click.ClickException(
            "The reconstructed checkpoint must contain a GridPadTiler, "
            f"got {type(tiler).__name__ if tiler is not None else None}."
        )

    if components.shape[1] != feature_dim:
        raise click.ClickException(
            f"PCA components expect {components.shape[1]} features, "
            f"but the PCA artifact declares {feature_dim}."
        )

    scorer = ResidualPCAScorer(
        mean=mean,
        components=components,
        eigenvalues=eigenvalues,
        score_type=score_type,
        n_major_components=n_major_components,
        end_component=end_component,
        min_eigenvalue_ratio=min_eigenvalue_ratio,
        relative_ridge=relative_ridge,
        normalize_embeddings=normalize_embeddings,
        normalize_by_dimension=normalize_by_dimension,
        device=inference_device,
    )

    images = discover_images(input_dir, recursive=recursive)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    previews_dir = output_dir / "previews"
    scores_dir.mkdir(parents=True, exist_ok=True)

    click.echo(
        f"Found {len(images):,} images. Using GridPadTiler(tile={tiler.tile}, "
        f"stride={tiler.stride}, weighted_blend={tiler.weighted_blend}, "
        f"pad_mode={tiler.pad_mode!r})."
    )
    click.echo(
        f"Score: {scorer.score_type}, components "
        f"[{scorer.n_major_components}:{scorer.end_component}] "
        f"({scorer.n_selected_components} dimensions)."
    )

    results: list[ImageResult] = []

    for image_index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(input_dir)
        relative_stem = relative_path.with_suffix("")

        click.echo(f"[{image_index}/{len(images)}] {relative_path}")

        image = load_rgb_image(image_path)
        score_map_tensor = score_image(
            image=image,
            network=network,
            tiler=tiler,
            scorer=scorer,
            batch_size=batch_size,
            inference_device=inference_device,
            stitch_device=requested_stitch_device,
            use_amp=amp,
        )

        score_map = score_map_tensor.numpy().astype(np.float32, copy=False)

        raw_path = scores_dir / relative_stem.parent / f"{relative_stem.name}.npy"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, score_map)

        preview_path: Path | None = None
        if save_preview:
            preview_path = (
                previews_dir
                / relative_stem.parent
                / f"{relative_stem.name}.png"
            )
            save_score_preview(
                score_map=score_map,
                path=preview_path,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )

        finite = score_map[np.isfinite(score_map)]
        if finite.size == 0:
            raise RuntimeError(f"Score map for {image_path} contains no finite values.")

        results.append(
            ImageResult(
                image=str(relative_path),
                score_path=str(raw_path.relative_to(output_dir)),
                preview_path=(
                    str(preview_path.relative_to(output_dir))
                    if preview_path is not None
                    else None
                ),
                height=int(score_map.shape[0]),
                width=int(score_map.shape[1]),
                score_min=float(finite.min()),
                score_max=float(finite.max()),
                score_mean=float(finite.mean()),
                score_std=float(finite.std()),
            )
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    run_config = {
        "pca_path": str(pca_path.resolve()),
        "segmentation_checkpoint": str(checkpoint_path),
        "input_dir": str(input_dir.resolve()),
        "n_images": len(images),
        "feature_dim": feature_dim,
        "score_type": scorer.score_type,
        "n_major_components": scorer.n_major_components,
        "end_component": scorer.end_component,
        "n_selected_components": scorer.n_selected_components,
        "min_eigenvalue_ratio": min_eigenvalue_ratio,
        "relative_ridge": relative_ridge,
        "normalize_embeddings": normalize_embeddings,
        "normalize_by_dimension": normalize_by_dimension,
        "tiler": {
            "tile": tiler.tile,
            "stride": tiler.stride,
            "weighted_blend": tiler.weighted_blend,
            "pad_mode": tiler.pad_mode,
            "pad_value": tiler.pad_value,
        },
        "batch_size": batch_size,
        "device": str(inference_device),
        "stitch_device": str(requested_stitch_device),
        "amp": amp,
        "preview_percentiles": [lower_percentile, upper_percentile],
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    click.echo(f"Saved {len(results):,} raw score maps under {scores_dir}.")
    if save_preview:
        click.echo(f"Saved visualization previews under {previews_dir}.")
    click.echo(f"Saved manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
