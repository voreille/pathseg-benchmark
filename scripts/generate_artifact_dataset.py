"""Generate a balanced Histopath-C/OOD artefact dataset with pixel masks."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from PIL import Image

from pathseg.histopathc.transforms import (
    ArtifactResult,
    ArtifactTransform,
    build_default_transforms,
)


CORRUPTION_TYPES = tuple(build_default_transforms())

REQUIRED_METADATA_COLUMNS = {
    "dataset_id",
    "sample_id",
    "group",
    "image_relpath",
    "mask_relpath",
    "width",
    "height",
    "mpp_x",
    "mpp_y",
    "magnification",
    "split",
    "validation_fold",
    "patient_id",
    "roi_id",
    "source",
    "scanner",
    "histological_subtype",
}

ARTIFACT_METADATA_COLUMNS = (
    "is_ood",
    "ood_type",
    "artifact_mask_path",
    "artifact_soft_mask_path",
    "artifact_parent_name",
    "artifact_metadata_json",
    "variant",
)


def build_balanced_schedule(
    corruption_types: list[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Return a shuffled schedule whose type counts differ by at most one."""

    if count <= 0:
        return []
    if not corruption_types:
        raise ValueError("corruption_types cannot be empty")

    schedule = [
        corruption_types[index % len(corruption_types)] for index in range(count)
    ]
    rng.shuffle(schedule)
    return schedule


def resolve_input_path(
    data_root: Path,
    path_value: object,
    *,
    column_name: str,
    sample_id: str,
) -> Path:
    """Resolve and validate one metadata path."""

    if pd.isna(path_value) or not str(path_value).strip():
        raise ValueError(
            f"Empty {column_name!r} value for sample_id={sample_id!r}"
        )

    path = Path(str(path_value).strip()).expanduser()
    if not path.is_absolute():
        path = data_root / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Invalid {column_name!r} for sample_id={sample_id!r}: {path}"
        )
    return path


def resolve_metadata_paths(row: pd.Series, data_root: Path) -> tuple[Path, Path]:
    sample_id = str(row["sample_id"]).strip()
    return (
        resolve_input_path(
            data_root,
            row["image_relpath"],
            column_name="image_relpath",
            sample_id=sample_id,
        ),
        resolve_input_path(
            data_root,
            row["mask_relpath"],
            column_name="mask_relpath",
            sample_id=sample_id,
        ),
    )


def clean_metadata_row(
    row: pd.Series,
    *,
    image_path: Path,
    mask_path: Path,
) -> dict[str, Any]:
    normalized = row.to_dict()
    normalized.update(
        {
            "sample_id": str(row["sample_id"]).strip(),
            "image_relpath": str(image_path),
            "mask_relpath": str(mask_path),
            "split": str(row["split"]).strip().lower(),
            "is_ood": 0,
            "ood_type": "",
            "artifact_mask_path": "",
            "artifact_soft_mask_path": "",
            "artifact_parent_name": "",
            "artifact_metadata_json": "",
            "variant": "original",
        }
    )
    return normalized


def build_nnunet_overview_rows(
    metadata_rows: list[dict[str, Any]],
    *,
    task_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in metadata_rows:
        width = row.get("width", "")
        height = row.get("height", "")
        shape = ""
        area_mm2: float | str = ""

        if pd.notna(width) and pd.notna(height):
            shape = f"({height}, {width})"
            try:
                area_mm2 = (
                    float(width)
                    * float(height)
                    * float(row.get("mpp_x", ""))
                    * float(row.get("mpp_y", ""))
                    / 1_000_000.0
                )
            except (TypeError, ValueError):
                pass

        rows.append(
            {
                "name": str(row["sample_id"]).strip(),
                "task": task_name,
                "image_path": str(row["image_relpath"]),
                "annotation_path": str(row["mask_relpath"]),
                "split": row.get("split", ""),
                "patient_id": row.get("patient_id", ""),
                "roi_id": row.get("roi_id", ""),
                "source": row.get("source", ""),
                "scanner": row.get("scanner", ""),
                "histological_subtype": row.get("histological_subtype", ""),
                "stain": row.get("stain", "H&E"),
                "shape": shape,
                "area_mm2": area_mm2,
                "validation_fold": row.get("validation_fold", ""),
                "is_ood": row.get("is_ood", 0),
                "ood_type": row.get("ood_type", ""),
                "artifact_mask_path": row.get("artifact_mask_path", ""),
                "artifact_soft_mask_path": row.get(
                    "artifact_soft_mask_path", ""
                ),
                "artifact_parent_name": row.get("artifact_parent_name", ""),
                "artifact_metadata_json": row.get("artifact_metadata_json", ""),
                "variant": row.get("variant", "original"),
            }
        )

    return rows


def load_semantic_mask(
    mask_path: Path,
    *,
    expected_size: tuple[int, int],
) -> np.ndarray:
    """Load a 2-D semantic label image without changing label values."""

    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)

    if mask.ndim != 2:
        raise ValueError(
            f"Semantic mask {mask_path} must be 2-D, got shape {mask.shape}"
        )

    expected_width, expected_height = expected_size
    if mask.shape != (expected_height, expected_width):
        raise ValueError(
            f"Mask {mask_path} has shape {mask.shape}, expected "
            f"{(expected_height, expected_width)}"
        )
    return np.ascontiguousarray(mask)


def foreground_from_labels(
    labels: np.ndarray,
    *,
    background_label: int,
) -> np.ndarray:
    return np.asarray(labels != background_label, dtype=bool)


def _save_label_mask(mask: np.ndarray, path: Path) -> None:
    """Save integer labels without normalizing or converting them to RGB."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"label mask must be 2-D, got {array.shape}")

    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    elif not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"label mask must have an integer dtype, got {array.dtype}")

    # Pillow supports uint8 and uint16 PNGs directly. Promote unusual integer
    # dtypes to the smallest safe standard representation.
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum >= 0 and maximum <= 255:
        array = array.astype(np.uint8, copy=False)
    elif minimum >= 0 and maximum <= 65535:
        array = array.astype(np.uint16, copy=False)
    else:
        raise ValueError(
            f"cannot save label range [{minimum}, {maximum}] as a PNG mask"
        )

    Image.fromarray(array).save(path)


def save_artifact_result(
    result: ArtifactResult,
    *,
    sample_id: str,
    image_dir: Path,
    mask_dir: Path,
    soft_mask_dir: Path | None,
    transformed_target_dir: Path,
) -> tuple[Path, Path, Path | None, Path | None]:
    image_path = (image_dir / f"{sample_id}.png").resolve()
    mask_path = (mask_dir / f"{sample_id}.png").resolve()

    result.image.save(image_path)
    Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L").save(mask_path)

    soft_mask_path: Path | None = None
    if soft_mask_dir is not None:
        soft_mask_path = (soft_mask_dir / f"{sample_id}.png").resolve()
        soft_uint8 = np.clip(
            np.rint(result.soft_mask * 255.0), 0, 255
        ).astype(np.uint8)
        Image.fromarray(soft_uint8, mode="L").save(soft_mask_path)

    transformed_target_path: Path | None = None
    if result.target_mask is not None:
        transformed_target_path = (
            transformed_target_dir / f"{sample_id}.png"
        ).resolve()
        _save_label_mask(result.target_mask, transformed_target_path)

    return image_path, mask_path, soft_mask_path, transformed_target_path


def apply_transform(
    image: Image.Image,
    *,
    transform: ArtifactTransform,
    rng: np.random.Generator,
    semantic_mask_path: Path,
    use_segmentation_for_folds: bool,
    background_label: int,
) -> ArtifactResult:
    if transform.name != "tissue_fold":
        return transform(image, rng=rng, tissue_mask=None)

    labels = load_semantic_mask(
        semantic_mask_path,
        expected_size=image.size,
    )
    tissue_mask = (
        foreground_from_labels(labels, background_label=background_label)
        if use_segmentation_for_folds
        else None
    )

    # AddTissueFold accepts semantic_mask in addition to the shared protocol.
    return transform(
        image,
        rng=rng,
        tissue_mask=tissue_mask,
        semantic_mask=labels,
    )


def generate_artifact_dataset(
    metadata_csv: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    n_modify: int,
    corruption_types: list[str],
    *,
    seed: int = 42,
    split: str = "test",
    task_name: str = "he_tissue_segmentation",
    copy_clean_for_selected: bool = True,
    save_soft_masks: bool = True,
    use_segmentation_for_folds: bool = True,
    background_label: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    metadata_csv = Path(metadata_csv).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    transforms = build_default_transforms()
    unsupported = sorted(set(corruption_types) - set(transforms))
    if unsupported:
        raise ValueError(f"Unsupported corruption types: {unsupported}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "artifact_images"
    artifact_mask_dir = output_dir / "artifact_masks"
    soft_mask_dir = output_dir / "artifact_soft_masks" if save_soft_masks else None
    transformed_target_dir = output_dir / "transformed_segmentation_masks"
    clean_image_dir = output_dir / "clean_images"

    image_dir.mkdir(exist_ok=True)
    artifact_mask_dir.mkdir(exist_ok=True)
    transformed_target_dir.mkdir(exist_ok=True)
    if soft_mask_dir is not None:
        soft_mask_dir.mkdir(exist_ok=True)
    if copy_clean_for_selected:
        clean_image_dir.mkdir(exist_ok=True)

    metadata = pd.read_csv(metadata_csv)
    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            "The metadata CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata["sample_id"] = metadata["sample_id"].astype(str).str.strip()
    selected = metadata[
        metadata["split"].astype(str).str.strip().str.casefold()
        == split.strip().casefold()
    ].copy()

    if selected.empty:
        available = sorted(
            metadata["split"].dropna().astype(str).str.strip().unique().tolist()
        )
        raise RuntimeError(
            f"No samples found for split={split!r}; available splits: {available}"
        )

    schedule = build_balanced_schedule(corruption_types, n_modify, rng)
    selected_indices = rng.integers(0, len(selected), size=n_modify)

    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    resolved_paths: dict[str, tuple[Path, Path]] = {}

    for _, row in selected.iterrows():
        sample_id = str(row["sample_id"]).strip()
        image_path, mask_path = resolve_metadata_paths(row, data_root)
        resolved_paths[sample_id] = (image_path, mask_path)
        metadata_rows.append(
            clean_metadata_row(row, image_path=image_path, mask_path=mask_path)
        )

    for artifact_index, (row_index, corruption_type) in enumerate(
        zip(selected_indices, schedule, strict=True)
    ):
        row = selected.iloc[int(row_index)].copy()
        parent_id = str(row["sample_id"]).strip()
        source_image_path, semantic_mask_path = resolved_paths[parent_id]
        transform = transforms[corruption_type]

        # Each variant receives an independent deterministic RNG stream.
        variant_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        variant_rng = np.random.default_rng(variant_seed)

        with Image.open(source_image_path) as source_image:
            clean_image = source_image.convert("RGB")
            result = apply_transform(
                clean_image,
                transform=transform,
                rng=variant_rng,
                semantic_mask_path=semantic_mask_path,
                use_segmentation_for_folds=use_segmentation_for_folds,
                background_label=background_label,
            )

            artifact_sample_id = (
                f"{parent_id}__histopathc_{corruption_type}_{artifact_index:04d}"
            )
            (
                artifact_image_path,
                artifact_mask_path,
                soft_mask_path,
                transformed_target_path,
            ) = save_artifact_result(
                result,
                sample_id=artifact_sample_id,
                image_dir=image_dir,
                mask_dir=artifact_mask_dir,
                soft_mask_dir=soft_mask_dir,
                transformed_target_dir=transformed_target_dir,
            )

            if copy_clean_for_selected:
                clean_image_path = (clean_image_dir / f"{parent_id}.png").resolve()
                if not clean_image_path.exists():
                    clean_image.save(clean_image_path)
            else:
                clean_image_path = source_image_path

        metadata_json = json.dumps(
            {"variant_seed": variant_seed, **result.metadata},
            sort_keys=True,
        )

        artifact_row = row.to_dict()
        artifact_row.update(
            {
                "sample_id": artifact_sample_id,
                "image_relpath": str(artifact_image_path),
                "mask_relpath": str(
                    transformed_target_path
                    if transformed_target_path is not None
                    else semantic_mask_path
                ),
                "width": result.image.size[0],
                "height": result.image.size[1],
                "split": str(row["split"]).strip().lower(),
                "is_ood": 1,
                "ood_type": corruption_type,
                "artifact_mask_path": str(artifact_mask_path),
                "artifact_soft_mask_path": (
                    str(soft_mask_path) if soft_mask_path is not None else ""
                ),
                "artifact_parent_name": parent_id,
                "artifact_metadata_json": metadata_json,
                "variant": "artifact",
            }
        )
        metadata_rows.append(artifact_row)

        summary_rows.append(
            {
                "sample_id": artifact_sample_id,
                "artifact_parent_name": parent_id,
                "ood_type": corruption_type,
                "original_image_path": str(source_image_path),
                "clean_image_path": str(clean_image_path),
                "artifact_image_path": str(artifact_image_path),
                "original_segmentation_mask_path": str(semantic_mask_path),
                "segmentation_mask_path": str(
                    transformed_target_path
                    if transformed_target_path is not None
                    else semantic_mask_path
                ),
                "artifact_mask_path": str(artifact_mask_path),
                "artifact_soft_mask_path": (
                    str(soft_mask_path) if soft_mask_path is not None else ""
                ),
                "artifact_metadata_json": metadata_json,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_columns = [
        *metadata.columns.tolist(),
        *[
            column
            for column in ARTIFACT_METADATA_COLUMNS
            if column not in metadata.columns
        ],
    ]

    output_metadata_csv = output_dir / f"metadata_artifacts_{timestamp}.csv"
    pd.DataFrame(metadata_rows).reindex(columns=output_columns).to_csv(
        output_metadata_csv, index=False
    )

    nnunet_csv = output_dir / f"data_overview_artifacts_nnunet_{timestamp}.csv"
    pd.DataFrame(
        build_nnunet_overview_rows(metadata_rows, task_name=task_name)
    ).to_csv(nnunet_csv, index=False)

    summary_csv = output_dir / f"artifact_generation_summary_{timestamp}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    click.echo(f"Augmented metadata: {output_metadata_csv}")
    click.echo(f"nnU-Net overview: {nnunet_csv}")
    click.echo(f"Generation summary: {summary_csv}")
    click.echo(f"Generated variants: {len(summary_rows)}")
    click.echo(f"Artifact images: {image_dir}")
    click.echo(f"Artifact masks: {artifact_mask_dir}")
    click.echo(f"Transformed segmentation masks: {transformed_target_dir}")
    if soft_mask_dir is not None:
        click.echo(f"Soft masks: {soft_mask_dir}")

    click.echo("\nCorruption distribution:")
    if schedule:
        click.echo(pd.Series(schedule).value_counts().sort_index().to_string())
    else:
        click.echo("No corruptions generated")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-root",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--metadata-csv",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option("--n-modify", type=click.IntRange(min=0), default=100, show_default=True)
@click.option(
    "--corruption-type",
    "corruption_types",
    type=click.Choice(CORRUPTION_TYPES, case_sensitive=True),
    multiple=True,
    default=CORRUPTION_TYPES,
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--split", type=str, default="test", show_default=True)
@click.option(
    "--task-name",
    type=str,
    default="he_tissue_segmentation",
    show_default=True,
)
@click.option(
    "--no-copy-clean-for-selected",
    is_flag=True,
    help="Do not copy clean parent images into the output directory.",
)
@click.option(
    "--no-soft-masks",
    is_flag=True,
    help="Save only binary artefact masks.",
)
@click.option(
    "--fold-use-full-image",
    is_flag=True,
    help="Allow tissue folds anywhere instead of restricting them to non-background labels.",
)
@click.option(
    "--background-label",
    type=int,
    default=0,
    show_default=True,
    help="Background value in the semantic mask used to place tissue folds.",
)
def main(
    data_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    n_modify: int,
    corruption_types: tuple[str, ...],
    seed: int,
    split: str,
    task_name: str,
    no_copy_clean_for_selected: bool,
    no_soft_masks: bool,
    fold_use_full_image: bool,
    background_label: int,
) -> None:
    """Generate corrupted images plus binary and continuous artefact masks."""

    generate_artifact_dataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        output_dir=output_dir,
        n_modify=n_modify,
        corruption_types=list(corruption_types),
        seed=seed,
        split=split,
        task_name=task_name,
        copy_clean_for_selected=not no_copy_clean_for_selected,
        save_soft_masks=not no_soft_masks,
        use_segmentation_for_folds=not fold_use_full_image,
        background_label=background_label,
    )


if __name__ == "__main__":
    main()