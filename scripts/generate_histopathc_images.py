from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
from typing import Any

import click
import numpy as np
import pandas as pd
from PIL import Image

from pathseg.histopathc.transforms import (
    AddAirBubble,
    AddDust,
    CorruptTransform,
    Staining,
)


CORRUPTION_TYPES = (
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "dust",
    "air_bubble",
    "stain_light",
    "stain_heavy",
)

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
    "artifact_parent_name",
    "variant",
)


# ============================================================
# Histopath-C transforms
# ============================================================


def build_histopathc_transforms() -> dict[str, Any]:
    return {
        "gaussian_noise": CorruptTransform("gaussian_noise", 3),
        "shot_noise": CorruptTransform("shot_noise", 3),
        "defocus_blur": CorruptTransform("defocus_blur", 4),
        "motion_blur": CorruptTransform("motion_blur", 5),
        "brightness": CorruptTransform("brightness", 2),
        "contrast": CorruptTransform("contrast", 1),
        "dust": AddDust(),
        "air_bubble": AddAirBubble(
            min_bubbles=10,
            max_bubbles=20,
            transparency=0.3,
            blur_severity=2,
        ),
        "stain_light": Staining(0.15),
        "stain_heavy": Staining(0.25),
    }


def build_balanced_corruption_schedule(
    corruption_types: list[str],
    n_modify: int,
    rng: random.Random,
) -> list[str]:
    """Create a shuffled schedule with approximately equal type frequencies."""
    if n_modify <= 0:
        return []

    if not corruption_types:
        raise ValueError("corruption_types cannot be empty.")

    base_count = n_modify // len(corruption_types)
    remainder = n_modify % len(corruption_types)

    schedule: list[str] = []
    for corruption_type in corruption_types:
        schedule.extend([corruption_type] * base_count)

    extra_types = list(corruption_types)
    rng.shuffle(extra_types)
    schedule.extend(extra_types[:remainder])
    rng.shuffle(schedule)

    return schedule


def apply_histopathc_corruption(
    image: Image.Image,
    corruption_type: str,
    transforms: dict[str, Any],
) -> tuple[Image.Image, str]:
    if corruption_type not in transforms:
        raise ValueError(
            f"Unsupported Histopath-C corruption: {corruption_type}"
        )

    image = image.convert("RGB")
    output_image = transforms[corruption_type](image).convert("RGB")
    corruption_note = f"histopathc_{corruption_type}"

    return output_image, corruption_note


# ============================================================
# Metadata and path helpers
# ============================================================


def resolve_input_path(
    data_root: Path,
    path_value: object,
    *,
    column_name: str,
    sample_id: str,
) -> Path:
    """Resolve an absolute or data-root-relative metadata path."""
    if pd.isna(path_value) or not str(path_value).strip():
        raise ValueError(
            f"Empty {column_name!r} value for sample_id={sample_id!r}."
        )

    path = Path(str(path_value).strip()).expanduser()
    if not path.is_absolute():
        path = data_root / path

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"File referenced by {column_name!r} does not exist for "
            f"sample_id={sample_id!r}: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Path referenced by {column_name!r} is not a file for "
            f"sample_id={sample_id!r}: {path}"
        )

    return path


def resolve_metadata_paths(row: pd.Series, data_root: Path) -> tuple[Path, Path]:
    sample_id = str(row["sample_id"]).strip()
    image_path = resolve_input_path(
        data_root,
        row["image_relpath"],
        column_name="image_relpath",
        sample_id=sample_id,
    )
    mask_path = resolve_input_path(
        data_root,
        row["mask_relpath"],
        column_name="mask_relpath",
        sample_id=sample_id,
    )
    return image_path, mask_path


def normalize_metadata_row(
    row: pd.Series,
    *,
    image_path: Path,
    mask_path: Path,
) -> dict[str, Any]:
    """Convert one source row to the augmented output metadata schema."""
    normalized = row.to_dict()
    normalized["sample_id"] = str(row["sample_id"]).strip()
    normalized["image_relpath"] = str(image_path)
    normalized["mask_relpath"] = str(mask_path)
    normalized["split"] = str(row["split"]).strip().lower()
    normalized["is_ood"] = 0
    normalized["ood_type"] = ""
    normalized["artifact_mask_path"] = ""
    normalized["artifact_parent_name"] = ""
    normalized["variant"] = "original"
    return normalized


def build_nnunet_overview_rows(
    metadata_rows: list[dict[str, Any]],
    *,
    task_name: str,
) -> list[dict[str, Any]]:
    """Convert the augmented metadata rows to the legacy nnU-Net overview schema."""
    overview_rows: list[dict[str, Any]] = []

    for row in metadata_rows:
        width = row.get("width", "")
        height = row.get("height", "")

        shape = ""
        if pd.notna(width) and pd.notna(height):
            shape = f"({height}, {width})"

        area_mm2: float | str = ""
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

        overview_rows.append(
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
                "artifact_parent_name": row.get("artifact_parent_name", ""),
                "variant": row.get("variant", "original"),
            }
        )

    return overview_rows


# ============================================================
# Main dataset generation
# ============================================================


def generate_histopathc_dataset(
    metadata_csv: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    n_modify: int,
    corruption_types: list[str],
    seed: int = 42,
    split: str = "test",
    task_name: str = "he_tissue_segmentation",
    copy_clean_for_selected: bool = True,
) -> None:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    metadata_csv = Path(metadata_csv).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_image_dir = output_dir / "artifact_images"
    clean_image_dir = output_dir / "clean_images"
    artifact_image_dir.mkdir(exist_ok=True)

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
    selected_metadata = metadata[
        metadata["split"].astype(str).str.strip().str.casefold()
        == split.strip().casefold()
    ].copy()

    if selected_metadata.empty:
        available_splits = sorted(
            metadata["split"].dropna().astype(str).str.strip().unique().tolist()
        )
        raise RuntimeError(
            f"No samples found for split={split!r}. "
            f"Available splits: {available_splits}"
        )

    transforms = build_histopathc_transforms()
    unsupported = sorted(set(corruption_types) - set(transforms))
    if unsupported:
        raise ValueError(f"Unsupported corruption types: {unsupported}")

    corruption_schedule = build_balanced_corruption_schedule(
        corruption_types=corruption_types,
        n_modify=n_modify,
        rng=rng,
    )

    # Sampling with replacement allows n_modify to exceed the number of samples.
    selected_indices = [
        rng.randrange(len(selected_metadata)) for _ in range(n_modify)
    ]

    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    resolved_paths: dict[str, tuple[Path, Path]] = {}

    # Preserve every clean sample from the selected split in the output metadata.
    for _, row in selected_metadata.iterrows():
        sample_id = str(row["sample_id"]).strip()
        image_path, mask_path = resolve_metadata_paths(row, data_root)
        resolved_paths[sample_id] = (image_path, mask_path)
        metadata_rows.append(
            normalize_metadata_row(
                row,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    # Generate corrupted variants.
    for artifact_index, (row_index, corruption_type) in enumerate(
        zip(selected_indices, corruption_schedule)
    ):
        row = selected_metadata.iloc[row_index].copy()
        parent_sample_id = str(row["sample_id"]).strip()
        source_image_path, mask_path = resolved_paths[parent_sample_id]

        with Image.open(source_image_path) as source_image:
            clean_image = source_image.convert("RGB")
            artifact_image, corruption_note = apply_histopathc_corruption(
                image=clean_image,
                corruption_type=corruption_type,
                transforms=transforms,
            )

            artifact_sample_id = (
                f"{parent_sample_id}__histopathc_"
                f"{corruption_type}_{artifact_index:04d}"
            )
            artifact_image_path = (
                artifact_image_dir / f"{artifact_sample_id}.png"
            ).resolve()
            artifact_image.save(artifact_image_path)

            if copy_clean_for_selected:
                clean_image_path = (
                    clean_image_dir / f"{parent_sample_id}.png"
                ).resolve()
                if not clean_image_path.exists():
                    clean_image.save(clean_image_path)
            else:
                clean_image_path = source_image_path

        artifact_row = row.to_dict()
        artifact_row["sample_id"] = artifact_sample_id
        artifact_row["image_relpath"] = str(artifact_image_path)
        artifact_row["mask_relpath"] = str(mask_path)
        artifact_row["split"] = str(row["split"]).strip().lower()
        artifact_row["is_ood"] = 1
        artifact_row["ood_type"] = corruption_type
        artifact_row["artifact_mask_path"] = ""
        artifact_row["artifact_parent_name"] = parent_sample_id
        artifact_row["variant"] = "artifact"
        metadata_rows.append(artifact_row)

        summary_rows.append(
            {
                "sample_id": artifact_sample_id,
                "artifact_parent_name": parent_sample_id,
                "ood_type": corruption_type,
                "artifact_note": corruption_note,
                "original_image_path": str(source_image_path),
                "clean_image_path": str(clean_image_path),
                "artifact_image_path": str(artifact_image_path),
                "mask_path": str(mask_path),
                "artifact_mask_path": "",
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_metadata_csv = output_dir / f"metadata_artifacts_{timestamp}.csv"
    output_columns = [
        *metadata.columns.tolist(),
        *[
            column
            for column in ARTIFACT_METADATA_COLUMNS
            if column not in metadata.columns
        ],
    ]
    pd.DataFrame(metadata_rows).reindex(columns=output_columns).to_csv(
        output_metadata_csv,
        index=False,
    )

    nnunet_overview_csv = (
        output_dir / f"data_overview_artifacts_nnunet_{timestamp}.csv"
    )
    pd.DataFrame(
        build_nnunet_overview_rows(metadata_rows, task_name=task_name)
    ).to_csv(nnunet_overview_csv, index=False)

    summary_csv = output_dir / f"artifact_generation_summary_{timestamp}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    click.echo(f"Augmented metadata saved to: {output_metadata_csv}")
    click.echo(f"nnU-Net overview saved to: {nnunet_overview_csv}")
    click.echo(f"Generation summary saved to: {summary_csv}")
    click.echo(f"Generated Histopath-C images: {len(summary_rows)}")
    click.echo(f"Artifact image directory: {artifact_image_dir}")

    click.echo("\nCorruption distribution:")
    if corruption_schedule:
        click.echo(
            pd.Series(corruption_schedule).value_counts().sort_index().to_string()
        )
    else:
        click.echo("No corruptions generated.")


# ============================================================
# CLI
# ============================================================


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-root",
    "--data_root",
    "data_root",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
    help=(
        "Dataset root used to resolve relative image_relpath and mask_relpath "
        "values from the metadata CSV."
    ),
)
@click.option(
    "--metadata-csv",
    "--data-overview-csv",
    "--data_overview_csv",
    "metadata_csv",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
    help="Input metadata CSV using the foundation-style schema.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    "output_dir",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        path_type=Path,
    ),
    required=True,
    help="Directory in which generated images and CSV files are written.",
)
@click.option(
    "--n-modify",
    "--n_modify",
    "n_modify",
    type=click.IntRange(min=0),
    default=100,
    show_default=True,
    help="Number of corrupted image variants to generate.",
)
@click.option(
    "--corruption-type",
    "--corruption-types",
    "--corruption_types",
    "corruption_types",
    type=click.Choice(CORRUPTION_TYPES, case_sensitive=True),
    multiple=True,
    default=CORRUPTION_TYPES,
    show_default=True,
    help=(
        "Corruption type to use. Repeat this option to select multiple types, "
        "for example: --corruption-type dust --corruption-type motion_blur."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed.",
)
@click.option(
    "--split",
    type=str,
    default="test",
    show_default=True,
    help="Metadata split from which images are selected.",
)
@click.option(
    "--task-name",
    "--task_name",
    "task_name",
    type=str,
    default="he_tissue_segmentation",
    show_default=True,
    help="Task name written to the generated nnU-Net overview CSV.",
)
@click.option(
    "--no-copy-clean-for-selected",
    "--no_copy_clean_for_selected",
    "no_copy_clean_for_selected",
    is_flag=True,
    help="Do not copy clean versions of selected images.",
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
) -> None:
    """Generate a Histopath-C corruption dataset and augmented metadata."""
    generate_histopathc_dataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        output_dir=output_dir,
        n_modify=n_modify,
        corruption_types=list(corruption_types),
        seed=seed,
        split=split,
        task_name=task_name,
        copy_clean_for_selected=not no_copy_clean_for_selected,
    )


if __name__ == "__main__":
    main()