from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd


REQUIRED_CLASS_COLS = {"dataset_id", "sample_id", "dataset_class_id"}
REQUIRED_META_COLS = {"dataset_id", "sample_id", "image_relpath", "mask_relpath"}


def _read_metadata(dataset_dir: Path) -> pd.DataFrame:
    p_parquet = dataset_dir / "metadata.parquet"
    p_csv = dataset_dir / "metadata.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Missing metadata.(parquet|csv) in {dataset_dir}")


def _read_class_index(dataset_dir: Path) -> pd.DataFrame:
    p = dataset_dir / "class_index.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing class_index.parquet in {dataset_dir}")
    return pd.read_parquet(p)


def _read_label_map(dataset_dir: Path) -> Dict[str, int]:
    p = dataset_dir / "label_map.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing label_map.json in {dataset_dir}")
    with open(p, "r") as f:
        lm = json.load(f)
    if not isinstance(lm, dict):
        raise ValueError(f"label_map.json must be dict[str,int] in {dataset_dir}")
    out: Dict[str, int] = {}
    for k, v in lm.items():
        if not isinstance(v, int):
            raise ValueError(f"Non-int label id for key={k!r} in {p}")
        out[str(k)] = int(v)
    return out


def _dataset_id_from_folder(dataset_dir: Path) -> str:
    return dataset_dir.name


def _ensure_dataset_id(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    if "dataset_id" not in df.columns:
        df = df.copy()
        df["dataset_id"] = dataset_id
        return df

    # If present but inconsistent, fail fast (keeps data clean)
    unique = set(str(x) for x in df["dataset_id"].dropna().unique().tolist())
    if len(unique) == 0:
        df = df.copy()
        df["dataset_id"] = dataset_id
        return df
    if len(unique) == 1 and next(iter(unique)) == dataset_id:
        return df
    raise ValueError(
        f"Inconsistent dataset_id in {dataset_id}: found {sorted(unique)}. "
        "Fix your per-dataset files or rename the folder."
    )


def _maybe_add_paths_from_metadata(
    class_df: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure class_df contains image_relpath and mask_relpath by merging from metadata if needed.
    """
    need = {"image_relpath", "mask_relpath"}
    if need.issubset(class_df.columns):
        return class_df

    missing = need - set(class_df.columns)
    if missing:
        # Require metadata to have the needed columns
        miss_meta = ({"dataset_id", "sample_id"} | need) - set(meta_df.columns)
        if miss_meta:
            raise ValueError(
                f"metadata missing required columns for merge: {miss_meta}"
            )

        # Merge by (dataset_id, sample_id)
        keep_cols = ["dataset_id", "sample_id", "image_relpath", "mask_relpath"]
        merged = class_df.merge(
            meta_df[keep_cols], on=["dataset_id", "sample_id"], how="left"
        )

        if merged["image_relpath"].isna().any() or merged["mask_relpath"].isna().any():
            bad = merged[
                merged["image_relpath"].isna() | merged["mask_relpath"].isna()
            ][["dataset_id", "sample_id"]].head(10)
            raise ValueError(
                "Failed to add image/mask paths to class_index for some rows. "
                f"Examples:\n{bad}"
            )
        return merged

    return class_df


def _infer_valid_label_ids(label_map: Dict[str, int]) -> List[int]:
    return sorted(set(int(v) for v in label_map.values()))


@click.command()
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Preprocessed data root dir containing dataset subfolders.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write global indexes (default: root-dir).",
)
@click.option(
    "--include",
    type=str,
    default=None,
    help="Comma-separated list of dataset folder names to include (default: all).",
)
@click.option(
    "--exclude",
    type=str,
    default=None,
    help="Comma-separated list of dataset folder names to exclude.",
)
@click.option(
    "--write-preview-csv/--no-write-preview-csv",
    default=True,
    show_default=True,
    help="Write small human-readable CSV previews at root.",
)
@click.option(
    "--preview-rows",
    type=int,
    default=5000,
    show_default=True,
    help="How many rows to include in class_index_preview.csv.",
)
def main(
    root_dir: Path,
    out_dir: Optional[Path],
    include: Optional[str],
    exclude: Optional[str],
    write_preview_csv: bool,
    preview_rows: int,
) -> None:
    """
    Build global indexes from per-dataset prepared outputs.

    Expected per-dataset layout:
      root_dir/
        DATASET_A/
          label_map.json
          metadata.(parquet|csv)
          class_index.parquet
          images/...
          masks_semantic/...
        DATASET_B/
          ...

    Outputs written to out-dir (default: root_dir):
      class_index.parquet              (concatenated)
      datasets_index.parquet           (dataset-level metadata)
      datasets.csv                     (human readable)
      class_index_preview.csv          (human readable subset)
    """
    out_dir = out_dir or root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    include_set = set(x.strip() for x in include.split(",")) if include else None
    exclude_set = set(x.strip() for x in exclude.split(",")) if exclude else set()

    dataset_dirs: List[Path] = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        # Minimal signals of a prepared dataset
        if not (p / "class_index.parquet").exists():
            continue
        if not ((p / "metadata.parquet").exists() or (p / "metadata.csv").exists()):
            continue
        if not (p / "label_map.json").exists():
            continue
        dataset_dirs.append(p)

    if not dataset_dirs:
        raise click.ClickException(
            f"No prepared datasets found in {root_dir}. "
            "Expected dataset folders containing class_index.parquet, metadata.(parquet|csv), label_map.json."
        )

    all_class: List[pd.DataFrame] = []
    datasets_rows: List[Dict[str, Any]] = []

    for ds_dir in dataset_dirs:
        dataset_id = _dataset_id_from_folder(ds_dir)

        meta = _read_metadata(ds_dir)
        meta = _ensure_dataset_id(meta, dataset_id)

        class_df = _read_class_index(ds_dir)
        class_df = _ensure_dataset_id(class_df, dataset_id)

        # Ensure minimal columns exist
        miss = REQUIRED_CLASS_COLS - set(class_df.columns)
        if miss:
            raise click.ClickException(
                f"{ds_dir}: class_index.parquet missing columns {miss}"
            )

        # Ensure we have image/mask relpaths in class index (merge from metadata if needed)
        class_df = _maybe_add_paths_from_metadata(class_df, meta)

        # Normalize dtypes
        class_df = class_df.copy()
        class_df["sample_id"] = class_df["sample_id"].astype(str)
        class_df["dataset_class_id"] = class_df["dataset_class_id"].astype(int)
        class_df["image_relpath"] = class_df["image_relpath"].astype(str)
        class_df["mask_relpath"] = class_df["mask_relpath"].astype(str)

        # Attach dataset_dir_rel so downstream can locate quickly if you want
        # (optional; comment out if you prefer pure relative paths only)
        class_df["dataset_dir"] = dataset_id

        all_class.append(class_df)

        # Dataset-level summary
        label_map = _read_label_map(ds_dir)
        valid_ids = _infer_valid_label_ids(label_map)

        n_samples = int(meta["sample_id"].nunique())
        n_classes = int(class_df["dataset_class_id"].nunique())

        datasets_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_dir": dataset_id,
                "n_samples": n_samples,
                "n_classes_indexed": n_classes,
                "valid_label_ids": json.dumps(valid_ids),
                "label_map_path": str(Path(dataset_id) / "label_map.json"),
                "metadata_path": str(
                    Path(dataset_id)
                    / (
                        "metadata.parquet"
                        if (ds_dir / "metadata.parquet").exists()
                        else "metadata.csv"
                    )
                ),
                "class_index_path": str(Path(dataset_id) / "class_index.parquet"),
            }
        )

    global_class_index = pd.concat(all_class, ignore_index=True)

    # Write global parquet
    global_class_index.to_parquet(out_dir / "class_index.parquet", index=False)

    datasets_index = pd.DataFrame(datasets_rows).sort_values("dataset_id")
    datasets_index.to_parquet(out_dir / "datasets_index.parquet", index=False)

    if write_preview_csv:
        datasets_index.to_csv(out_dir / "datasets.csv", index=False)
        # Write a preview of class index (parquet stays the real artifact)
        global_class_index.head(int(preview_rows)).to_csv(
            out_dir / "class_index_preview.csv", index=False
        )

    click.echo(f"Found datasets: {[d.name for d in dataset_dirs]}")
    click.echo(
        f"Wrote: {out_dir / 'class_index.parquet'}  (rows={len(global_class_index)})"
    )
    click.echo(
        f"Wrote: {out_dir / 'datasets_index.parquet'}  (datasets={len(datasets_index)})"
    )
    if write_preview_csv:
        click.echo(f"Wrote: {out_dir / 'datasets.csv'}")
        click.echo(f"Wrote: {out_dir / 'class_index_preview.csv'}")


if __name__ == "__main__":
    main()
