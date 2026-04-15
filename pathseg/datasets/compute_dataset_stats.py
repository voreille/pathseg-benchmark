from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class DatasetPaths:
    root_dir: Path
    images_dir: Path
    masks_dir: Path
    label_map_path: Path
    output_dir: Path


def load_label_map(label_map_path: Path) -> dict[str, int]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict):
        raise ValueError(
            "label_map.json must contain a dict like {'class_name': 0, ...}"
        )

    if len(set(label_map.values())) != len(label_map):
        raise ValueError("label_map.json contains duplicated class ids.")

    return label_map


def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
    return {idx: name for name, idx in label_map.items()}


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel integer image: {mask_path}")
    return mask


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def resolve_paths(root_dir: Path, output_dir: Path | None) -> DatasetPaths:
    images_dir = root_dir / "images"
    masks_dir = root_dir / "masks_semantic"
    label_map_path = root_dir / "label_map.json"
    out_dir = output_dir if output_dir is not None else root_dir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise click.ClickException(f"Missing images directory: {images_dir}")
    if not masks_dir.exists():
        raise click.ClickException(f"Missing masks directory: {masks_dir}")
    if not label_map_path.exists():
        raise click.ClickException(f"Missing label_map.json: {label_map_path}")

    return DatasetPaths(
        root_dir=root_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        label_map_path=label_map_path,
        output_dir=out_dir,
    )


def ensure_matching_files(images_dir: Path, masks_dir: Path) -> list[str]:
    image_stems = {p.stem for p in images_dir.glob("*.png")}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}

    only_images = sorted(image_stems - mask_stems)
    only_masks = sorted(mask_stems - image_stems)

    if only_images:
        raise click.ClickException(
            f"Some images do not have a matching mask. Example: {only_images[:5]}"
        )
    if only_masks:
        raise click.ClickException(
            f"Some masks do not have a matching image. Example: {only_masks[:5]}"
        )

    return sorted(image_stems)


def pixel_area_mm2_from_mpp(mpp: float) -> float:
    if mpp <= 0:
        raise click.ClickException(f"mpp must be > 0, got {mpp}")
    return (mpp * 1e-3) ** 2


def apply_common_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_barplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 35,
    annotate: bool = True,
    fmt: str = ".3g",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    values = df[y_col].to_numpy()

    ax.bar(x, values, width=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=rotation, ha="right")

    if annotate and len(values) <= 20:
        ymax = max(values) if len(values) > 0 else 0
        offset = 0.01 * ymax if ymax > 0 else 0.01
        for i, v in enumerate(values):
            ax.text(i, v + offset, format(v, fmt), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 30,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if log_x:
        positive = values[values > 0]
        if len(positive) > 0:
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_per_class(
    df_per_image_class: pd.DataFrame,
    class_order: list[str],
    output_path: Path,
    title: str,
) -> None:
    pivot = (
        df_per_image_class.pivot(
            index="image_name",
            columns="class_name",
            values="area_fraction",
        )
        .fillna(0.0)
        .reindex(columns=class_order)
    )

    data = [pivot[col].to_numpy() for col in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=list(pivot.columns), showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Fraction of image area")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_violinplot_per_class(
    df_per_image_class: pd.DataFrame,
    class_order: list[str],
    output_path: Path,
    title: str,
) -> None:
    pivot = (
        df_per_image_class.pivot(
            index="image_name",
            columns="class_name",
            values="area_fraction",
        )
        .fillna(0.0)
        .reindex(columns=class_order)
    )

    data = [pivot[col].to_numpy() for col in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)

    ax.set_title(title)
    ax.set_ylabel("Fraction of image area")
    ax.set_xticks(np.arange(1, len(class_order) + 1))
    ax.set_xticklabels(class_order, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_per_image_class(
    df_per_image_class: pd.DataFrame,
    class_order: list[str],
    output_path: Path,
    max_images: int = 100,
) -> None:
    pivot = (
        df_per_image_class.pivot(
            index="image_name",
            columns="class_name",
            values="area_fraction",
        )
        .fillna(0.0)
        .reindex(columns=class_order)
    )

    row_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    if len(pivot) > max_images:
        pivot = pivot.iloc[:max_images]

    fig_h = max(5, min(18, 0.22 * len(pivot) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of image area")

    ax.set_title(f"Per-image class area fractions (first {len(pivot)} images)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_statistics(
    stems: list[str],
    images_dir: Path,
    masks_dir: Path,
    id_to_name: dict[int, str],
    pixel_area_mm2: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[int]]:
    class_ids = sorted(id_to_name.keys())

    area_mm2_per_class = {cid: 0.0 for cid in class_ids}
    pixels_per_class = {cid: 0 for cid in class_ids}
    image_count_per_class = {cid: 0 for cid in class_ids}

    per_image_rows: list[dict] = []
    per_image_class_rows: list[dict] = []

    unknown_labels_found: set[int] = set()

    with click.progressbar(stems, label="Processing files") as bar:
        for stem in bar:
            image_path = images_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"

            width, height = load_image_size(image_path)
            mask = load_mask(mask_path)

            if mask.shape != (height, width):
                raise click.ClickException(
                    f"Image/mask size mismatch for {stem}: "
                    f"image={(width, height)}, mask={(mask.shape[1], mask.shape[0])}"
                )

            total_pixels = int(mask.size)
            image_area_mm2 = total_pixels * pixel_area_mm2

            unique_labels, counts = np.unique(mask, return_counts=True)
            unique_labels = unique_labels.tolist()
            counts = counts.tolist()

            present_classes = 0
            dominant_class_id = None
            dominant_pixels = -1

            for label_id, count in zip(unique_labels, counts):
                label_id = int(label_id)
                count = int(count)

                if label_id not in id_to_name:
                    unknown_labels_found.add(label_id)
                    continue

                class_area_mm2 = count * pixel_area_mm2

                pixels_per_class[label_id] += count
                area_mm2_per_class[label_id] += class_area_mm2
                image_count_per_class[label_id] += 1
                present_classes += 1

                per_image_class_rows.append(
                    {
                        "image_name": stem,
                        "class_id": label_id,
                        "class_name": id_to_name[label_id],
                        "pixel_count": count,
                        "area_mm2": class_area_mm2,
                        "area_fraction": class_area_mm2 / image_area_mm2
                        if image_area_mm2 > 0
                        else 0.0,
                    }
                )

                if count > dominant_pixels:
                    dominant_pixels = count
                    dominant_class_id = label_id

            per_image_rows.append(
                {
                    "image_name": stem,
                    "width_px": width,
                    "height_px": height,
                    "area_px": total_pixels,
                    "area_mm2": image_area_mm2,
                    "num_classes_present": present_classes,
                    "dominant_class_id": dominant_class_id,
                    "dominant_class_name": (
                        id_to_name[dominant_class_id]
                        if dominant_class_id is not None
                        else None
                    ),
                }
            )

    n_images = len(per_image_rows)
    total_area_mm2 = sum(area_mm2_per_class.values())

    global_rows = []
    for cid in class_ids:
        global_rows.append(
            {
                "class_id": cid,
                "class_name": id_to_name[cid],
                "pixel_count": pixels_per_class[cid],
                "area_mm2": area_mm2_per_class[cid],
                "global_area_fraction": area_mm2_per_class[cid] / total_area_mm2
                if total_area_mm2 > 0
                else 0.0,
                "images_with_class": image_count_per_class[cid],
                "image_presence_fraction": image_count_per_class[cid] / n_images
                if n_images > 0
                else 0.0,
            }
        )

    df_global = pd.DataFrame(global_rows)
    df_per_image = pd.DataFrame(per_image_rows)
    df_per_image_class = pd.DataFrame(per_image_class_rows)

    return df_global, df_per_image, df_per_image_class, unknown_labels_found


def add_per_class_summary(
    df_per_image_class: pd.DataFrame, df_global: pd.DataFrame
) -> pd.DataFrame:
    summary = (
        df_per_image_class.groupby(["class_id", "class_name"])["area_fraction"]
        .agg(
            per_image_fraction_mean="mean",
            per_image_fraction_median="median",
            per_image_fraction_q25=lambda x: x.quantile(0.25),
            per_image_fraction_q75=lambda x: x.quantile(0.75),
            max_fraction_in_one_image="max",
        )
        .reset_index()
    )

    merged = df_global.merge(summary, on=["class_id", "class_name"], how="left").fillna(
        0.0
    )
    return merged


def write_summary_text(
    output_path: Path,
    mpp: float,
    pixel_area_mm2: float,
    df_global: pd.DataFrame,
    df_per_image: pd.DataFrame,
    unknown_labels_found: set[int],
) -> None:
    lines = [
        f"Number of images: {len(df_per_image)}",
        f"MPP used: {mpp}",
        f"Pixel area in mm^2: {pixel_area_mm2:.12f}",
        f"Total dataset area (mm^2): {df_per_image['area_mm2'].sum():.6f}",
        f"Median image width (px): {df_per_image['width_px'].median():.1f}",
        f"Median image height (px): {df_per_image['height_px'].median():.1f}",
        f"Median image area (mm^2): {df_per_image['area_mm2'].median():.6f}",
        "",
        "Class summary:",
    ]

    for _, row in df_global.iterrows():
        lines.append(
            f"- {row['class_name']}: "
            f"area={row['area_mm2']:.6f} mm^2, "
            f"global_fraction={row['global_area_fraction']:.4f}, "
            f"images_with_class={int(row['images_with_class'])}, "
            f"image_presence_fraction={row['image_presence_fraction']:.4f}, "
            f"median_per_image_fraction={row['per_image_fraction_median']:.4f}"
        )

    if unknown_labels_found:
        lines.extend(
            [
                "",
                f"Warning: labels found in masks but missing from label_map.json: {sorted(unknown_labels_found)}",
            ]
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_outputs(
    output_dir: Path,
    df_global: pd.DataFrame,
    df_per_image: pd.DataFrame,
    df_per_image_class: pd.DataFrame,
) -> None:
    df_global.to_csv(output_dir / "global_class_stats.csv", index=False)
    df_per_image.to_csv(output_dir / "per_image_stats.csv", index=False)
    df_per_image_class.to_csv(output_dir / "per_image_class_stats.csv", index=False)


def make_plots(
    output_dir: Path,
    df_global: pd.DataFrame,
    df_per_image: pd.DataFrame,
    df_per_image_class: pd.DataFrame,
) -> None:
    df_global_sorted = df_global.sort_values("area_mm2", ascending=False).reset_index(
        drop=True
    )
    class_order = df_global_sorted["class_name"].tolist()

    save_barplot(
        df=df_global_sorted,
        x_col="class_name",
        y_col="area_mm2",
        title="Global class distribution by area",
        ylabel="Area (mm²)",
        output_path=output_dir / "global_class_area_mm2.png",
        fmt=".2f",
    )

    save_barplot(
        df=df_global_sorted,
        x_col="class_name",
        y_col="global_area_fraction",
        title="Global class distribution by relative area",
        ylabel="Fraction of total annotated area",
        output_path=output_dir / "global_class_area_fraction.png",
        fmt=".3f",
    )

    save_barplot(
        df=df_global.sort_values(
            "image_presence_fraction", ascending=False
        ).reset_index(drop=True),
        x_col="class_name",
        y_col="images_with_class",
        title="Image-level class presence",
        ylabel="Number of images containing class",
        output_path=output_dir / "class_presence_across_images.png",
        fmt=".0f",
    )

    save_barplot(
        df=(
            df_per_image["dominant_class_name"]
            .value_counts()
            .rename_axis("class_name")
            .reset_index(name="num_images")
        ),
        x_col="class_name",
        y_col="num_images",
        title="Dominant class per image",
        ylabel="Number of images",
        output_path=output_dir / "dominant_class_per_image.png",
        fmt=".0f",
    )

    save_histogram(
        values=df_per_image["width_px"].to_numpy(),
        title="Image width distribution",
        xlabel="Width (px)",
        output_path=output_dir / "image_width_distribution.png",
    )

    save_histogram(
        values=df_per_image["height_px"].to_numpy(),
        title="Image height distribution",
        xlabel="Height (px)",
        output_path=output_dir / "image_height_distribution.png",
    )

    save_histogram(
        values=df_per_image["area_mm2"].to_numpy(),
        title="Image area distribution",
        xlabel="Area (mm²)",
        output_path=output_dir / "image_area_distribution_mm2.png",
        bins=30,
    )

    save_histogram(
        values=df_per_image["num_classes_present"].to_numpy(),
        title="Number of classes present per image",
        xlabel="Number of classes present",
        output_path=output_dir / "num_classes_present_per_image.png",
        bins=max(5, int(df_per_image["num_classes_present"].max())),
    )

    save_boxplot_per_class(
        df_per_image_class=df_per_image_class,
        class_order=class_order,
        output_path=output_dir / "per_image_class_fraction_boxplot.png",
        title="Per-image class area fraction distribution",
    )

    save_violinplot_per_class(
        df_per_image_class=df_per_image_class,
        class_order=class_order,
        output_path=output_dir / "per_image_class_fraction_violin.png",
        title="Per-image class area fraction distribution",
    )

    save_heatmap_per_image_class(
        df_per_image_class=df_per_image_class,
        class_order=class_order,
        output_path=output_dir / "per_image_class_fraction_heatmap.png",
        max_images=100,
    )


@click.command()
@click.argument(
    "root_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory where CSV files and plots will be saved. Default: <root_dir>/stats",
)
@click.option(
    "--mpp",
    type=float,
    default=0.5,
    show_default=True,
    help="Microns per pixel used to convert pixel counts into physical area in mm^2.",
)
def main(root_dir: Path, output_dir: Path | None, mpp: float) -> None:
    """
    Compute semantic segmentation dataset statistics for:

    ROOT_DIR/
      images/
      masks_semantic/
      label_map.json
    """
    apply_common_style()

    paths = resolve_paths(root_dir, output_dir)
    label_map = load_label_map(paths.label_map_path)
    id_to_name = invert_label_map(label_map)
    stems = ensure_matching_files(paths.images_dir, paths.masks_dir)
    pixel_area_mm2 = pixel_area_mm2_from_mpp(mpp)

    df_global, df_per_image, df_per_image_class, unknown_labels_found = (
        compute_statistics(
            stems=stems,
            images_dir=paths.images_dir,
            masks_dir=paths.masks_dir,
            id_to_name=id_to_name,
            pixel_area_mm2=pixel_area_mm2,
        )
    )

    df_global = add_per_class_summary(df_per_image_class, df_global)

    save_outputs(
        output_dir=paths.output_dir,
        df_global=df_global,
        df_per_image=df_per_image,
        df_per_image_class=df_per_image_class,
    )

    make_plots(
        output_dir=paths.output_dir,
        df_global=df_global,
        df_per_image=df_per_image,
        df_per_image_class=df_per_image_class,
    )

    write_summary_text(
        output_path=paths.output_dir / "summary.txt",
        mpp=mpp,
        pixel_area_mm2=pixel_area_mm2,
        df_global=df_global,
        df_per_image=df_per_image,
        unknown_labels_found=unknown_labels_found,
    )

    click.echo(f"Saved stats to: {paths.output_dir}")


if __name__ == "__main__":
    main()
