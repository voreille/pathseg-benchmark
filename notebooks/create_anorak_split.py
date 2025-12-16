#!/usr/bin/env python3
"""
Create 5-fold CV splits for segmentation masks, trying to balance
the number of pixels per class across folds.

Result: one CSV with ONE ROW PER (image_id, fold) pair:

image_id,fold,is_train,is_val,is_test
train001_Da382,0,False,True,True
train002_Da425,0,True,False,False
...
train001_Da382,1,True,False,False
...

For now: is_val == is_test (same images).
Later you can change the logic to separate val/test.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
MASKS_DIR = Path(
    "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x/mask")
IMG_DIR = Path(
    "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x/image")
OUTPUT_CSV = Path(
    "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK_10x/split_df.csv"
)

NUM_CLASSES = 7  # labels 0..6
IGNORE_BACKGROUND = True  # if True, class 0 is ignored for balancing
NUM_FOLDS = 5


# -----------------------------
# Step 1: Compute per-image class counts
# -----------------------------
def compute_class_counts(
    masks_dir: Path,
    img_dir: Path,
    num_classes: int = 7,
) -> pd.DataFrame:
    """
    Walk through masks_dir, load each PNG, and compute pixel counts per class.

    Returns a DataFrame with columns:
      - image_id
      - path
      - count_0, ..., count_(num_classes-1)
    """
    rows = []
    img_ids = sorted([f.stem for f in img_dir.rglob("*.png")])
    mask_paths = sorted(
        [f for f in masks_dir.rglob("*.png") if f.stem in img_ids])

    if not mask_paths:
        raise RuntimeError(f"No PNG masks found under {masks_dir}")

    for path in mask_paths:
        image_id = path.stem  # customize if needed

        img = Image.open(path)
        arr = np.array(img)

        # If mask is RGB for some reason, take first channel (adapt if needed)
        if arr.ndim == 3:
            arr = arr[..., 0]

        # Ensure integer type
        if not np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32)

        flat = arr.reshape(-1)
        counts = np.bincount(flat, minlength=num_classes)

        row = {
            "image_id": image_id,
            "path": str(path),
        }
        for c in range(num_classes):
            row[f"count_{c}"] = int(counts[c])

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Step 2: Greedy fold assignment (pixel-balanced)
# -----------------------------
def assign_folds_greedy_pixel_balance(
    df: pd.DataFrame,
    num_folds: int,
    num_classes: int,
    ignore_background: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Assign each image a single "base fold" trying to balance the total pixels
    per class across folds.

    Strategy:
      - We choose some set of classes to balance (e.g., 1..6 if ignoring 0).
      - Compute total pixels per class over all images, then a target per fold.
      - Sort images by total pixels of those classes (descending).
      - For each image, try assigning it to each fold and pick the fold that
        minimizes the "excess over target" cost.

    Returns:
      - df with a new column "fold" (this is the validation fold for that image)
      - fold_sums: [num_folds, num_balanced_classes] array with pixel sums
    """

    # Classes to consider for balancing
    if ignore_background:
        class_indices = list(range(1, num_classes))
    else:
        class_indices = list(range(num_classes))

    class_cols = [f"count_{c}" for c in class_indices]

    # Total pixels per class over the whole dataset
    totals = df[class_cols].sum(axis=0).values.astype(float)  # shape [C]
    targets = totals / float(num_folds)  # target per fold per class

    # Initialize per-fold class sums
    num_balanced_classes = len(class_indices)
    fold_sums = np.zeros((num_folds, num_balanced_classes), dtype=float)

    # We'll assign a fold index to each row
    folds = np.full(len(df), -1, dtype=int)

    # Sort images by total pixels of the balanced classes (descending)
    total_per_image = df[class_cols].sum(axis=1).values
    sorted_indices = np.argsort(-total_per_image)  # largest first

    for idx in sorted_indices:
        counts = df.loc[idx, class_cols].values.astype(float)

        best_fold = None
        best_cost = None

        for f in range(num_folds):
            new_sums = fold_sums[f] + counts

            # Cost: sum of positive excess beyond the target
            excess = np.maximum(0.0, new_sums - targets)
            cost = excess.sum()

            # Tiny regularization on total pixels to spread overall load
            total_pixels_fold = fold_sums[f].sum()
            cost += 1e-6 * total_pixels_fold

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_fold = f

        folds[idx] = best_fold
        fold_sums[best_fold] += counts

    df = df.copy()
    df["fold"] = folds
    return df, fold_sums


# -----------------------------
# Step 3: Build (image, fold) matrix CSV
# -----------------------------
def build_cv_matrix(df_with_folds: pd.DataFrame, num_folds: int,
                    output_csv: Path) -> pd.DataFrame:
    """
    Build a CSV with one row per (image_id, fold) pair.

    For each image:
      - df_with_folds["fold"] is the "validation fold" for that image.
      - For that fold: is_val = is_test = True, is_train = False
      - For all other folds: is_train = True, is_val = is_test = False

    Columns:
      image_id, fold, is_train, is_val, is_test
    """
    rows = []

    for _, row in df_with_folds.iterrows():
        image_id = row["image_id"]
        base_fold = int(row["fold"])  # this is the fold where it's used as val

        for f in range(num_folds):
            is_val = (f == base_fold)
            is_test = is_val  # for now, val == test
            is_train = not is_val

            rows.append({
                "image_id": image_id,
                "fold": f,
                "is_train": bool(is_train),
                "is_val": bool(is_val),
                "is_test": bool(is_test),
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"Saved CV matrix CSV to: {output_csv}")
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Scanning masks in: {MASKS_DIR}")
    df = compute_class_counts(MASKS_DIR, IMG_DIR, num_classes=NUM_CLASSES)
    print(f"Found {len(df)} masks")

    df_with_folds, fold_sums = assign_folds_greedy_pixel_balance(
        df,
        num_folds=NUM_FOLDS,
        num_classes=NUM_CLASSES,
        ignore_background=IGNORE_BACKGROUND,
    )

    # Optional: print global per-fold class sums (only balanced classes)
    balanced_cols = [
        c for c in df_with_folds.columns if c.startswith("count_")
    ]
    if IGNORE_BACKGROUND:
        balanced_cols = [c for c in balanced_cols if c != "count_0"]

    print("\nPer-fold pixel sums for balanced classes:")
    for f in range(NUM_FOLDS):
        idx = df_with_folds["fold"] == f
        sums = df_with_folds.loc[idx, balanced_cols].sum()
        print(f"\nFold {f}:")
        print(sums)

    # Build the big CSV: one row per (image_id, fold)
    build_cv_matrix(df_with_folds, NUM_FOLDS, OUTPUT_CSV)


if __name__ == "__main__":
    main()
