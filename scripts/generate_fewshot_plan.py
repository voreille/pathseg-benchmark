import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def pick_one_patch_for_image(
    mask_path: Path, label: int, patch_size: int, max_tries: int = 100
):
    """
    Sample one patch_size x patch_size patch where 'label' is present with decent ratio.
    Returns (x0, y0).
    """
    mask = np.array(Image.open(mask_path))
    H, W = mask.shape[:2]

    ys, xs = np.where(mask == label)
    if len(xs) == 0:
        raise RuntimeError(f"No pixels of label={label} in {mask_path}")

    rng = np.random.default_rng()

    for _ in range(max_tries):
        idx = rng.integers(len(xs))
        cy, cx = ys[idx], xs[idx]

        x0 = int(np.clip(cx - patch_size // 2, 0, W - patch_size))
        y0 = int(np.clip(cy - patch_size // 2, 0, H - patch_size))

        patch = mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
        ratio = (patch == label).mean()
        if ratio >= 0.6:  # tweak threshold if needed
            return x0, y0

    # fallback: center on the first pixel
    cy, cx = ys[0], xs[0]
    x0 = int(np.clip(cx - patch_size // 2, 0, W - patch_size))
    y0 = int(np.clip(cy - patch_size // 2, 0, H - patch_size))
    return x0, y0


def main(
    root: Path,
    class_ratios_csv: Path,
    output_csv: Path,
    patch_size: int = 448,
    labels=(1, 2, 3, 4, 5, 6),
    pool_size_per_label: int = 30,
    n_support_sets: int = 3,
    n_shot: int = 10,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # Base dataframe: one row per image, from class_ratios_old.csv
    df = pd.read_csv(class_ratios_csv)

    # Sanity checks
    required_cols = ["image_id", "image_area"] + [f"label{i}_ratio" for i in labels]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in class_ratios_csv: {missing}")

    # New columns (same length as df => same #rows as images)
    df["primary_label"] = -1
    df["is_test"] = True
    df["support_set"] = -1  # 0,1,2 for support; -1 for test
    df["k_shot"] = -1  # 1..10 for support; -1 for test
    df["x0"] = -1
    df["y0"] = -1
    df["w"] = -1
    df["h"] = -1

    # Compute primary_label = argmax over 1..6
    ratio_cols = [f"label{i}_ratio" for i in labels]
    ratios = df[ratio_cols].to_numpy()
    primary_idx = np.argmax(ratios, axis=1)
    df["primary_label"] = np.array(labels)[primary_idx]

    masks_dir = root / "mask"

    # For each label: pick pool_size_per_label, then split into support sets and k_shot
    for c in labels:
        rcol = f"label{c}_ratio"
        sub = df[df["primary_label"] == c].copy()
        if sub.empty:
            print(f"[WARN] No images with primary_label={c}")
            continue

        sub["eff_pix"] = sub["image_area"] * sub[rcol]
        sub = sub.sort_values(by=[rcol, "eff_pix"], ascending=[False, False])

        # Take up to pool_size_per_label
        sub = sub.head(pool_size_per_label)
        idxs = sub.index.to_numpy()
        rng.shuffle(idxs)

        print(
            f"[INFO] Label {c}: {len(idxs)} candidates, "
            f"target {n_support_sets} sets of {n_shot} shots."
        )

        for s in range(n_support_sets):
            start = s * n_shot
            end = min(start + n_shot, len(idxs))
            if start >= len(idxs):
                break

            chosen = idxs[start:end]
            if len(chosen) == 0:
                continue

            # k_shot within this (label, support_set): 1..len(chosen)
            ks = np.arange(1, len(chosen) + 1)

            df.loc[chosen, "support_set"] = s
            df.loc[chosen, "k_shot"] = ks
            df.loc[chosen, "is_test"] = False

            # compute bbox for each chosen support image
            for row_idx in chosen:
                row = df.loc[row_idx]
                image_id = row["image_id"]
                mask_path = masks_dir / f"{image_id}.png"  # adapt extension if needed
                x0, y0 = pick_one_patch_for_image(
                    mask_path, label=c, patch_size=patch_size
                )
                df.loc[row_idx, "x0"] = x0
                df.loc[row_idx, "y0"] = y0
                df.loc[row_idx, "w"] = patch_size
                df.loc[row_idx, "h"] = patch_size

    n_train = (df["support_set"] >= 0).sum()
    n_test = (df["is_test"]).sum()
    print(
        f"[OK] total support images (train) = {n_train}, total test images = {n_test}"
    )
    print(f"[OK] total rows in output = {len(df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, required=True, help="Root folder with mask/ subdir"
    )
    parser.add_argument("--class_ratios_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=448)
    parser.add_argument("--pool_size_per_label", type=int, default=30)
    parser.add_argument("--n_support_sets", type=int, default=3)
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        root=args.root,
        class_ratios_csv=args.class_ratios_csv,
        output_csv=args.output_csv,
        patch_size=args.patch_size,
        pool_size_per_label=args.pool_size_per_label,
        n_support_sets=args.n_support_sets,
        n_shot=args.n_shot,
        seed=args.seed,
    )
