from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

pattern_to_class = {
    "label0": 0,
    "label1": 1,
    "label2": 2,
    "label3": 3,
    "label4": 4,
    "label5": 5,
    "label6": 6,
}


@click.command()
@click.option("--root_dir", default="", help="Root directory for the dataset.")
@click.option("--output_csv", default="", help="Path to save the class ratios CSV.")
def main(root_dir, output_csv):
    masks_dir = Path(root_dir) / "mask"
    images_dir = Path(root_dir) / "image"
    image_ids = [f.stem for f in images_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    mask_paths = [f.resolve() for f in masks_dir.glob("*.png") if f.stem in image_ids]

    ratios_list = []
    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        image_id = mask_path.stem

        try:
            # Read mask with PIL and convert to NumPy array
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as e:
            print(f"Error reading mask for {image_id}: {e}")
            continue

        # Count the proportion of pixels for each class
        pattern_dict = {f"{k}_ratio": np.mean(mask == v) for k, v in pattern_to_class.items()}

        ratios_list.append(
            {
                "image_id": image_id,
                "image_width": mask.shape[1],
                "image_height": mask.shape[0],
                "image_area": mask.shape[0] * mask.shape[1],
                **pattern_dict,
            }
        )

    ratio_df = pd.DataFrame(ratios_list)
    ratio_df.to_csv(output_csv, index=False)
    print(f"[OK] Saved class ratios to {output_csv}")


if __name__ == "__main__":
    main()
