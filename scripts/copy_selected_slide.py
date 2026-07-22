import shutil
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.option("--root-dir", default="", help="Root directory for the dataset.")
@click.option("--output-dir", default="", help="Path to save the class ratios CSV.")
@click.option("--slide-list", default="", help="")
def main(root_dir, output_dir, slide_list):
    """Simple CLI program to greet someone"""
    root_dir = Path(root_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(slide_list, "r") as f:
        for line in tqdm(f, desc="Processing slides"):
            slide_id = line.strip()
            if slide_id:
                matching_files = list(root_dir.rglob(f"*{slide_id}*.svs"))
                if not matching_files:
                    print(f"No files found for slide ID: {slide_id}")
                    continue
                elif len(matching_files) > 1:
                    print(
                        f"Multiple files found for slide ID: {slide_id}. Using the first one."
                    )

                src = matching_files[0]
                dst = output_dir / src.name
                shutil.copy(src, dst)


if __name__ == "__main__":
    main()
