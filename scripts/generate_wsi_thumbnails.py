from pathlib import Path

import click
from tqdm import tqdm
import openslide


@click.command()
@click.option("--root-dir", default="", help="Root directory for the dataset.")
@click.option("--output-dir", default="", help="Path to save the class ratios CSV.")
@click.option(
    "--regex-filter", default="*DX*.svs", help="Regex filter for selecting WSI files."
)
@click.option(
    "--thumbnail-size", default=1024, help="Size of the generated thumbnails."
)
def main(root_dir, output_dir, regex_filter, thumbnail_size):
    files = list(Path(root_dir).rglob(regex_filter))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc="Processing WSI files"):
        try:
            slide = openslide.OpenSlide(str(file))
            thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
            thumbnail.save(output_dir / f"{file.stem}.png")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    main()
