from pathlib import Path

import openslide
import pandas as pd
from tqdm import tqdm


cptac_slides_dir = Path("/mnt/nas6/data/CPTAC/CPTAC-LUAD_v12/LUAD")
clinical_csv = Path("/mnt/nas6/data/CPTAC/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv")
output_dir = Path("/home/valentin/workspaces/pathseg-benchmark/data/cptac_pattern_thumbnails")

slide_id_col = "Slide_ID"
histology_col = "Tumor_Histological_Type"

pattern_map_cptac = {
    "lepidic": ["Lepidic adenocarcinoma"],
    "acinar": [
        "Acinar adenocarcinoma",
        "Adenocarcinoma, acinar predominant",
        "Adenocarcinoma, acinic subtype",
        "Adenocarcinoma, acinar and papillary predominant.",
    ],
    "papillary": ["Papillary adenocarcinoma"],
    "micropapillary": ["Micropapillary adenocarcinoma"],
    "solid": ["Solid adenocarcinoma"],
}

thumbnail_max_size = (1024, 1024)

df_cptac = pd.read_csv(clinical_csv)

# Build Slide_ID -> path mapping
slide_paths = list(cptac_slides_dir.rglob("*.svs"))
slides_by_id = {p.stem: p for p in slide_paths}

thumbnail_jobs: list[tuple[Path, Path, str, str]] = []
missing_slides: list[tuple[str, str]] = []

for pattern, histology_values in pattern_map_cptac.items():
    df_subset = df_cptac[df_cptac[histology_col].isin(histology_values)]

    for _, row in df_subset.iterrows():
        slide_id = str(row[slide_id_col])

        src = slides_by_id.get(slide_id)
        if src is None:
            missing_slides.append((pattern, slide_id))
            continue

        dst = output_dir / pattern / f"{slide_id}.jpg"
        thumbnail_jobs.append((src, dst, pattern, slide_id))


print(f"Found {len(thumbnail_jobs)} thumbnails to generate.")
print(f"Slides missing on disk: {len(missing_slides)}")

if missing_slides:
    print("Examples of missing slides:")
    for pattern, slide_id in missing_slides[:10]:
        print(f"  {pattern}: {slide_id}")


def save_thumbnail(src: Path, dst: Path, max_size: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with openslide.OpenSlide(str(src)) as slide:
        thumb = slide.get_thumbnail(max_size)

        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")

        thumb.save(dst, format="JPEG", quality=90)


errors: list[tuple[str, str]] = []

for src, dst, _, slide_id in tqdm(thumbnail_jobs, desc="Generating thumbnails", unit="slide"):
    try:
        save_thumbnail(src, dst, thumbnail_max_size)
    except Exception as e:
        errors.append((slide_id, str(e)))

print("Done.")
print(f"Errors: {len(errors)}")

if errors:
    print("Examples of errors:")
    for slide_id, err in errors[:10]:
        print(f"  {slide_id}: {err}")