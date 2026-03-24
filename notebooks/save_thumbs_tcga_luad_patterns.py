from pathlib import Path

import openslide
import pandas as pd
from tqdm import tqdm


tcga_slides_dir = Path("/mnt/nas7/data/TCGA_Lung_svs/LUAD")
clinical_tsv = Path(
    "/home/valentin/workspaces/pathseg-benchmark/data/clinical/"
    "luad_tcga_pan_can_atlas_2018_clinical_data.tsv"
)
output_dir = Path("/home/valentin/workspaces/pathseg-benchmark/data/tcga_pattern_thumbnails")

pattern_map_tcga = {
    "papillary": "Lung Papillary Adenocarcinoma",
    "acinar": "Lung Acinar Adenocarcinoma",
    "solid": "Lung Solid Pattern Predominant Adenocarcinoma",
    "micropapillary": "Lung Micropapillary Adenocarcinoma",
}

thumbnail_max_size = (1024, 1024)  # change if you want smaller/larger


df_tcga = pd.read_csv(clinical_tsv, sep="\t")

# Only diagnostic slides
slide_paths = [p for p in tcga_slides_dir.rglob("*.svs") if "-DX" in p.stem]

# Map patient_id -> list of slide paths
slides_by_patient: dict[str, list[Path]] = {}
for slide_path in slide_paths:
    patient_id = "-".join(slide_path.stem.split("-")[:3])
    slides_by_patient.setdefault(patient_id, []).append(slide_path)

thumbnail_jobs: list[tuple[Path, Path, str, str]] = []
missing_patients: list[tuple[str, str]] = []

for pattern, tumor_type in pattern_map_tcga.items():
    patient_ids = (
        df_tcga.loc[df_tcga["Tumor Type"] == tumor_type, "Patient ID"]
        .dropna()
        .unique()
    )

    for patient_id in patient_ids:
        patient_slides = slides_by_patient.get(patient_id, [])

        if not patient_slides:
            missing_patients.append((pattern, patient_id))
            continue

        pattern_out_dir = output_dir / pattern
        for src in patient_slides:
            dst = pattern_out_dir / f"{src.stem}.jpg"
            thumbnail_jobs.append((src, dst, pattern, patient_id))

print(f"Found {len(thumbnail_jobs)} thumbnails to generate.")
print(f"Patients with no matching diagnostic slide: {len(missing_patients)}")

if missing_patients:
    print("Examples of missing patients:")
    for pattern, patient_id in missing_patients[:10]:
        print(f"  {pattern}: {patient_id}")


def save_thumbnail(src: Path, dst: Path, max_size: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with openslide.OpenSlide(str(src)) as slide:
        thumb = slide.get_thumbnail(max_size)

        # Ensure RGB before saving as JPEG
        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")

        thumb.save(dst, format="JPEG", quality=90)


errors: list[tuple[Path, str]] = []

for src, dst, _, _ in tqdm(thumbnail_jobs, desc="Generating thumbnails", unit="slide"):
    try:
        save_thumbnail(src, dst, thumbnail_max_size)
    except Exception as e:
        errors.append((src, str(e)))

print("Done.")
print(f"Errors: {len(errors)}")

if errors:
    print("Examples of errors:")
    for src, err in errors[:10]:
        print(f"  {src}: {err}")