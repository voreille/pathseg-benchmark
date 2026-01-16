# Semantic Segmentation Benchmark for Computational Pathology (vFMs)

This repository provides a **benchmark framework for few-shot semantic segmentation in histopathology**, with a focus on evaluating **vision foundation models (vFMs)** under episodic training and evaluation.

The goal of this project is to make it easy to:

* Benchmark multiple heterogeneous histopathology segmentation datasets
* Standardize preprocessing across scanners, magnifications, and formats
* Add new datasets with minimal boilerplate
* Hand over the project smoothly to new contributors 👋

If you are taking over this codebase, **start by reading this README and then inspect the YAML config files** used with the Lightning CLI. Most behavior is controlled from configuration rather than hard-coded logic.

---

## Project structure (high-level)

```
pathseg/
  preprocessing/
    <dataset_name>/
      prepare.py        # Dataset-specific preprocessing CLI
  datasets/
  models/
  training/
configs/
  *.yaml               # Lightning CLI configs (main entry point)
```

The benchmark is driven by:

* **Dataset-specific preprocessing scripts**
* A **standardized preprocessed data layout**
* **Lightning CLI configuration files** for training and evaluation

---

## 1. Datasets and preprocessing

Each dataset included in the benchmark has its own preprocessing script:

```
pathseg/preprocessing/<dataset_name>/prepare.py
```

### Purpose of preprocessing

The preprocessing step converts raw, dataset-specific formats into a **common, standardized representation** that the rest of the benchmark relies on.

Responsibilities of preprocessing include:

* Resampling images to a target magnification / MPP
* Converting annotations to a common semantic mask format
* Computing dataset metadata and per-class statistics
* Writing outputs to the standardized directory layout (see below)

### CLI interface

Each `prepare.py` script exposes a small CLI. At minimum, it supports:

* `--raw-data-dir`: path to the original downloaded dataset
* `--output-dir`: destination directory for preprocessed data
* `--target-magnification`: typically `10` or `20` (WARNING: not implemented for now, ANORAK and IGNITE are 20x by default)

Example:

```
python pathseg/preprocessing/ignite/prepare.py \
  --raw-data-dir /path/to/raw/ignite \
  --output-dir /path/to/preprocessed/ignite \
  --target-magnification 20
```

⚠️ There is **no global download-and-prepare script** on purpose. Datasets are handled individually during early development.

---

## 2. Magnification and MPP handling

Different scanners may report different microns-per-pixel (MPP) values for the same nominal magnification.

We use the following conventions:

* **20x ≈ 0.5 MPP**
* **10x ≈ 1.0 MPP**

During preprocessing:

* Images are resampled to the closest achievable target MPP
* Integer downsampling factors are preferred when possible

Example:

> An image reported as 20x with MPP = 0.24 will be downsampled by a factor of 2 to reach ≈ 0.48 MPP.

Both the **original MPP** and the **target MPP** are stored in the dataset metadata for traceability.

---

## 3. Preprocessed data layout

The preprocessed data directory can live anywhere on disk. In examples below, we assume:

```
repo_root/data/
```

Standardized layout:

```
preprocessed_data_rootdir/
  dataset_name/
    label_map.json     <-- label map used by the benchmark
    src_label_map.json <-- Original label map of the data, it is remapped to satisfy Background: 0 and Ignore: 255
    metadata.csv
    class_index.parquet
    images/
      sample_001.png
      sample_002.png
    masks_semantic/
      sample_001.png
      sample_002.png
```

### Per-dataset files

Each dataset folder contains:

* `label_map.json`

  * Mapping from **class name (str)** → **class id (int)**

* `metadata.csv`

  * One row per image (see below)

* `class_index.parquet`

  * Per-class candidate regions for episodic sampling

* `images/`

  * RGB images

* `masks_semantic/`

  * Semantic segmentation masks

---

## 4. Annotation format

* Annotations are stored as **grayscale `uint8` images** (`.png` or `.jpg`)
* Pixel values correspond to class IDs

### Reserved labels

* **Background**

  * Always mapped to `0`

* **Ignore**

  * Always mapped to `255`
  * Typically used for unlabeled or ambiguous regions (e.g. in IGNITE)

These conventions are assumed throughout the codebase.

---

## 5. Dataset metadata (`metadata.csv`)

`metadata.csv` is the **primary entry point for dataset loading and episode construction**.

* One row per image
* Human-readable and easy to inspect/debug

### Required columns

* `dataset_id` – dataset identifier (usually the folder name)
* `sample_id` – unique ID within the dataset
* `image_relpath` – relative path to the image
* `mask_relpath` – relative path to the semantic mask
* `width` – image width in pixels
* `height` – image height in pixels
* `mpp_x`, `mpp_y` – microns per pixel
* `magnification` – nominal magnification (e.g. 10, 20)

Additional dataset-specific columns are allowed and ignored by default.

---

## Getting started

### 1. Download datasets

Dataset download is **optional** and depends on which benchmarks you plan to run.

---

### 2. Environment setup

```
conda create -n pathseg-benchmark python=3.10 -y
conda activate pathseg-benchmark

python -m pip install --upgrade pip
```

---

### 3. Install PyTorch

```
pip install torch==2.2.2 torchvision==0.17.2 \
  --extra-index-url https://download.pytorch.org/whl/cu123
```

Replace `cu123` with your CUDA version (e.g. `cu121`, `cu118`).

---

### 4. Install this repository

Editable install:

```
pip install -e ".[parquet]"
```

Development install:

```
pip install -e ".[dev,parquet]"
```

---

### 5. Prepare a dataset (example: IGNITE)

```
python pathseg/preprocessing/ignite/prepare.py \
  --raw-data-dir /path/to/raw/ignite \
  --output-dir /path/to/preprocessed/ignite
```

---

## Training and evaluation

Training and evaluation are driven by **PyTorch Lightning CLI**.

You typically run:

```
pathseg fit \
  -c configs/ignite_linear_semantic.yaml \
  --data.root=/path/to/preprocessed/ignite \
  --data.num_workers=32 \
  --model.freeze_encoder=False \
  --no_compile
```

### Important notes

* **Most logic is in the YAML config**, not in the code
* You are encouraged to **inspect and modify configs** before touching Python
* CLI arguments always override config values

If you are new to Lightning CLI, start here:

👉 [https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)

---

## Handover notes

If you are continuing development:

1. Start from an existing config (e.g. `ignite_linear_semantic.yaml`)
2. Check:

   * dataset class (`data.class_path`)
   * paths (`data.root`)
   * model encoder choice
   * tiling parameters
3. Add new datasets by:

   * Writing a new `preprocessing/<dataset>/prepare.py`
   * Adding a dataset class under `pathseg/datasets/`
   * Creating a new config YAML

This README + the configs should be enough to get productive quickly 🚀
