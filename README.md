# Semantic Segmentation Benchmark for Computational Pathology vFMs

This repository provides a benchmark for few-shot semantic segmentation in histopathology, with a focus on evaluating vision foundation models (vFMs) under episodic training and evaluation.

The benchmark is designed to:

* Support multiple heterogeneous segmentation datasets
* Standardize preprocessing across scanners and magnifications
* Make it easy to add new datasets with minimal boilerplate

---

## Project structure

### 1. Datasets and preprocessing

For each dataset included in the benchmark, a dedicated preprocessing CLI is provided under:

pathseg/preprocessing/<dataset_name>/prepare.py

Each CLI takes as input the raw data downloaded from the original source and outputs a standardized directory structure compatible with the benchmark.

Currently, preprocessing scripts accept:

* --target-magnification (e.g. 10x or 20x)

A global download-and-prepare script is intentionally not provided yet, as datasets are handled individually during early development.

---

### 2. Magnification and MPP handling

Different scanners may report different microns-per-pixel (MPP) values for the same nominal magnification.

We assume the following conventions:

* 20x ≈ 0.5 MPP
* 10x ≈ 1.0 MPP

During preprocessing, images are resampled to the closest achievable target MPP using integer downsampling factors when possible.

Example:
An image reported as 20x with MPP = 0.24 will be downsampled by a factor of 2 to reach approximately MPP = 0.48.

Original and target MPP values are stored in the dataset metadata.

---

### 3. Preprocessed data layout

The preprocessed data directory can be located anywhere on disk. In the examples below, we assume it is stored under repo_root/data/.
```
preprocessed_data_rootdir/
  datasets.csv
  datasets_index.parquet <- contains (dataset_id[str], valid_labels[int], maybe label_map[dict]) 
  class_index.parquet <- concatenation of each class_index (dataset_id, class_id, list of candidates for that class with sufficient min area containing the path of the image and mask, ...) 
  dataset_name/
    label_map.json
    metadata.csv
    class_index.parquet
    images/
      sample_001.png
      sample_002.png
    masks_semantic/
      sample_001.png
      sample_002.png

```
---

### 4. Annotations format
Annotations are expected to be stored as grayscale images uint8 as png or jpg.
A label_map.json (str to int) is provided to map the int values to a string values.

The label "Background" is reserverd and always mapped to 0.
The label "Ignore" is reserved and always mapped to 255 (usually unnatoted pixels as it is the case in IGNITE are mapped to this)

### 5. Dataset metadata

metadata.csv (one row per image)

This file is the primary entry point for dataset loading and episode construction. It is designed to be human-readable and easy to inspect.

Required fields:

* dataset_id: identifier of the dataset (typically the folder name)
* sample_id: unique identifier within the dataset
* image_relpath: relative path to the image file
* mask_relpath: relative path to the semantic mask
* width: image width in pixels
* height: image height in pixels
* mpp_x, mpp_y: microns per pixel
* magnification: nominal magnification (e.g. 10, 20)
* split: dataset split (e.g. train / val / test)

Additional dataset-specific columns are allowed.

---

### 6. Per-class annotations (derived)

To enable fast episode construction without repeatedly reading segmentation masks, per-image-per-class statistics are stored in a separate file.

annotations.parquet (one row per image x class)

Typical fields include:

* dataset_id
* sample_id
* dataset_class_id
* present
* area_ratio
* bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax

This file is generated during preprocessing and treated as a cache.

---

## Getting started

### 1. Download datasets

Downloading datasets is optional and depends on which benchmarks you intend to run.

### 2. Environment setup

conda create -n pathseg-benchmark python=3.10 -y
conda activate pathseg-benchmark

python -m pip install --upgrade pip

### 3. Install Pytorch

pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu123

Replace cu123 with your CUDA version (e.g. cu121, cu118).

### 3. Install this repo

pip install -e ".[parquet]"

or in dev mode

pip install -e ".[dev,parquet]"

---

## Training and evaluation

TODO


---

## Reproducing results

TODO

