# Semantic Segmentation Benchmark for Computational Pathology (vFMs)

This repository provides a **benchmark framework for few-shot semantic segmentation in histopathology**, with a focus on evaluating **vision foundation models (vFMs)** under episodic training and evaluation.

The goal of this project is to make it easy to:

* Benchmark multiple heterogeneous histopathology segmentation datasets
* Standardize preprocessing across scanners, magnifications, and formats
* Add new datasets with minimal boilerplate
* Hand over the project smoothly to new contributors 👋

If you are taking over this codebase, **start by reading this README and then inspect the YAML config files** used with the Lightning CLI. Most behavior is controlled from configuration rather than hard-coded logic.

---
## TODO
### IGNITE
- [ ] Run training on all folds    (Currently using the original folds provided with the dataset; some folds contain 0 samples for certain classes, e.g. Muscle.   Options: discard problematic folds or redefine a stratified split.)
- [ ] Store model checkpoints cleanly for reuse (naming + directory convention)
- [ ] Perform clean test-set evaluation  (e.g. ensemble over fold-specific models)
- [ ] Implement resampling to 10x if needed

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
* `--target-magnification`: typically `10` or `20` 
  ⚠️ Currently **not implemented** — ANORAK and IGNITE are fixed at 20x.

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

PathSeg is tested with Python 3.12.

Create and activate a dedicated Conda environment:

```bash
conda create -n pathseg-benchmark python=3.12 pip -y
conda activate pathseg-benchmark
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

### 3. Install PyTorch

Install PyTorch 2.13.0 and TorchVision 0.28.0 with the CUDA 12.6 runtime:

```bash
python -m pip install \
  torch==2.13.0 \
  torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

The CUDA runtime bundled with PyTorch does not have to match the CUDA version displayed by `nvidia-smi` exactly. The NVIDIA driver must support the selected runtime.

---

### 4. Install Faiss GPU

Install the CUDA 12 Faiss wheel:

```bash
python -m pip install \
  faiss-gpu-cu12==1.14.1.post1
```

Install PyTorch before Faiss so that both packages resolve against a compatible CUDA 12.x runtime.

Do not use the `fix-cuda` extra here:

```text
faiss-gpu-cu12[fix-cuda]
```

That extra fixes the Faiss CUDA dependencies to CUDA 12.1, whereas this environment uses the CUDA 12.6 libraries installed with PyTorch.

Do not install multiple Faiss distributions in the same environment. In particular, avoid combining `faiss-gpu-cu12` with:

```text
faiss-gpu
faiss-gpu-cuvs
faiss-cpu
```

The `faiss-gpu-cu12` package is a third-party GPU wheel. The upstream Faiss project officially supports its Conda packages, but the available Conda builds may not resolve cleanly for every Python and CUDA combination.

---

### 5. Install PathSeg

Move to the repository directory:

```bash
cd /path/to/pathseg
```

Install PathSeg with Parquet support:

```bash
python -m pip install -e ".[parquet]"
```

For development:

```bash
python -m pip install -e ".[dev,parquet]"
```

Verify that the installed package requirements are consistent:

```bash
python -m pip check
```

---

### 6. Verify the GPU environment

```bash
python - <<'PY'
import torch
import torchvision
import faiss
import numpy as np

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("PyTorch CUDA available:", torch.cuda.is_available())
print("PyTorch GPU count:", torch.cuda.device_count())

print("Faiss:", faiss.__version__)
print("Faiss GPU count:", faiss.get_num_gpus())

print("NumPy:", np.__version__)
PY
```

On a working GPU installation, the output should be similar to:

```text
PyTorch: 2.13.0+cu126
TorchVision: 0.28.0+cu126
PyTorch CUDA runtime: 12.6
PyTorch CUDA available: True
PyTorch GPU count: 2
Faiss: 1.14.1
Faiss GPU count: 2
```

The number of visible GPUs depends on the machine and any `CUDA_VISIBLE_DEVICES` setting.

---

### 7. Test Faiss on the GPU

Run a small end-to-end nearest-neighbour search:

```bash
python - <<'PY'
import faiss
import numpy as np

dimension = 128
num_vectors = 10_000
num_queries = 10
k = 5

rng = np.random.default_rng(42)

database = rng.random(
    (num_vectors, dimension),
    dtype=np.float32,
)

queries = rng.random(
    (num_queries, dimension),
    dtype=np.float32,
)

resources = faiss.StandardGpuResources()

cpu_index = faiss.IndexFlatL2(dimension)
gpu_index = faiss.index_cpu_to_gpu(
    resources,
    0,
    cpu_index,
)

gpu_index.add(database)
distances, indices = gpu_index.search(queries, k)

print("Indexed vectors:", gpu_index.ntotal)
print("Result shape:", indices.shape)
print("First query neighbours:", indices[0])
print("First query distances:", distances[0])
PY
```

The expected result shape is:

```text
(10, 5)
```

---

### CPU-only installation

For a CPU-only environment, install PyTorch and Faiss from their CPU wheel indexes instead:

```bash
python -m pip install \
  torch==2.13.0 \
  torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m pip install "faiss-cpu>=1.14,<1.15"
```

Then install PathSeg normally:

```bash
python -m pip install -e ".[dev,parquet]"
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
