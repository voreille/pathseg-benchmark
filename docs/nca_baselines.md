# NCA baselines on IGNITE: progress (2026-10-10)

Summary of what we tried to add Neural Cellular Automata to the pathseg benchmark. The
details, deviation tables and checks live in `experiments/mednca/README.md` and
`experiments/octreenca/README.md`. This page only summarises them.

## Goal and setup

- Two NCA baselines, ported faithfully from the authors' code (MECLabTUDA) as
  `SemanticSegmenter`s:
  - **Med-NCA** (Kalkhof et al., IPMI 2023): 2 levels (×4), 64 steps per level.
  - **OctreeNCA** (Lemke et al., BMVC 2025): 5 levels (×2), 10 steps per level and 28 at
    the coarsest.
- Evaluation is the benchmark's, identical for every model: IGNITE tissue compartments
  (16 classes, 0.5 µm/px), fold 0, 448 px tiles, stride 224, the shared tiler, stitching
  and metrics.
- All numbers below are **val mIoU from one stochastic pass** (`n_eval_runs=1`), logged
  during training. There are no test numbers yet.
- Muscle (class 13) is absent from fold 0's training split. Its IoU is 0 for every model,
  and the data are kept as shipped.

## Timeline

| Date | Step | Outcome |
|---|---|---|
| 10-06 | Med-NCA port: backbone and two-level forward, parity test vs upstream (1e-5), sanity checks | Pass |
| 10-06 | **Variant A**: benchmark protocol (full-tile BPTT, `cross_entropy_dice`, AdamW 1e-4, grad checkpointing), `channel_n` 64 | Trained, but slowly: 0.315 at 39k steps after 19 h (`ogemw00x`) |
| 10-06 | Throughput: same function and `state_dict`, NHWC layout, folded convs, fused pad / conv+ReLU, `torch.compile` per step | 8.3× faster (0.079 → 0.656 it/s) |
| 10-06 | **Variant B**: authors' recipe (112 px fine-level crop, upstream Dice + BCE, Adam (0.5, 0.5), lr 1.6e-3, batch 20, `channel_n` 48) | **0.406** (`7ro3wqzo`), the best NCA so far |
| 10-07 | Hypothesis: Dice + BCE over present classes leaves absent-class channels untrained → rerun Variant B with `cross_entropy_dice` | 0.383 (`58pyapml`): refuted, the loss is not the bottleneck |
| 10-07 | Eval-time averaging, `n_eval_runs=8` on `7ro3wqzo` | 0.412 (+0.5 pt) |
| 10-07 | Level ablations at eval (`diagnose_levels.py`) | Fine level alone ≈ 0; coarse level very sensitive to its downscaling. Otherwise inconclusive without retraining |
| 10-08 | OctreeNCA port on the shared NCA base (`pathseg/models/nca.py`), parity test (1e-5), sanity checks | Pass. 47.8k params vs Med-NCA's 132.5k |
| 10-08 | First OctreeNCA runs, LR/EMA every 200 steps (one upstream epoch) | wandb val ≈ 0.03: **the EMA weights didn't work** at 200 epochs. Raw weights ≈ 0.30 at 18k. Runs stopped |
| 10-08 | Fix: LR decay and EMA every 20 steps (upstream's 2000 epochs in 40k steps) | **0.387** (`guv3jqtv`, batch 20 × 40k) |
| 10-08 | Data pipeline turned out to be the bottleneck (CPU augmentations on whole ROIs) → decoded `.npy` cache | Same pixels, 1.34× per worker |
| 10-09 | OctreeNCA at exact upstream batch and length (batch 3 × 400k) | **0.396** (`z46nm48t`), +0.9 pt for 2.2× the wall time |

## Results

| Model | Recipe | Loss | Params | val mIoU | Run |
|---|---|---|---|---|---|
| Med-NCA, Variant A | benchmark protocol | `cross_entropy_dice` | 213.5k | 0.315 (39k, stopped) | `ogemw00x` |
| Med-NCA, Variant B | authors' (crop) | `upstream_dice_bce` | 132.5k | **0.406** (0.412 with 8 runs) | `7ro3wqzo` |
| Med-NCA, Variant B | authors' (crop) | `cross_entropy_dice` | 132.5k | 0.383 | `58pyapml` |
| OctreeNCA | authors', batch 20 × 40k | `upstream_dice_bce` | 47.8k | 0.387 (EMA) | `guv3jqtv` |
| OctreeNCA | authors', batch 3 × 400k | `upstream_dice_bce` | 47.8k | 0.396 (EMA) | `z46nm48t` |
| OctreeNCA | authors', batch 20 × 40k | `cross_entropy_dice` | 47.8k | not started | |

Per-class val IoU (last validation of each run):

| Class | Med-NCA A | Med-NCA B | Med-NCA B (CE) | Octree b20 | Octree b3 |
|---|---|---|---|---|---|
| **mIoU** | **0.315** | **0.406** | **0.383** | **0.387** | **0.396** |
| 0 Background | 0.824 | 0.834 | 0.827 | 0.834 | 0.838 |
| 1 Tumor epithelium | 0.546 | 0.614 | 0.605 | 0.550 | 0.612 |
| 2 Reactive epithelium | 0.081 | 0.201 | 0.150 | 0.225 | 0.225 |
| 3 Stroma | 0.550 | 0.609 | 0.604 | 0.564 | 0.613 |
| 4 Inflammation | 0.224 | 0.464 | 0.392 | 0.280 | 0.371 |
| 5 Alveolar tissue | 0.333 | 0.459 | 0.435 | 0.445 | 0.495 |
| 6 Fatty tissue | 0.236 | 0.391 | 0.370 | 0.352 | 0.253 |
| 7 Necrotic tissue | 0.063 | 0.117 | 0.071 | 0.067 | 0.134 |
| 8 Erythrocytes | 0.169 | 0.326 | 0.249 | 0.117 | 0.312 |
| 9 Bronchial epithelium | 0.220 | 0.185 | 0.080 | 0.345 | 0.240 |
| 10 Mucus/Plasma/Fluids | 0.345 | 0.374 | 0.367 | 0.431 | 0.422 |
| 11 Cartilage/Bone | 0.438 | 0.528 | 0.592 | 0.689 | 0.633 |
| 12 Macrophages | 0.204 | 0.253 | 0.254 | 0.253 | 0.320 |
| 13 Muscle | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 14 Liver | 0.430 | 0.629 | 0.675 | 0.537 | 0.387 |
| 15 Keratinization | 0.370 | 0.509 | 0.457 | 0.506 | 0.489 |

## What we learned

- **The authors' recipe matters more than the architecture.** Med-NCA goes from 0.315
  (benchmark protocol) to 0.406 (authors' recipe). The two NCAs end within 1–2 pt of each
  other: Med-NCA 0.406 vs OctreeNCA 0.387–0.396. That is despite OctreeNCA having 2.8×
  fewer parameters and much more context (5 levels).
- **The paper's ranking doesn't carry over.** OctreeNCA beats Med-NCA on PESO (binary).
  On 16-class IGNITE it is slightly below. Published NCA work stops at about 10 classes.
- **The loss isn't the bottleneck (Med-NCA).** Benchmark CE + Dice is 2.3 pt below upstream
  Dice + BCE. The OctreeNCA CE run is still to do.
- **OctreeNCA's training budget isn't the bottleneck either.** 10× more weight updates
  (batch 3 × 400k) give +0.9 pt. The per-class differences (±10–20 pt) go both ways, so
  one seed can't separate the two.
- **Single-seed noise.** Classes such as Liver, Fatty tissue, Bronchial epithelium and
  Erythrocytes move by 10–20 pt between runs that differ by a single setting. The mIoU
  differences of ≤1 pt above should not be over-read.
- **Pitfalls hit along the way:**
  - EMA horizon vs run length: an NCA averaged over far-apart weights is not a working
    model.
  - CPU-bound data loading: whole-ROI augmentations.
  - LightningCLI silently applies the checkpoint's hparams over the CLI.
  - `/dev/shm` limits in the sandbox.

## Open items

1. Raw-weight (`current_model_state`) val numbers for `guv3jqtv` and `z46nm48t`, to go
   with the EMA numbers.
2. OctreeNCA batch 20 × 40k with `cross_entropy_dice` (the pair of `58pyapml`): not
   started.
3. Linear-probe / U-Net / ViT reference numbers in the same results table.
4. Decide `n_eval_runs` for the final numbers (1 = upstream, 8 = +0.5 pt for Med-NCA), then
   run test.
5. Improvement phase on the better NCA. Candidates are in `experiments/nca_literature.md`.
   This phase deviates from upstream on purpose, so its allowed deviations have to be
   agreed first. The faithful ports stay as their own rows.
