# OctreeNCA baseline

OctreeNCA (Lemke et al., "OctreeNCA: Single-Pass 184 MP Segmentation on Consumer
Hardware", BMVC 2025, [arXiv 2508.06993](https://arxiv.org/abs/2508.06993)) ported as a
`SemanticSegmenter` for the pathseg benchmark, as a second NCA baseline next to Med-NCA
(`experiments/mednca/README.md`). Same lab as Med-NCA.

The port mirrors the Med-NCA port wherever possible (user decision, 2026-10-07): same
config layout and loss switch (`tasks.ignite.loss_name`), same data, tiles, batch,
precision, step budget and model knobs as Med-NCA Variant B. Only what defines OctreeNCA
(backbone, levels, steps, hidden size, optimizer, EMA) comes from upstream.

## Upstream reference

- Repo: https://github.com/MECLabTUDA/OctreeNCA
- Commit: `70f774232305b76330a75edef9fddda8ec960cab`
- Local clone (read-only, never imported by `pathseg`): `/home/valentin/external-repos/OctreeNCA`
  (override with `OCTREENCA_REPO` for the parity test).
- The pathology run is `train_peso.py`. It merges `configs/models/peso.py`,
  `configs/trainers/nca.py`, `configs/datasets/peso.py`, `configs/tasks/segmentation.py`
  and `configs/default.py`, then overrides some keys. It builds the model with
  `src/utils/BaselineConfigs.py::EXP_OctreeNCA` and trains it with
  `src/agents/Agent_MedNCA_Simple.py::MedNCAAgent` (→ `Agent_UNet` → `Agent`).

### What upstream does (read before porting)

| Item | Upstream (PESO run) |
|---|---|
| Model class | `src/models/Model_OctreeNCA_2d_patching2.py::OctreeNCA2DPatch2` (`EXP_OctreeNCA` uses it, not `Model_OctreeNCA.py`) |
| Backbone class | `src/models/Model_BasicNCA2D_fast.py::BasicNCA2DFast` (`model.backbone_class`). One per level (`model.octree.separate_models: True`), stored in `backbone_ncas` (`nn.ModuleList`, index 0 = finest) |
| Perception | `cat(state, conv(state))` with `conv = Conv2d(C, C, 3, padding=1, padding_mode="reflect", groups=C)`: **one depthwise** 3×3 conv with bias (`model.kernel_size: [3]*5`) |
| Update rule | `fc0: Conv2d(2C, hidden, 1)` → `bn` → ReLU → `fc1: Conv2d(hidden, C − in, 1, bias=False)` (zero-initialised). `bn` is `Identity`, since `train_peso.py` sets `model.normalization: "none"` (the code default is BatchNorm). New state = `cat(image, state[:, in:] + dx · mask)` |
| Fire mask | `zeros(B, 1, H, W, device).bernoulli_(p=self.fire_rate)`: a cell **updates with probability `fire_rate`**. The opposite convention to Med-NCA (`rand > fire_rate`); the two are equal at 0.5. Drawn on the state's device, redrawn every step, shared by all channels. The `fire_rate` argument of `update`/`forward` is ignored. |
| Layout | The level's state is channels-last `B×H×W×C` between levels. `BasicNCA2DFast.forward` rearranges it to BCHW and runs the convs there, **without** Med-NCA's H/W transpose |
| Frozen input | The image channels are concatenated back after every step (`const_inputs`) |
| State channels | `[input | output | hidden]`. Logits = `state[..., in : in + out]` |
| Octree / pyramid | `model.octree.res_and_steps`. PESO: 5 levels, 320 → 160 → 80 → 40 → 20 (×2 per level). The image for level *l* is `F.interpolate(original, size=res_l)` with the **default mode, i.e. nearest**, always from the full-resolution image |
| Steps per level | `train_peso.py`: `steps = 10` for levels 0–3 and `int(alpha * 20)` with `alpha = 1.0` for the 20×20 coarsest level, i.e. 20. That is the paper's α₀·max(H, W) rule, hard-coded for 20 px |
| Seed | Zeros at the coarsest resolution, image in channels `0:in` |
| Between levels | `nn.Upsample(scale_factor=2, mode="nearest")` of the whole state, then channels `0:in` are overwritten by the next level's image. Output and hidden channels are carried over |
| `channel_n` / `hidden_size` | 16 (= 3 in + 1 out + **12 hidden**) / 64 (update-MLP width) |
| `fire_rate` | 0.5 |
| Parameters | 3,104 per level, 15,520 in total, which matches the paper |
| Inference path | `forward_eval`: the training path with `patch_sizes = None`, under `no_grad`, one stochastic pass, same fire rate. For an input of another size, the level sizes become `ceil(prev / 2)`. `MedNCAAgent.test` evaluates the EMA weights (`ema.apply_shadow`) |
| Training patches | `model.train.patch_sizes: [None] * 5`, so **every level runs on the whole 320 px patch; there is no crop**. `loss_weighted_patching: False` |
| Loss | `DiceLoss` + `BCELoss`, weights 1 / 1 (`dice_loss_weight = 1.0`). Both are sigmoid, `smooth = 1` for Dice, computed over the flattened batch, **per output channel whose target has a positive pixel**, summed. That is the same function as Med-NCA's `DiceBCELoss` and our `UpstreamDiceBCELoss` |
| Optimizer | `Adam(lr=1.6e-3, betas=(0.9, 0.99))`, no weight decay, over all levels at once |
| LR schedule | `ExponentialLR(γ = 0.9999⁸)`, stepped **once per epoch** (`trainer.update_lr_per_epoch: True`) |
| EMA | `src/utils/EMA.py`, decay 0.99, updated **once per epoch** (`trainer.ema.update_per: "epoch"`). The shadow is initialised from the initial weights. Evaluation swaps the shadow weights in |
| Batch / epoch | Batch 3, 200 steps per epoch (`trainer.num_steps_per_epoch`), 2000 epochs (400k steps, 1.2M samples), `batch_duplication` 1 |
| Gradients | No clipping or normalisation (`trainer.normalize_gradients: None`); the norm is only logged |
| Data | PESO 320 px patches at `img_level` 1 (0.48 µm/px), patches with epithelium only. Input `/255` then **ImageNet mean/std**. batchgenerators augmentations (`get_transform_arr`) |

### Upstream discrepancies noticed

- **Pyramid downscaling.** The paper builds the octree "by averaging pixels in each node",
  i.e. area downscaling (also in `experiments/nca_literature.md`). The code uses
  `F.interpolate` with the default `mode="nearest"`, for the seed and for every
  re-injected image. The code wins: the port uses nearest.
- **Steps rule.** The paper states α₀·max(H, W) for the coarsest level. The code
  hard-codes `int(alpha * 20)`, which equals that rule only for 20 px. For our 448 px
  tiles the coarsest level is 28 px, so the rule gives 28 steps.
- **`fire_rate` argument ignored.** `BasicNCA2DFast.update` uses `self.fire_rate` for the
  mask, whatever `fire_rate` it receives. They are equal in all upstream configs.
- **Whole-slide demo script.** `inference2_peso_oct.py` (the 184 MP demo) runs level 2
  twice and skips the separate upsampling to level 2's size. The model's own
  `forward_eval`, which the training and evaluation code use, does not, and it is what
  we port.
- **Inference at other sizes** uses `ceil(prev / 2)` sizes but upsamples by a fixed ×2,
  which only fits for sizes divisible by 2⁴. Our tiles (448) are, so this does not arise;
  the port upsamples to the next level's size explicitly.

## The port

- Architecture: `pathseg/models/architectures/octree_nca.py` (`OctreeBackboneNCA`,
  `OctreeNCAEncoder`, `OctreeNCASegmenter`), on the base shared with Med-NCA in
  `pathseg/models/nca.py` (`FoldedNCA`: the fused/compiled step; `MultiLevelNCAEncoder`:
  seed, upscale, `run_level`, checkpointing, `n_eval_runs`, `max_batch_size`).
  `StateSliceDecoder` is reused from `med_nca.py`.
- Training module: `pathseg/training/octree_nca.py::OctreeNCATraining`; EMA callback
  `pathseg/training/callbacks/ema.py::UpstreamEMA`.
- Configs: `configs/octreenca/ignite_octreenca_upstream.yaml` (`upstream_dice_bce`) and
  `ignite_octreenca_upstream_celoss.yaml` (`cross_entropy_dice`). They differ only in the
  loss, like the Med-NCA pair.
- Tests: `tests/test_octree_nca_parity.py`, `tests/test_octree_nca.py`,
  `tests/test_octree_nca_training.py`.
- Sanity script: `experiments/octreenca/sanity_step.py` (throughput, peak memory, overfit).

Run (on the host; `--no_compile` as for Med-NCA):

    pathseg fit -c configs/octreenca/ignite_octreenca_upstream.yaml --data.num_workers=8 --no_compile
    pathseg fit -c configs/octreenca/ignite_octreenca_upstream_celoss.yaml --data.num_workers=8 --no_compile

The update folds like Med-NCA's: `fc0(cat(x, conv(x)))` is linear in `x`, so the depthwise
conv and `fc0` become one 3×3 conv, and `fc1` gets zero rows for the image channels. That
way OctreeNCA runs on the same fused/compiled step as Med-NCA, with upstream's
`state_dict` (`backbone_ncas.{l}.conv/fc0/fc1`) unchanged.

### What upstream does in training, and what we do

| | Upstream (PESO) | Ours (`OctreeNCATraining`) |
|---|---|---|
| Levels / sizes | 5: 320 → 20 | 5: 448 → 224 → 112 → 56 → 28 |
| Steps | 10, 10, 10, 10, 20 (α·max(H, W) at 20 px) | 10, 10, 10, 10, **28** (the same rule at 28 px) |
| Training input | Whole patch at every level, no crop | same (whole 448 tile, the inherited `SemanticTraining` step) |
| Loss | `DiceLoss + BCELoss` per present class, summed | `loss_name: upstream_dice_bce` (same function), or `cross_entropy_dice` in the `_celoss` config |
| Optimizer | Adam(1.6e-3, (0.9, 0.99)), no weight decay | same |
| LR schedule | `ExponentialLR(0.9999⁸)` once per epoch (200 steps) | same γ, every 20 steps (`lr_every_n_steps`): 2000 decays in 40k steps, as upstream (deviation 4) |
| EMA | decay 0.99, once per epoch, shadow starts at the initial weights, used for evaluation | same (`UpstreamEMA`), every 20 steps: 2000 updates in 40k steps, as upstream (deviation 4); used for validation, test and the checkpoint `state_dict` |
| Batch | 3 | 20 (as the Med-NCA run) |
| Length | 2000 epochs × 200 steps = 400k steps (1.2M samples) | 40k steps (800k samples), the Med-NCA budget. The LR ends at 1.6e-3 × 0.9999^(8·2000) ≈ 0.32e-3, as upstream (1.36e-3 in the first runs, with 200-step intervals) |
| `channel_n` / `hidden_size` | 16 (12 hidden) / 64 | **48 (29 hidden)**, as the Med-NCA run / 64 |
| Inference | One stochastic pass, EMA weights | same, through the benchmark tiler (448 / 224), `n_eval_runs=1` |

## Deviations from upstream

The allowed adaptations are listed in `CLAUDE.md`. "Mirror Med-NCA" was the user's
choice (2026-10-07).

| # | Deviation | Why |
|---|---|---|
| 1 | 16 output channels (IGNITE, background included) instead of 1 binary output | Task |
| 2 | `channel_n = 48` (3 + 16 + 29 hidden) instead of 16 (3 + 1 + 12 hidden); `hidden_size = 64` kept | Room for 16 outputs; same state size as the Med-NCA Variant B run, so the two NCAs differ in architecture, not state width (user: mirror Med-NCA) |
| 3 | Levels sized for 448 tiles: 448 → 28, 28 steps at the coarsest level | Tile size. Same 5 levels, ×2 per level, and the paper's α·max(H, W) rule with α = 1 |
| 4 | Batch 20 instead of 3, 40k steps instead of 400k; upstream's per-epoch LR decay and EMA updates every **20** steps, i.e. upstream's 2000 epochs rescaled to our run | Mirror the Med-NCA run (batch 20, 40k steps). The first runs kept upstream's 200-step epoch: then the run has only 200 epochs, the LR barely decays and the EMA weights don't work (see "EMA at our run length"). With 20-step epochs, the LR schedule and the EMA horizon (~5 % of training) match upstream's relative to the run. **This matches the schedule's shape, not the training:** 40k weight updates instead of 400k (10× fewer; with Adam each update moves the weights by ~lr whatever the batch), and 800k samples instead of 1.2M. `ignite_octreenca_upstream_bs3.yaml` runs the exact upstream batch and length to measure what this costs (see "Batch 20 × 40k vs upstream batch 3 × 400k") |
| 5 | 40k steps (the benchmark budget) instead of 400k | Same budget as Med-NCA. With the rescaled interval (deviation 4) the LR ends at 0.2× its start, as upstream's (×0.85 in the first runs, with 200-step intervals) |
| 6 | Input = RGB / 255 in [0, 1] instead of ImageNet mean/std | Shared benchmark pipeline, as for Med-NCA |
| 7 | Benchmark augmentations and sampling, not batchgenerators | Shared benchmark pipeline |
| 8 | Loss: pixels with the ignore label are left out; BCE from logits. With `_celoss`: the benchmark loss | As Med-NCA (`UpstreamDiceBCELoss`) |
| 9 | Kernel size fixed at 3 (upstream: configurable per level, 3 in the PESO run); `normalization` fixed at `"none"` (upstream default: BatchNorm, PESO run: `"none"`) | The PESO run; the folded step needs both |
| 10 | Implementation: the update folded into one 3×3 conv on the NHWC state, the shared fused/compiled step, nearest upscaling to the next level's explicit size | Speed; the same function. Parity holds to 1e-5 (float round-off); upstream weights load |
| 11 | `n_eval_runs`, `grad_checkpointing_every`, `max_batch_size`, `compile_step` (defaults off; the config sets `max_batch_size: 32`, `compile_step: true`) | The approved Med-NCA knobs |
| 12 | EMA: Lightning swaps the EMA weights in for validation and copies them into the model at the end of training; the checkpoint's `state_dict` holds the EMA weights (the raw weights are under `current_model_state`) | Upstream evaluates and exports the EMA weights; this makes `pathseg validate/test --ckpt_path` evaluate them too |

Not ported: the CUDA inference kernel (the same outputs, speed/VRAM only), ViTCA backbones,
loss-weighted patching (`assert False` upstream), and training crops (`patch_sizes`; the
PESO run uses none).

Parameters at `channel_n = 48`, `hidden_size = 64`: 9,568 per level, 47,840 in total
(Med-NCA Variant B: 132,544). Upstream PESO: 15,520, which the tests reproduce.

## Batch 20 × 40k vs upstream batch 3 × 400k

The batch-20 runs keep Med-NCA Variant B's budget (40k steps × 20, upstream Med-NCA's batch),
so the two NCAs are compared at the same cost. Upstream OctreeNCA trains batch 3 × 400k
steps: 10× more weight updates. The end of a val curve can't tell whether 40k steps are
enough, because the LR decays to 0.2× and flattens the curve either way. So the question is
answered by a run, not by the curve:

- `configs/octreenca/ignite_octreenca_upstream_bs3.yaml` (`upstream_dice_bce`): batch 3,
  400k steps, LR decay and EMA every 200 steps (upstream's epoch, no rescaling),
  `num_iterations_per_epoch: 10000` (30k samples per Lightning epoch, as at batch 20),
  validation every 10k and logging every 10 steps (40 validations and 40k logged steps,
  as at batch 20). wandb tag `batch3`. 300 steps + validation ran without error (sandbox,
  no workers).
- Cost: model-only 14.5 it/s at batch 3 (≈ 7.7 h for 400k); data-bound at ~45 samples/s,
  i.e. ~14 workers with the `.npy` cache. Expect ~8–12 h.
- Compare its final val mIoU (EMA and raw) with the batch-20 `upstream_dice_bce` run
  (`guv3jqtv`). If batch 3 is clearly better, it becomes the faithful OctreeNCA row and the
  batch-20 run stays as the equal-budget comparison with Med-NCA.

**Result (2026-10-09): batch 3 is +0.9 pt, not clearly better.** Both runs use
`upstream_dice_bce` and finished their schedule; values are the EMA weights on the full val
set, one pass (`n_eval_runs=1`), as logged during training.

| | `guv3jqtv` (batch 20 × 40k) | `z46nm48t` (batch 3 × 400k) |
|---|---|---|
| val mIoU | 0.387 | **0.396** |
| val mF1 | 0.520 | 0.536 |
| Wall time (host, A100) | 5.1 h (16 workers) | 11.5 h (14 workers, ~10.9 it/s) |
| Curve at the end | +0.9 pt over the last 5k steps (0.378 → 0.387), still rising | flat over the last 10k steps (0.3964 at 390k and 400k) |

Per class, the two runs differ in both directions, by more than the overall gap:

| Class | batch 20 | batch 3 |
|---|---|---|
| 1 Tumor epithelium | 0.550 | 0.612 |
| 4 Inflammation | 0.280 | 0.371 |
| 8 Erythrocytes | 0.117 | 0.312 |
| 12 Macrophages | 0.253 | 0.320 |
| 6 Fatty tissue | 0.352 | 0.253 |
| 9 Bronchial epithelium | 0.345 | 0.240 |
| 14 Liver | 0.537 | 0.387 |

(Classes with a difference of 5 pt or more; the full table is in `docs/nca_baselines.md`.)

Reading: the 10× more weight updates of upstream's schedule buy under 1 pt mIoU at 2.2× the
wall time. With one seed per setting and per-class swings of ±10–20 pt, the gap is within
what a second seed could change. Following the rule above, batch 20 stays the OctreeNCA row
(equal budget with Med-NCA); `z46nm48t` is reported next to it as the exact-upstream
reference. Raw weights (`current_model_state`) are not evaluated yet for either run.

## Deviations from the Med-NCA Variant B config

Base: `configs/mednca/ignite_mednca_upstream.yaml`. Same data, tiles, batch, precision,
`max_steps`, validation interval, tiler, `channel_n`, `n_eval_runs`, `max_batch_size`,
`compile_step` and loss switch.

| Key | Med-NCA Variant B | OctreeNCA | Why |
|---|---|---|---|
| `model.class_path` | `MedNCATraining` | `OctreeNCATraining` | Upstream trains on the whole patch, plus Adam/EMA |
| `network` | `MedNCASegmenter`, `hidden_size` 128, `steps` 64, `scale_factor` 4 | `OctreeNCASegmenter`, `hidden_size` 64, `steps` [10, 10, 10, 10, 28] | The model under test, with upstream values |
| `crop_size` | 112 | (none) | No crops upstream |
| `betas` / `lr_gamma` | (0.5, 0.5) / 0.9999 every step | (0.9, 0.99) / 0.9999⁸ every 20 steps | Upstream (epoch rescaled, deviation 4) |
| `ema_decay` / `ema_every_n_steps` | (none) | 0.99 / 20 | Upstream (epoch rescaled, deviation 4) |
| wandb tags | `med_nca`, `variant_b` | `octree_nca` | |

## Sanity checks (CLAUDE.md)

| # | Check | Status |
|---|---|---|
| 1 | Unit: `{"ignite": B×16×T×T}`, no upsampling, param count | Pass (`tests/test_octree_nca.py`): 47,840 parameters at `channel_n` 48, decoder has none; 15,520 for upstream PESO |
| 2 | Parity vs upstream (`tests/test_octree_nca_parity.py`) | Pass (CPU): seeded init identical, `state_dict` keys identical; backbone and the 5-level chain (train and eval mode) to 1e-5. A transposed kernel or Med-NCA's fire-mask draw fails the test |
| 3 | `fast_dev_run` through LightningCLI with the real config | Pass on an A100 (`fit`, bf16, `compile_step`, initial loss 20.5) and `validate` on 2 ROIs (448/224 tiler, stitch, metrics, EMA callback outside `fit`) |
| 4 | Overfit one batch | A100 (shared), full size: one fixed real batch of 4 × 448 (augmented), bf16, `compile_step`, the module's Adam (lr 1.6e-3), 1500 steps (`sanity_step.py --overfit 1500 --batch-size 4`). **`cross_entropy_dice`: pass**, loss 1.03 (step 250) → 0.29–0.39, pixel accuracy 0.28 → 0.82 → 0.91–0.94; still decreasing, not yet ~0 (as for Med-NCA, which flattened at 0.13–0.17). **`upstream_dice_bce`: the loss decreases** (3.80 at step 250 → 1.18) **but the argmax pixel accuracy stays at 0.03–0.10**: the channels of classes absent from the batch get no gradient and win the argmax. This is the hypothesis behind the Med-NCA `_celoss` run, reproduced here on one batch |
| 5 | Peak GPU memory at the target tile | Batch 20 × 448, bf16, no checkpointing: **17.6 GB** with `compile_step`, 15.1 GB eager. No checkpointing needed |
| 6 | Stage composition = `forward_feature_maps` | Pass (exact, train mode, fixed seed) |
| 7 | Grad checkpointing on vs off | Pass: identical outputs and gradients, recomputation verified |
| – | Compiled step | Pass: compiled = eager on CUDA with TF32 off, used only in CUDA training |
| – | Training module | Pass (`tests/test_octree_nca_training.py`): full-tile loss through `loss_name` (both losses), gradients reach every level, Adam/ExponentialLR settings, EMA timing, initial-weights blend and EMA weights in the checkpoint |

## Throughput

`experiments/octreenca/sanity_step.py --iters 10` (the module's own `training_step` + Adam on
a real batch of 20 × 448, bf16, A100). Measured on GPU 0 while **another job kept it at 100 %
utilisation**, so these are pessimistic:

| `compile_step` | s/iter | it/s | peak memory |
|---|---|---|---|
| on | 0.665 | 1.50 | 17.6 GB |
| off | 0.973 | 1.03 | 15.1 GB |

40k steps at 1.5 it/s take about 7.4 h, so no further optimisation for now. The step is
the one Med-NCA already optimised (`experiments/mednca/README.md` → Throughput).

### The real bottleneck is the data pipeline (2026-10-08)

Model only, on an idle A100 (448 px, bf16, forward + backward + Adam, random inputs):

| batch | `compile_step` | it/s | samples/s | peak memory |
|---|---|---|---|---|
| 3 | on | 14.5 | 43.5 | 2.7 GB |
| 3 | off | 8.6 | 25.9 | 2.3 GB |
| 20 | on | 2.4 | 48.4 | 15.5 GB |
| 20 | off | 2.9 | 58.4 | 15.1 GB |

The host runs (`--data.num_workers=8`) reached only 1.2–1.4 it/s, with the GPU idle much of
the time. One worker (single thread) of the shared IGNITE training pipeline produces
**2.8 samples/s**:

| Stage, per sample | ms |
|---|---|
| PNG decode of the whole ROI + one bool mask per class | 104 |
| color jitter on the whole ROI | 151 |
| scale jitter + flips (whole ROI and masks) | ~68 |
| pad + crop to 448 | 37 |

8 workers × 2.8 ≈ 22 samples/s ≈ 1.1 it/s at batch 20, the rate observed. Batch 20 at
~2.5 it/s needs ~18 workers (the host has 32 physical cores). Options that leave the
pipeline's output unchanged: more workers, or caching the decoded ROIs. Cropping before
the augmentations would be ~5–10× faster per worker, but it changes the shared transforms
(`adjust_contrast` uses the whole-image mean, and the RNG order changes), so it needs the
user's approval.

**Decoded `.npy` cache (done).** `python -m pathseg.preprocessing.npy_cache data/fss/ignite`
writes `images_npy/` and `masks_semantic_npy/` (2.5 GB, 28 s). The shared `Dataset` loads a
`.npy` file as the same pixels as the PNG (additive: PNG paths behave as before; a mask
falls back to `.npy` only when its `.png` is missing). Both OctreeNCA configs point at the
cache. Measured on 80 IGNITE training samples, one thread:

| | ms/sample | samples/s per worker |
|---|---|---|
| PNG | 423–432 | 2.35 |
| `.npy` | 316–323 | 3.13 (**1.34×**) |

The decode itself drops from ~69 to ~4 ms per ROI. 40 augmented samples under the same seed
are bit-identical (image, masks, labels, tensor types); `tests/test_dataset_npy_cache.py`
checks the same on synthetic ROIs. With the cache, batch 20 at ~2.5 it/s needs ~16 workers
instead of ~21.

## EMA at our run length: the EMA weights don't work (2026-10-08)

In the first two runs (`2q9vrks3` Dice + BCE, `jd1vbzbz` CE + Dice), the val mIoU logged in
wandb stayed at 0.02–0.06 through step 20k, with background IoU ≈ 0.01. The CE train loss
(≈ 1.3) was meanwhile on par with Med-NCA's. Validation uses the EMA weights (deviation 12),
so `last.ckpt` at step 18k was validated twice, on the same 40 val batches (A100, bf16,
`n_eval_runs=1`):

| Run | EMA weights (`state_dict`) | Raw weights (`current_model_state`) |
|---|---|---|
| `jd1vbzbz` (`cross_entropy_dice`) | 0.035 | **0.294** |
| `2q9vrks3` (`upstream_dice_bce`) | 0.03 (full val, wandb) | **0.300** |

For reference, Med-NCA Variant B at 20k steps: 0.280 (`58pyapml`, CE + Dice) and 0.340
(`7ro3wqzo`, Dice + BCE) on the full val set. The subset is representative: EMA on the 40
batches gives 0.035, and the full-val wandb value is 0.037.

Cause: upstream's EMA (decay 0.99, once per epoch) runs over 2000 epochs, so its horizon
(~100 epochs) is the last ~5 % of training. We keep the per-epoch timing (every 200 steps)
but train for only 200 such epochs. The shadow then averages over half the run and still
holds 0.99¹⁰⁰ ≈ 37 % of the *initial* weights at 20k steps (13 % at 40k). An NCA iterates
its update 10–28 times per level, so an average of weights from far-apart points of
training is not a working model.

Consequences:
- The val curves of these two runs in wandb measure the EMA, not the model being trained.
- The raw weights are in every checkpoint under `current_model_state`. At the end of
  training, Lightning copies the EMA into the model, so evaluate a copy of the checkpoint
  whose `state_dict` is replaced by `current_model_state`.
- Fix (decided 2026-10-08): the configs now update the EMA and the LR every 20 steps, so
  40k steps hold upstream's 2000 epochs (deviation 4). The EMA is kept, as in upstream's
  code (the paper mentions it briefly). Every checkpoint still holds the raw weights, so
  each run gives the number with and without EMA.

## Results

IGNITE val fold 0, 448 tiles / stride 224, one stochastic pass (`n_eval_runs=1`), values
logged during training on the full val set. OctreeNCA values are the **EMA weights**
(deviation 12); the raw-weights numbers are not evaluated yet.

| Model | Loss | Batch × steps | val mIoU (EMA) | val mIoU (raw) | test mIoU | Params | Run |
|---|---|---|---|---|---|---|---|
| OctreeNCA | `upstream_dice_bce` | 20 × 40k | 0.387 | | | 47.8k | `guv3jqtv` |
| OctreeNCA (exact upstream length) | `upstream_dice_bce` | 3 × 400k | 0.396 | | | 47.8k | `z46nm48t` |
| OctreeNCA | `cross_entropy_dice` | 20 × 40k | (not started) | | | 47.8k | |
| Med-NCA (Variant B) | `cross_entropy_dice` | 20 × 40k | 0.383 (no EMA) | | | 132.5k | `58pyapml` |
| Med-NCA (Variant B) | `upstream_dice_bce` | 20 × 40k | 0.407 (no EMA) | | | 132.5k | `7ro3wqzo` |

Superseded: `2q9vrks3` (`upstream_dice_bce`) and `jd1vbzbz` (`cross_entropy_dice`), the
first runs with 200-step LR/EMA intervals, stopped by the user. Their EMA weights did not
work (see "EMA at our run length"); raw weights at 18k: 0.300 and 0.294 (40 val batches).
