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
| LR schedule | `ExponentialLR(0.9999⁸)` once per epoch (200 steps) | same, every 200 steps (`lr_every_n_steps`) |
| EMA | decay 0.99, once per epoch, shadow starts at the initial weights, used for evaluation | same (`UpstreamEMA`, every 200 steps); used for validation, test and the checkpoint `state_dict` |
| Batch | 3 | 20 (as the Med-NCA run) |
| Length | 2000 epochs × 200 steps = 400k steps (1.2M samples) | 40k steps (800k samples), the Med-NCA budget. The LR ends at 1.6e-3 × 0.9999^(8·200) ≈ 1.36e-3 |
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
| 4 | Batch 20 instead of 3; upstream's per-epoch LR decay and EMA updates every 200 steps | Mirror the Med-NCA run (batch 20). The 200-step "epoch" keeps upstream's per-step timing, as Med-NCA keeps its per-step γ at batch 20 |
| 5 | 40k steps (the benchmark budget) instead of 400k | Same budget as Med-NCA. Upstream's lr decay is slow, so the lr barely decays (×0.85) |
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

## Deviations from the Med-NCA Variant B config

Base: `configs/mednca/ignite_mednca_upstream.yaml`. Same data, tiles, batch, precision,
`max_steps`, validation interval, tiler, `channel_n`, `n_eval_runs`, `max_batch_size`,
`compile_step` and loss switch.

| Key | Med-NCA Variant B | OctreeNCA | Why |
|---|---|---|---|
| `model.class_path` | `MedNCATraining` | `OctreeNCATraining` | Upstream trains on the whole patch, plus Adam/EMA |
| `network` | `MedNCASegmenter`, `hidden_size` 128, `steps` 64, `scale_factor` 4 | `OctreeNCASegmenter`, `hidden_size` 64, `steps` [10, 10, 10, 10, 28] | The model under test, with upstream values |
| `crop_size` | 112 | (none) | No crops upstream |
| `betas` / `lr_gamma` | (0.5, 0.5) / 0.9999 every step | (0.9, 0.99) / 0.9999⁸ every 200 steps | Upstream |
| `ema_decay` / `ema_every_n_steps` | (none) | 0.99 / 200 | Upstream |
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
- Before the next run, the EMA has to be disabled (`ema_decay: null`) or rescaled to our
  length (e.g. decay 0.99 every 10 steps ≈ upstream's 5 % horizon). Undecided.

## Results

| Model | Loss | val mIoU | test mIoU | Params | Run |
|---|---|---|---|---|---|
| OctreeNCA | `upstream_dice_bce` | (running; 0.300 raw at 18k, 40 batches) | | 47.8k | `2q9vrks3` |
| OctreeNCA | `cross_entropy_dice` | (running; 0.294 raw at 18k, 40 batches) | | 47.8k | `jd1vbzbz` |
| Med-NCA (Variant B) | `cross_entropy_dice` | 0.383 | | 132.5k | `58pyapml` |
| Med-NCA (Variant B) | `upstream_dice_bce` | 0.407 | | 132.5k | `7ro3wqzo` |
