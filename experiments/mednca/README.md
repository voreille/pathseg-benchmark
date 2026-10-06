# Med-NCA baseline

Med-NCA (Kalkhof, González, Mukhopadhyay, IPMI 2023, https://arxiv.org/abs/2302.03473)
ported as a `SemanticSegmenter` for the pathseg benchmark.

- Architecture: `pathseg/models/architectures/med_nca.py`
- Config (Variant A, benchmark protocol): `configs/mednca/ignite_mednca.yaml`
- Config (Variant B, the authors' training recipe): `configs/mednca/ignite_mednca_upstream.yaml`,
  training module `pathseg/training/med_nca.py` (see [Variant B](#variant-b-the-authors-training-recipe))
- Tests: `tests/test_med_nca.py`, `tests/test_med_nca_parity.py`, `tests/test_med_nca_training.py`

## Upstream reference

- Repo: https://github.com/MECLabTUDA/Med-NCA
- Commit: `a844a72a1099db165d57f2d87bbcd5afd23586cb`
- Local clone (read-only, never imported by `pathseg`): `/home/valentin/external-repos/Med-NCA`
  (override with `MEDNCA_REPO` for the parity test).

### What upstream does (read before porting)

| Item | Upstream |
|---|---|
| Backbone class | `src/models/Model_BackboneNCA.py::BackboneNCA(BasicNCA)` |
| Perception | `cat(x, p0(x), p1(x))`, where `p0`, `p1` are learned `Conv2d(C, C, 3, padding=1, padding_mode="reflect")` with bias. The Sobel filters belong only to `BasicNCA`; Med-NCA does not use them. |
| Update rule | `fc0: Linear(3C, hidden_size)` → ReLU → `fc1: Linear(hidden_size, C, bias=False)` (zero-initialised). `x = x + dx * mask` |
| Fire mask | `torch.rand([B, H, W, 1]) > fire_rate`, one mask per cell shared by all channels, redrawn every step. It is drawn on the **CPU** generator and then moved to the device. |
| Layout | The state is channels-last `B×H×W×C`. `update` does `x.transpose(1, 3)` (→ `B×C×W×H`), so the convs run on the H/W-transposed grid. |
| Frozen input | `BasicNCA.forward` re-injects `x[..., :input_channels]` after every step. |
| State channels | `[input | output | hidden]`. The seed is zeros with the image in `0:input_channels` (`Agent_NCA.make_seed`). Outputs are read at `[input_channels : input_channels + output_channels]`. |
| Multi-level forward | `src/agents/Agent_Med_NCA.py::get_outputs`. The whole seed is downscaled to `int(H/4)` with `torchio.Resize` (linear, no antialiasing). Level 0 runs, then `nn.Upsample(scale_factor=4, mode="nearest")`, then the full-res image is re-injected into the input channels, then level 1 runs. |
| Inference path | `full_img=True`: both levels on the full image, `no_grad`, a single stochastic pass with the same fire rate as training. No averaging. |
| Train path | Level 1 runs on a random per-sample crop of size `input_size[0]` (= the coarse size) of the upscaled state and target. |
| Levels | 2 (`train_model: 1`). Scale factor 4 is hard-coded; the `scaling_factor` and `stacked_models` config keys are read but unused. |
| Steps per level | `inference_steps = 64`, the same for both levels |
| `fire_rate` | 0.5 (`cell_fire_rate`) |
| `hidden_size` | 128. This is the update-MLP width, **not** the number of hidden state channels. |
| `channel_n` | 16 in `src/examples/train_Med_NCA.py` (64 px images, crop 16, batch 48). 32 in `train_Med_NCA.ipynb` (256 px → coarse 64, crop 64, batch 20). The notebook matches the paper's "70k parameters" (2 × 35,008). |
| Input normalisation | z-norm, then rescale to [0, 1] |
| Optimizer | One Adam per level, lr 16e-4, betas (0.5, 0.5), ExponentialLR γ = 0.9999 |
| Loss | DiceBCE (sigmoid, binary) |

### Upstream discrepancies noticed

- **`input_channels` is dropped by `BackboneNCA`.** Its `__init__` calls
  `super().__init__(channel_n, fire_rate, device, hidden_size)` without `input_channels`,
  so `BasicNCA.input_channels` is always 1. Upstream only uses 1-channel MRI, so it
  has no effect there. With RGB only channel 0 would stay fixed. Our port freezes
  all 3 image channels, which is the intended behaviour. The parity test patches
  `upstream.input_channels = 3`.
- The example script and the notebook disagree on `channel_n` and image sizes (see
  the table above).

## Deviations from upstream

The allowed adaptations are listed in `CLAUDE.md`.

| # | Deviation | Why |
|---|---|---|
| 1 | 3 input channels (RGB) instead of 1 (MRI) | Histopathology input |
| 2 | Output channels = 16 (IGNITE), 7 (ANORAK) or 23 (multitask). Raw outputs are trained as softmax logits with the benchmark's `cross_entropy_dice`. | Benchmark loss |
| 3 | `channel_n = 64` single-task, 96 multitask. `hidden_size = 128` is kept. | Room for 3 + K + hidden state channels |
| 4 | The fire mask is drawn on the state's device, not on the CPU. Same distribution; on GPU the RNG stream differs from upstream. | Drawing `B×H×W` CPU randoms and copying them every step is a serious slowdown at histopathology tile sizes |
| 5 | Downscale with `F.interpolate(bilinear, align_corners=False, antialias=False)` instead of `torchio.Resize` (linear) | Same sampling positions, so `torchio` is not needed as a dependency. Checked in the parity test when `torchio` is installed. |
| 6 | Optional gradient checkpointing over NCA steps (`grad_checkpointing_every`, default off). The computed function is unchanged. | Full-tile BPTT in Variant A |
| 7 | `n_eval_runs` eval-time averaging of the state (default 1 = upstream) | Optional variance reduction |
| 8 | The fine level runs on the whole tile in training (Variant A), not on a random crop | Benchmark protocol. The crop recipe is Variant B, in a training module only. |
| 9 | Input = RGB / 255 in [0, 1], with no z-norm | Shared benchmark pipeline. The range is comparable to upstream's [0, 1] rescale. |
| 10 | `max_batch_size` (default off; 32 in the IGNITE config): `forward_feature_maps` splits larger inputs into chunks and concatenates them. No cross-sample ops, so the function is unchanged; only the fire-mask draw order differs. Approved by the user (not on the CLAUDE.md list). | Validation sends all tiles of an ROI in one call, at about 0.57 GB per 448 px tile (no_grad, bf16, A100). Large ROIs ran out of memory. With the cap, 150 tiles peak at 23.4 GB. |
| 11 | Implementation-level rewrites of the update step that compute the same function from the same `state_dict`: NHWC convs with spatially transposed kernels instead of `transpose(1, 3)`; `p0`/`p1`/`fc0` folded into one 3×3 conv; fused reflect pad + cast; cuDNN conv + bias + ReLU; fire mask applied in bf16. Upstream weights still load, and parity holds to float round-off (1e-5). See [Throughput](#throughput). | Training speed (CLAUDE.md "Current focus") |
| 12 | `compile_step` (default off; `true` in the IGNITE config): in CUDA training only, each NCA step runs through `torch.compile`. The function is unchanged and the fire masks are still drawn eagerly. Approved by the user (not on the CLAUDE.md list). See [Compiled step](#compiled-step-compile_step). | Training speed |

The parameter count at `channel_n = 64`, `hidden_size = 128` is 106,752 per level,
so 213,504 in total. At `channel_n = 96` it is 430,720.

## Deviations from the baseline config

Base: `configs/semantic_two_heads_refactored.yaml`.

| Key | Baseline | Med-NCA | Why |
|---|---|---|---|
| `model.init_args.network` | `IgniteAnorakLinearSegmenter` (h0-mini) | `MedNCASegmenter` | The model under test |
| `lr_multiplier_encoder` | 0.1 | 1.0 | All Med-NCA parameters live in the encoder |
| `trainer.precision` | `16-mixed` | `bf16-mixed` | fp16 can diverge over many NCA steps. The state accumulates in fp32. |
| tasks / datasets | IGNITE + ANORAK | IGNITE only | Single-task run |
| wandb `group` | ANORAK | IGNITE | Single-task IGNITE run |
| wandb tags | `linear_decoder`, `h0-mini`, `896x896`, `ANORAK+IGNITE`, `multitask` | `med_nca`, `448x448`, `IGNITE` | |
| `network.init_args.compile_step` | (n/a) | `true` | `torch.compile` of the NCA step in CUDA training, 1.37× faster. Run with `--no_compile`. |
| `img_size` (data + transforms), tiler `tile`/`stride` | 896 / 448 | 448 / 224 | Memory: full-tile BPTT at 896² with batch 16 does not fit even with checkpointing. This halves the field of view per tile at 0.5 µm/px. |

### Notes for running

- **Run `pathseg fit` with `--no_compile`.** Without it, `pathseg fit` wraps the whole
  module in `torch.compile`, which would unroll 2 × 64 NCA steps (with checkpoint
  regions); that is untested. Compilation is handled inside the network instead, by
  `compile_step: true`, which compiles only the single-step update.
- With `compile_step: true`, the first training step takes about 1 minute longer
  (Inductor max-autotune, 2 graphs: coarse and fine level). Later runs reuse the Inductor
  cache (`/tmp/torchinductor_$USER`). It needs a C compiler for Triton (gcc).
  Inductor's per-kernel autotune reports (`AUTOTUNE mm(...)` tables, "Autotune Choices
  Stats") are switched off for this compile only. Set `TORCHINDUCTOR_*` env vars or
  `TORCH_LOGS` if you want to see what it tunes.
- Validation passes all tiles of an image through the network in one call
  (`eval_step` → `self(crops)`). IGNITE ROIs reach about 2800 × 2200 px, which is
  about 120 tiles of 448. Measured on an A100 (no_grad, bf16-mixed, 64 steps):
  about 0.57 GB per tile, so a large ROI needs about 70 GB and ran out of memory.
  `max_batch_size: 32` caps tiles per forward. With it, 150 tiles peak at 23.4 GB.

## Variant B: the authors' training recipe

Goal: train as close to the Med-NCA authors as the benchmark allows, while evaluating
exactly like every other model. Variant A (above) uses the benchmark's training protocol
and converged slowly. Variant B is the run meant to represent Med-NCA.

Run (on the host; `--no_compile` as for Variant A):

    pathseg fit -c configs/mednca/ignite_mednca_upstream.yaml --data.num_workers=8 --no_compile

### What upstream does in training, and what we do

Read from upstream `Agent_Med_NCA.get_outputs` (training path), `Agent_Multi_NCA.batch_step`,
`Agent.initialize`, `src/losses/LossFunctions.py` and `train_Med_NCA.ipynb` (commit `a844a72`).

| | Upstream | Variant B (`MedNCATraining`) |
|---|---|---|
| Coarse level | Whole image downscaled ×4, 64 steps | same (whole 448 tile → 112) |
| Fine level | Upscaled state, then **one random crop per sample** of size `input_size[0]` (= coarse size), 64 steps | same: 112 × 112 crop of the 448 tile, same position for state, image and target |
| Loss | Only on the fine-level crop. `DiceBCELoss` (sigmoid; BCE mean + Dice with `smooth=1`, both over the flattened batch) **per output channel, summed over the channels whose target has a positive pixel**. The step is skipped if none does | same (`upstream_dice_bce_loss`), over the 16 class channels |
| Optimizer | One `Adam(lr=1.6e-3, betas=(0.5, 0.5))` per level, no weight decay | One Adam over both levels with the same settings (identical: Adam is per-parameter) |
| LR schedule | `ExponentialLR(γ=0.9999)`, stepped after **every batch** | same (`interval: step`) |
| Batch size | 20 (notebook) | 20 |
| `channel_n` | 32 = 1 in + 1 out + **30 hidden** (notebook) | 48 = 3 in + 16 out + **29 hidden** |
| Steps / fire rate / `hidden_size` / levels / scale | 64 / 0.5 / 128 / 2 / 4 | same |
| Inference | Full image, `no_grad`, one stochastic pass | same, through the benchmark tiler (448 / 224), `n_eval_runs=1` |

Choices made by the user (2026-10-06): upstream Dice+BCE rather than the benchmark loss;
448 tiles with a 112 crop (the coarse level then covers most of the tile, as upstream's
covers the whole image), rather than 896 / 224; `channel_n` 48 to match upstream's
hidden-channel count; batch 20 as upstream.

### Remaining deviations from upstream in Variant B

| Deviation | Why |
|---|---|
| RGB input, 16 output channels (IGNITE classes, background included) | Task. Upstream is 1-channel MRI with 1 binary output. |
| Pixels with the ignore label (255) are left out of BCE and Dice | Upstream has no ignore label |
| BCE computed from logits (`binary_cross_entropy_with_logits`) | Same value as upstream's `sigmoid` + `binary_cross_entropy`, but numerically stable (no clamping) |
| Crop corners from the torch generator, not Python `random` | Seeded with the rest of the run |
| Input = RGB / 255 in [0, 1], no z-norm + min-max rescale | Shared benchmark pipeline (Variant A deviation 9) |
| Fire mask drawn on GPU; downscale with `F.interpolate` instead of torchio | Same as Variant A (deviations 4, 5) |
| Augmentations, sampling and training length (`max_steps: 40000`, the benchmark's) | Benchmark data pipeline. Upstream trains for 1000 epochs over its dataset. With γ = 0.9999 per step, the lr ends at 1.6e-3 × 0.9999^40000 ≈ 2.9e-5. |
| Evaluation with the tiler (448 tiles, stride 224, weighted blend) | Benchmark protocol, identical for all models |
| A skipped batch (no labelled pixel) may still advance Lightning's scheduler | Rare (all-ignore crop); upstream also skips the scheduler step |

### Checks

- `tests/test_med_nca_training.py`:
  - the loss matches upstream `DiceBCELoss` summed over present classes (imported from
    the upstream checkout, skipped without it);
  - ignored pixels are left out, and all-ignored batches are skipped;
  - the crop uses the same position for state, image and target;
  - a training step backpropagates into both levels;
  - the optimizer and scheduler use upstream settings.
- `fast_dev_run` through LightningCLI with the real config on an A100 (train + validation):
  pass. Initial loss 18.7 (≈ 16 present classes × ~1.2).
- Step throughput (synthetic batch 20 × 448, bf16, A100): **4.49 it/s with `compile_step`**
  (0.22 s/iter, 15.0 GB peak, no checkpointing), 3.17 it/s eager. Variant A runs at 0.66 it/s.
- Parameters at `channel_n = 48`: 66,272 per level, 132,544 in total (upstream notebook:
  2 × 35,008 at `channel_n = 32`).

## Sanity checks (CLAUDE.md)

| # | Check | Status |
|---|---|---|
| 1 | Unit: `{"ignite": B×16×T×T}`, no upsampling, param count | Pass (`tests/test_med_nca.py`): 213,504 parameters, decoder has none |
| 2 | Parity vs upstream (`tests/test_med_nca_parity.py`) | Pass (CPU): single level and two-level chain to 1e-5 (float round-off; it was 1e-6 for the single level before the throughput rewrites changed the summation order); seeded init identical; downscale matches `torchio.Resize` |
| 3 | `fast_dev_run` through LightningCLI with the real config | Pass on CPU with `steps=2`: `fit` (train + val) and `validate` (448/224 tiler, `n_eval_runs=2`, stitch, metrics) |
| 4 | Overfit one batch | CPU, reduced setup (128 px tiles, `steps=16`, batch 2, fixed batch with ≥4 classes per sample, real `routed_forward` + `cross_entropy_dice`). lr 5e-4: loss 3.70 → 0.13–0.17, pixel accuracy 0.07 → 0.97 in 1500 iterations. It flattens there rather than reaching 0; the cause (fire-mask noise in each pass, capacity at 16 steps) is untested. lr 2e-3 learns but has loss spikes. Not yet repeated at full size on GPU. |
| 5 | Peak GPU memory at the target tile | TODO on GPU: `python experiments/mednca/probe_memory.py --tile 448 --batch-sizes 16` |
| 6 | Stage composition = `forward_feature_maps` | Pass (exact, train mode, fixed seed) |
| 7 | Grad checkpointing on vs off | Pass: identical outputs and gradients, recomputation verified |
| – | Compiled step (`compile_step`) | Pass: plain-ops step = eager step (CPU, values + gradients); used only in CUDA training; compiled = eager on CUDA with checkpointing (`tests/test_med_nca.py`) |

`overfit_batches=1` in Lightning is not a fixed batch with this datamodule
(`WeightedRandomSampler` + random augmentations), so check 4 uses a direct loop
over one fixed batch.

## Throughput

Measured with `experiments/mednca/profile_step.py`: one training step (forward,
`cross_entropy_dice`, backward, AdamW), synthetic batch, no data loading. Add
`--compile-step` to profile the compiled path.

**Summary (A100, tile 448, batch 16, `every=8`, bf16): 12.63 → 1.52 s/iter (8.3×),
0.079 → 0.656 it/s, peak memory 40.0 → 22.0 GB.** The model, `state_dict` and protocol
are unchanged. About 6.0× came from eager rewrites of the step (layout, folding,
fusion) and 1.37× more from `compile_step`.

### Baseline (2026-10-06, commit `ef68485`)

A100 80GB PCIe, tile 448, batch 16, `channel_n=64`, 64 steps/level,
`grad_checkpointing_every=8`, bf16 autocast, `cudnn.benchmark` off.

- **12.6 s/iter (0.079 it/s)**, matching the ~0.07 it/s seen in Lightning. The analytic
  cost is 186 TFLOP/iter, so the achieved rate is **14.7 TFLOP/s** (~5% of A100 bf16 peak).
  Peak memory is 40 GB.
- Device time by kernel category:

| category | s | share |
|---|---|---|
| copy (layout, strided `direct_copy`) | 5.98 | 47.0% |
| elementwise (adds, mask, relu, ...) | 2.70 | 21.2% |
| reflect pad (fp32) | 0.86 | 6.7% |
| conv | 0.78 | 6.1% |
| cast (fp32 → bf16) | 0.68 | 5.4% |
| layout transform (cuDNN NCHW↔NHWC) | 0.68 | 5.3% |
| gemm (`fc0`, `fc1`) | 0.62 | 4.8% |
| cat | 0.32 | 2.5% |

- The useful compute (conv + gemm) takes only 11% of device time. Most of the rest is
  memory traffic from layout copies. That is consistent with upstream's
  `transpose(1, 3)` producing a layout that neither cuDNN nor the GEMMs accept
  directly. The other costs are `torch.cat`, the separate fp32 reflection pad, and
  unfused elementwise ops.

### Optimizations (same function, same `state_dict`)

Same setup as the baseline (tile 448, batch 16, `every=8`, bf16, A100, GPU otherwise idle).
Each row is one commit. Equivalence is checked against the baseline on GPU (outputs and
gradients) with `experiments/mednca/check_equivalence.py`, which loads `med_nca.py` from
the pre-optimization commit `071dfa1`. fp32 matches to about 1e-6. The bf16 differences stay below the baseline's own
bf16-vs-fp32 gap.

| Commit | Change | s/iter | it/s | TFLOP/s | Peak mem | Speedup vs baseline |
|---|---|---|---|---|---|---|
| baseline | upstream layout (`transpose(1, 3)`) | 12.63 | 0.079 | 14.7 | 40.0 GB | 1.00× |
| layout | Perception on the channels-last state as a free NHWC view with spatially transposed kernels (no `transpose(1, 3)` copies). Reflect pad done in NHWC, cast to bf16 before pad/cat | 4.75 | 0.211 | 39.2 | 29.7 GB | 2.66× |
| fold | `p0`/`p1`/`fc0` folded into one 3×3 conv, built each forward from the existing weights. `fc1`'s image-channel rows are zeroed, so `dx = 0` there and the re-injection `cat` goes away. 98k → 74k MAC/pixel/step (TFLOP/s still counts the unfolded FLOPs). On GPU in true fp32 (TF32 off) it matches the baseline to 9e-8. | 3.21 | 0.312 | 58.0 | 19.4 GB | 3.94× |
| pad | Reflect pad + bf16 cast as one autograd Function: one full copy forward and one backward, plus border strips. Replaces cast + 2 `cat`s forward and the strided accumulation adds in backward. Exactly equal to `F.pad(reflect)` (values and gradients). | 2.55 | 0.392 | 72.9 | 19.7 GB | 4.95× |
| mask | Fire mask applied in bf16 (`dx * mask`, exact for a 0/1 mask) instead of upcasting `dx` to fp32 first | 2.47 | 0.404 | 75.2 | 19.7 GB | 5.11× |
| conv-relu | Conv + bias + ReLU as one cuDNN kernel (`torch.cudnn_convolution_relu`) in an autograd Function with an explicit backward (`threshold_backward` + `convolution_backward`). Removes the separate bias-add and ReLU passes over the 128-channel hidden map. One bf16 rounding instead of two. Plain conv + ReLU off CUDA. Traces under `torch.compile(backend="aot_eager")` with identical results. | 2.09 | 0.479 | 89.1 | 19.5 GB | 6.05× |
| compile | `compile_step: true`: `torch.compile` of a plain-ops step in CUDA training (see below) | 1.52 | 0.656 | 122.1 | 22.0 GB | 8.3× |

Profile at `conv-relu` (2.06 s of device time): elementwise 35% (mixed fp32+bf16 residual
add, mask multiplies, fp32 gradient accumulation, `threshold_backward`), conv 30%,
GEMM (`fc1`) 12%, copies 10%, casts 9% (pad forward/backward). The conv + GEMM share
rose from 11% to 42%. The rest is memory-bound pointwise work around the fp32 state. Eager
mode can't fuse it further, which is what `compile_step` addresses.

### Compiled step (`compile_step`)

Code paths in `pathseg/models/architectures/med_nca.py`. `BackboneNCA.update` draws the
fire mask eagerly and then calls one of two step functions with the same signature:

| Path | Function | Used when |
|---|---|---|
| eager | `_nca_step` (hand-fused: `_ReflectPadCastHW`, `_CudnnConvBiasReLU`) | default; always in eval / `no_grad` / CPU |
| compiled | `torch.compile(_nca_step_plain, dynamic=False, mode="max-autotune-no-cudagraphs")` | `compile_step=True` **and** training mode **and** grad enabled **and** CUDA |

- `_nca_step_plain` is the same step in plain ops (cast, `cat` reflect pad, conv, ReLU,
  `linear`, mask, residual). Inductor can't see inside the hand-fused autograd Functions.
  Compiling `_nca_step` itself reached only 1.83 s/iter. With plain ops, max-autotune
  fuses pad + cast into a Triton conv template, ReLU + mask + residual into the `fc1`
  GEMM epilogue, and the backward pointwise ops.
- The fire mask (`torch.rand`) stays outside the compiled region. Masks are therefore
  identical to the eager path, and checkpoint recomputation replays them through the
  preserved RNG state. Inductor's own RNG would draw different masks.
- **`dynamic=False` is required.** With the default (automatic dynamic shapes), checkpoint
  recomputation recompiled the step and failed with `CheckpointError: Recomputed values
  ... have different metadata`. Training has two fixed shapes (coarse 112², fine 448²),
  so two graphs are compiled.
- Eval stays eager because validation sends varying chunk sizes (`max_batch_size`
  remainders), and each new shape would recompile.
- Equivalence vs the baseline on GPU: fp32 with TF32 off matches to 9e-8. bf16 outputs
  differ by 1.2e-3 and gradients by 6.0e-3 (relative), against 1.0e-3 and 6.0e-3 for the
  baseline's own bf16 vs fp32.
- Measured variants (s/iter): eager 2.09; compiling `_nca_step` 1.92 (default mode) /
  1.83 (max-autotune); compiling `_nca_step_plain` 1.70 (default mode) / **1.52**
  (max-autotune, kept). Cold compile with max-autotune: about 66 s.
- Profile with `--compile-step` (1.52 s device time): Triton templates (conv / `fc1` with
  fused ops) 38%, Triton pointwise 32%, cuDNN conv backward (wgrad/dgrad) 23%, GEMM 3%.

Tried and not kept:
- `torch.addcmul(x, dx, mask)` for the masked residual: bit-identical but no faster
  (2.54 s/iter), because the mixed-dtype kernel is not vectorized and backward adds copies.
- `cudnn.benchmark=True`: no gain (2.55 s/iter at the `pad` commit). With NHWC inputs,
  cuDNN's heuristics already pick the same kernels.
- `torch.compile` with dynamic shapes, and compiling the hand-fused `_nca_step`: see
  "Compiled step" above.

## Results

| Model | Task | Tile | val mIoU | test mIoU | Params | Notes |
|---|---|---|---|---|---|---|
| h0-mini + linear (baseline) | IGNITE | 896 | | | | multitask config |
| Med-NCA (Variant A) | IGNITE | 448 | | | 213.5k | benchmark training protocol |
| Med-NCA (Variant B) | IGNITE | 448 | | | 132.5k | authors' recipe (crops, Dice+BCE, Adam) |
