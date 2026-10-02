# Med-NCA baseline

Med-NCA (Kalkhof, González, Mukhopadhyay, IPMI 2023, https://arxiv.org/abs/2302.03473)
ported as a `SemanticSegmenter` for the pathseg benchmark.

- Architecture: `pathseg/models/architectures/med_nca.py`
- Config (Variant A, benchmark protocol): `configs/mednca/ignite_mednca.yaml`
- Tests: `tests/test_med_nca.py`, `tests/test_med_nca_parity.py`

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
| `img_size` (data + transforms), tiler `tile`/`stride` | 896 / 448 | 448 / 224 | Memory: full-tile BPTT at 896² with batch 16 does not fit even with checkpointing. This halves the field of view per tile at 0.5 µm/px. |

### Notes for running

- `pathseg fit` wraps the module in `torch.compile` unless `--no_compile` is
  passed. Compiling would unroll 2 × 64 NCA steps (with checkpoint regions), so
  compile time and benefit are untested on GPU. Until they are measured, run
  with `--no_compile`.
- Validation passes all tiles of an image through the network in one call
  (`eval_step` → `self(crops)`). IGNITE ROIs reach about 2800 × 2200 px, which is
  up to about 120 tiles of 448. Under `no_grad`, one 64-channel fp32 state for
  120 tiles is about 6 GB, with several times that transient per step. Watch
  validation memory on the target GPU.

## Sanity checks (CLAUDE.md)

| # | Check | Status |
|---|---|---|
| 1 | Unit: `{"ignite": B×16×T×T}`, no upsampling, param count | Pass (`tests/test_med_nca.py`): 213,504 parameters, decoder has none |
| 2 | Parity vs upstream (`tests/test_med_nca_parity.py`) | Pass (CPU): single level exact to 1e-6; two-level chain to 1e-5 (float round-off); seeded init identical; downscale matches `torchio.Resize` |
| 3 | `fast_dev_run` through LightningCLI with the real config | Pass on CPU with `steps=2`: `fit` (train + val) and `validate` (448/224 tiler, `n_eval_runs=2`, stitch, metrics) |
| 4 | Overfit one batch | CPU, reduced setup (128 px tiles, `steps=16`, batch 2, fixed batch with ≥4 classes per sample, real `routed_forward` + `cross_entropy_dice`). lr 5e-4: loss 3.70 → 0.13–0.17, pixel accuracy 0.07 → 0.97 in 1500 iterations. It flattens there rather than reaching 0; the cause (fire-mask noise in each pass, capacity at 16 steps) is untested. lr 2e-3 learns but has loss spikes. Not yet repeated at full size on GPU. |
| 5 | Peak GPU memory at the target tile | TODO on GPU: `python experiments/mednca/probe_memory.py --tile 448 --batch-sizes 16` |
| 6 | Stage composition = `forward_feature_maps` | Pass (exact, train mode, fixed seed) |
| 7 | Grad checkpointing on vs off | Pass: identical outputs and gradients, recomputation verified |

`overfit_batches=1` in Lightning is not a fixed batch with this datamodule
(`WeightedRandomSampler` + random augmentations), so check 4 uses a direct loop
over one fixed batch.

## Results

| Model | Task | Tile | val mIoU | test mIoU | Params | Notes |
|---|---|---|---|---|---|---|
| h0-mini + linear (baseline) | IGNITE | 896 | | | | multitask config |
| Med-NCA (Variant A) | IGNITE | 448 | | | 213.5k | |
