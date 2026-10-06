# CLAUDE.md — Med-NCA baseline (branch: `exp/mednca-compartments`)

## What this repo is
`pathseg` is a segmentation **benchmark** for histopathology. Every semantic segmentation model is a
`SemanticSegmenter` (`encoder` + `decoder`, `forward(imgs, task) -> dict[task, logits]`,
logits `B×K×H×W` at input resolution) trained by `pathseg.training.semantic.SemanticTraining`
via LightningCLI configs. Data, transforms, tiling/stitching (`GridPadTiler`), losses,
and metrics are shared across models. That shared pipeline is what makes results
comparable. **Do not modify it on this branch.**

## Goal of this branch
Add Med-NCA (Kalkhof et al. 2023) as a benchmark model such that, for the benchmark
run (Variant A below), **the only config change is
`model.init_args.network.class_path` + its `init_args`** (plus the few
trainer/optimizer settings listed under "Required config deviations"). Same data,
same loss, same tiler, same evaluation as the ViT/linear-head baselines.

Order of work:
1. Single task: IGNITE tissue compartments (16 classes incl. background, 0.5 µm/px).
2. Single task: ANORAK LUAD growth pattern (7 classes).
3. Multitask (IGNITE + ANORAK), same pattern as `two_head_linear.IgniteAnorakLinearSegmenter`.

## Separation of concerns (repo rule)
- `pathseg.models`: contracts + concrete architectures (`pathseg.models.architectures`).
  Only what is **kept at inference** lives here.
- Lightning modules (`pathseg.training.*`): everything that exists only to train.
  That covers data fetching, augmentation, losses, crop/sampling strategies, auxiliary
  heads (e.g. GRL + linear probes), and optimizer setup.
- Test for any new piece: "is it used when we run inference with a trained
  checkpoint?" Yes means architecture; no means Lightning module.

## Design: fit Med-NCA into the SemanticSegmenter contract
Create the final concrete architecture `pathseg/models/architectures/med_nca.py`. Don't
hesitate to create base classes if they help (e.g. a generic multi-level NCA). Reusable
contracts/base classes go next to the existing contracts in `pathseg.models`, following
the repo's current layout; only the concrete Med-NCA lives in `architectures`. The file
must contain:

- `MedNCAEncoder(nn.Module)`: the multi-level Med-NCA, exposed as composable stages
  so that training code can recombine them without re-implementing the model:
  - `downscale(imgs)`, `init_state(imgs)`, `upscale_state(state, size)`, and
    `run_level(level, state, imgs)` (runs that level's NCA for its configured steps;
    the image channels are re-injected and never updated).
  - The NCA state = [image channels | output channels | hidden]. Image channels = 3
    (RGB). Output channels = sum of classes over the configured tasks. Verify the
    upstream layout and keep it.
  - `forward_feature_maps(imgs) -> tuple[Tensor]` is the **inference composition**:
    coarse level on the downscaled image, then upscale, then fine level on the full
    image. It returns `(state_bchw,)`, i.e. ONE `B×channel_n×H×W` map. Convert from
    channels-last if the NCA works channels-last.
  - Stochastic firing is part of the model (kept at inference). In eval mode, run
    `n_eval_runs` passes and return the mean state; in training mode, a single pass.
    Use the same fire rate at eval as in training.
  - `set_grad_checkpointing(every: int | None)` (timm-style). It is only active when
    `self.training and torch.is_grad_enabled()` and does not change the computed
    function. Use `torch.utils.checkpoint(..., use_reentrant=False)` with the default
    `preserve_rng_state=True`, so the stochastic fire masks are identical in the
    recomputation. Training-motivated knobs are allowed in the architecture only
    when they can't live outside the step loop and don't change the function. The
    approved ones are this, `compile_step` (see "Training throughput"), and
    `max_batch_size` (a memory cap for eval). All default off.
- `StateSliceDecoder(nn.Module)`: no parameters. Maps the state to
  `{task: state[:, slice_for_task]}`.
  - Exposes `num_classes_by_task`.
  - Honors the `task` argument exactly like the existing linear decoder does (read
    `two_head_linear.py` first and mirror its behavior for `task=None` vs `task="ignite"`).
- `MedNCASegmenter(SemanticSegmenter)`: builds the encoder + decoder from flat `init_args`
  (same style as `IgniteAnorakLinearSegmenter`). Calls `super().__init__(...,
  upsample_logits=False)`. **No overrides of `encode/decode/forward`**; if one becomes
  necessary, stop and ask.

The output channels are raw logits read straight from the NCA state, as in upstream.
They are trained by the benchmark's `cross_entropy_dice`, so they are softmax logits.
Overlap-averaging in `GridPadTiler.stitch` works unchanged. A learned 1×1 conv head on
the hidden channels would be a different model; it's a follow-up, not the baseline.

## Training variants
**Variant A: benchmark protocol (do first).** Plain `SemanticTraining`, unchanged. The
loss is on the full tile; memory is handled by `set_grad_checkpointing` (configured via
the network init arg `grad_checkpointing_every`). This is the number for the benchmark
table.

**Variant B: author recipe (only if A underperforms or fails to train).**
`pathseg/training/med_nca.py` → `MedNCATraining(SemanticTraining)`. It overrides only
the training step:
- Run the coarse level on the full tile via the encoder stages, then upscale.
- Take a random fine-level crop of the state, image, and target (same crop for all
  three).
- Run the fine level on the crop and decode with `network.decoder((state_crop,), task)`.
- Compute the inherited loss on the crop.

Optionally, it switches the optimizer to upstream settings (as init args, documented).
Validation/test go through the unchanged inherited path, i.e. the full-tile
`network.forward` with tiler + stitch, so evaluation stays identical to every other
model. Config change: `model.class_path` → `MedNCATraining` (+ crop size and optimizer
init args). The network config is the same as Variant A.

## Upstream code
- Reference implementation: the original Med-NCA repo
  (https://github.com/MECLabTUDA/Med-NCA), cloned outside this repo at
  `/home/valentin/external-repos/Med-NCA` and mounted into the sandbox via
  `.claude-box.mounts`. It is **read-only**: never edit, commit to, or copy it wholesale
  into this repo. If something there looks incomplete, the maintained successor is
  https://github.com/MECLabTUDA/M3D-NCA; note any discrepancy, don't silently mix them.
- Record its commit hash (`git -C /home/valentin/external-repos/Med-NCA rev-parse HEAD`)
  in `experiments/mednca/README.md`.
- `pathseg` must never import from that path at runtime: the benchmark has to run
  without it.
- We use ONLY the model: the NCA update rule (backbone) and the multi-level forward
  logic (which lives in the Med-NCA agent). We do NOT use their Agent training loop,
  Experiment/config system, datasets, or losses.
- Port the backbone + multi-level forward into `med_nca.py` faithfully (with
  attribution), keeping upstream defaults (hidden size, steps per level, fire rate,
  scale factor, perception filters).
- Add a parity test `tests/test_med_nca_parity.py`: same weights + same seed means our
  port and upstream produce the same state on a random input. The test imports upstream
  from the path in env var `MEDNCA_REPO` (default `/home/valentin/external-repos/Med-NCA`)
  and is skipped via `pytest.skip` when that path doesn't exist.
- Before porting, read upstream and write down in the README: the backbone class, the
  agent's inference path, state channel layout, steps per level, the default
  `channel_n`/`hidden_size`, and the fire rate. Don't assume names.

## Allowed adaptations (the complete list)
1. Input channels: 3 (RGB) instead of 1 (MRI).
2. Output channels = 16 (IGNITE), 7 (ANORAK), or 23 (multitask).
3. `channel_n` large enough for 3 + outputs + hidden. Start at 64 single-task and 96
   multitask; keep upstream `hidden_size`.
4. Number of levels / scale factor if needed for large tiles. Upstream 2D is tuned for
   small images; for 896 px tiles a coarse level at 1/4 is 224 px. Prefer upstream
   defaults first; log any change.
5. Gradient checkpointing over NCA steps (see `set_grad_checkpointing`) so that
   full-tile BPTT fits in Variant A. Upstream's random fine-level crop is a training
   strategy, so it lives only in Variant B's Lightning module, never in the architecture.
6. Eval-time averaging of `n_eval_runs` stochastic passes (inside the encoder).

Anything else: stop and ask.

## Required config deviations (keep these minimal and documented)
- `network.class_path: pathseg.models.architectures.med_nca.MedNCASegmenter`, with
  `init_args` for: `num_classes_by_task`, `channel_n`, `hidden_size`, `steps`,
  `fire_rate`, `scale_factor`, `n_eval_runs`, `grad_checkpointing_every`,
  `max_batch_size`, `compile_step`.
- `lr_multiplier_encoder: 1.0`. All Med-NCA parameters live in the encoder; the
  baseline's 0.1 would cripple it. `freeze_encoder: false`. Verify that
  `SemanticTraining` copes with a decoder that has no parameters (empty param group).
- `trainer.precision`: NCAs iterate many steps and fp16 can diverge. Use `"bf16-mixed"`
  if the GPU supports it, else `"32-true"`. Log this.
- Tile / `img_size`: start with the benchmark's value. If memory forces a smaller tile
  even with checkpointing, change `data.init_args.img_size`, the transforms `img_size`,
  and the tiler `tile`/`stride` consistently (stride = tile/2), and log it as a
  deviation.
- Optimizer: benchmark defaults (lr 1e-4, wd 0.05, poly decay) in Variant A. Upstream
  optimizer settings belong to Variant B only.
- wandb: same `project`. Tags: replace `linear_decoder`/`h0-mini` with `med_nca`, keep
  the data/size tags. `job_type: baseline`.
- Single-task runs: keep only the IGNITE entry in `data.init_args.datasets` and only
  `ignite` in `tasks`.

Config file: `configs/mednca/ignite_mednca.yaml`, copied from the corresponding linear
baseline config with only the changes above.

## Sanity checks before any long run
1. Unit: `MedNCASegmenter(...)(torch.rand(2,3,T,T))` returns `{"ignite": 2×16×T×T}`;
   `ensure_input_resolution` passes without upsampling; print the param count (expect
   tens of thousands).
2. Parity test vs upstream passes.
3. `fast_dev_run` through LightningCLI with the real config (train + val + tiler stitch).
4. Overfit one batch (loss → ~0). If this fails, the port or channel slicing is wrong.
5. Peak GPU memory at the target tile size with checkpointing; adjust
   `grad_checkpointing_every`.
6. Stage composition: with `xd = downscale(x)`, the manual chain
   `run_level("fine", upscale_state(run_level("coarse", init_state(xd), xd), x.shape[-2:]), x)`
   equals `forward_feature_maps(x)` under a fixed seed. Run it in train mode (or with
   `n_eval_runs=1`), since eval mode averages several passes. This guarantees Variant B
   trains the same model that inference runs.
7. Grad checkpointing on vs off gives identical outputs and gradients under a fixed
   seed.

## Conventions
- New files only. No edits to shared benchmark code; if one seems needed, stop and ask.
- Respect the separation of concerns above. Nothing training-only goes into
  `models.architectures`, apart from the approved switches (grad checkpointing,
  `compile_step`, `max_batch_size`).
- One adaptation per commit, clear messages.
- Log every deviation from upstream and from the baseline config in
  `experiments/mednca/README.md`, plus a results table next to the ViT baselines.
- Outputs (checkpoints, wandb) stay out of git.

## Training throughput (done: 8.3×, 2026-10-06)
Variant A started at 0.079 it/s (448 px, batch 16, bf16, A100), against 1.44 it/s for
h0-mini + linear at 896 px. Per iteration Med-NCA needs only ~3× the FLOPs (~185 vs ~63
TFLOP), so the gap was mostly implementation inefficiency. It now runs at **0.656 it/s
(1.52 s/iter, 122 effective TFLOP/s, 22 GB peak)**, with the model, `state_dict` and
protocol unchanged. Per-commit numbers, the profiles and the variants that failed are in
`experiments/mednca/README.md` → "Throughput".

How the step is implemented now (`pathseg/models/architectures/med_nca.py`):
- The state stays channels-last `B×H×W×C`. Convs run on the free NHWC view with
  spatially transposed kernels, which equals upstream's convs on `x.transpose(1, 3)`.
- `BackboneNCA.folded_weights()` builds, once per `forward`, one 3×3 conv
  (kernel + bias) from `p0`/`p1`/`fc0`, plus `fc1` with its image-channel rows zeroed
  (so `dx = 0` there; this replaces the re-injection `cat`). The parameters themselves
  are untouched.
- `BackboneNCA.update` draws the fire mask **eagerly**, then calls one of two step
  functions with the same signature:
  - `_nca_step` (eager, hand-fused: `_ReflectPadCastHW`, `_CudnnConvBiasReLU`, mask
    in bf16): the default, and always used in eval / `no_grad` / on CPU.
  - `torch.compile(_nca_step_plain, dynamic=False, mode="max-autotune-no-cudagraphs")`:
    used only when `compile_step=True` and in training mode with grad enabled on CUDA.
    `_nca_step_plain` is the same step in plain ops, because Inductor can't fuse
    through the hand-written autograd Functions. Keep the two in sync: a test checks
    that they agree.

Rules for further changes:
- Profile before optimizing: `experiments/mednca/profile_step.py` (`--compile-step` for
  the compiled path). Record findings in the README.
- Allowed: implementation-level rewrites that compute the same function.
- Constraints: keep parameter names and the state_dict, so checkpoints and upstream
  weights still load. `tests/test_med_nca_parity.py` (1e-5) and `tests/test_med_nca.py`
  (checkpointing, composition, compiled = eager) must pass. One optimization per commit,
  with its measured speedup. Check GPU equivalence against the original code: fp32 with
  TF32 off should match to about 1e-7; bf16 should stay within the original's own
  bf16-vs-fp32 gap.
- Keep the fire mask outside any compiled region, so the masks and checkpoint replay
  stay identical to eager.
- Not allowed without asking: anything that changes the model or the protocol, such
  as fewer steps, a smaller `channel_n`, or fine-level crops (that is Variant B).

## Gotchas
- `pathseg fit` wraps the model in `torch.compile` unless you pass `--no_compile`. For
  Med-NCA, always pass `--no_compile`: compilation is done per step by `compile_step`,
  and a whole-model compile would unroll 2 × 64 steps.
- `compile_step` needs `dynamic=False`. With dynamic shapes, checkpoint recomputation
  recompiles and fails the checkpoint metadata check. The first training step takes
  about 1 min longer (max-autotune; cached in `/tmp/torchinductor_$USER`), and Triton
  needs a C compiler.
- In fp32 on an A100, cuDNN convs use TF32 by default. Set
  `torch.backends.cudnn.allow_tf32 = False` when checking equivalence to 1e-7.
- Tiler settings can't be overridden on the CLI (jsonargparse yields a `NestedArg`); set them in YAML.
- `--trainer.overfit_batches=1` does not fix the batch here (`WeightedRandomSampler` + random augmentations); overfit with a manual loop.
- `tests/test_semantic_models.py` is broken (imports a missing `pathseg.models.decoders.linear`); run tests by path.

## Follow-ups (not now)
- Learned 1×1 head on hidden channels; more NCA steps.
- Per-pixel variance over `n_eval_runs` as an uncertainty/QC map.
- OctreeNCA (same lab) for large fields of view.
- Whole-ROI inference without tiling (NCA is local and translation-invariant).
