# CLAUDE.md

## What this repo is
`pathseg` is a segmentation **benchmark** for histopathology. Every semantic segmentation model is a
`SemanticSegmenter` (`encoder` + `decoder`, `forward(imgs, task) -> dict[task, logits]`,
logits `B×K×H×W` at input resolution) trained by `pathseg.training.semantic.SemanticTraining`
via LightningCLI configs. Data, transforms, tiling/stitching (`GridPadTiler`), losses,
and metrics are shared across models. That shared pipeline is what makes results
comparable. **Do not change its behaviour** (additive extensions: see "Conventions").

## Project docs
- NCA baselines (Med-NCA, OctreeNCA): spec and working rules in `docs/nca_spec.md` (read it
  before touching any NCA code); progress and results in `docs/nca_baselines.md`; details in
  `experiments/mednca/README.md` and `experiments/octreenca/README.md`. Tag `mednca-variant-b`
  marks the end of the Med-NCA work.

## Separation of concerns (repo rule)
- `pathseg.models`: contracts + concrete architectures (`pathseg.models.architectures`).
  Only what is **kept at inference** lives here.
- Lightning modules (`pathseg.training.*`): everything that exists only to train.
  That covers data fetching, augmentation, losses, crop/sampling strategies, auxiliary
  heads (e.g. GRL + linear probes), and optimizer setup.
- Test for any new piece: "is it used when we run inference with a trained
  checkpoint?" Yes means architecture; no means Lightning module.

## Conventions
- Shared benchmark code (data, transforms, tiler, losses, metrics, `SemanticTraining`):
  existing behaviour is frozen, so every model already in the table must give the same
  results.
  - **Allowed without asking (additive):** new classes or functions, and new entries in
    existing builders/registries under a new name (e.g. a new `loss_name` branch in
    `build_criterion`). Existing names, defaults and signatures stay unchanged. Add a test
    for the new entry and log it in the README of the experiment that needs it.
  - **Stop and ask:** anything that changes what existing code computes, its defaults or
    signatures, and anything touching evaluation (tiler, stitching, metrics, data splits).
  - Prefer extending the shared mechanism over working around it. Don't add a parallel
    knob that silently overrides an existing config field (like the former
    `MedNCATraining(loss=...)` vs `loss_name`).
- One change per commit, clear messages.
- Outputs (checkpoints, wandb) stay out of git.

## Gotchas
- `pathseg validate/test --ckpt_path X`: LightningCLI applies the hparams stored in the
  checkpoint over the YAML and the CLI, silently. To eval with other model args, patch
  `ckpt["hyper_parameters"]["network"]["init_args"]` in a copy of the checkpoint.
- Tiler settings can't be overridden on the CLI (jsonargparse yields a `NestedArg`); set them in YAML.
- `--trainer.overfit_batches=1` does not fix the batch here (`WeightedRandomSampler` + random augmentations); overfit with a manual loop.
- In the claude-box sandbox, `/dev/shm` is 64 MB, so DataLoader workers fail or hang
  ("unable to allocate shared memory"). Use `--data.num_workers=0` there. The host is
  fine with 8 workers.
- Local wandb runs: `experiments/read_wandb_history.py runs/wandb/run-*-<id>/run-<id>.wandb`
  prints the val curves (wandb 0.30 has no Python reader for `.wandb` files).
- IGNITE fold 0: the training split has no Muscle (class 13), so its IoU is 0 for every
  model. That is how the data are shipped; keep it.
- `tests/test_semantic_models.py` is broken (imports a missing `pathseg.models.decoders.linear`); run tests by path.
