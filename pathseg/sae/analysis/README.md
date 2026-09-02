# SAE semantic analysis

This package analyzes a trained `SAESemanticSegmenter`. Its Python API accepts
already-constructed objects, while its CLI reconstructs them from the current
SAE and semantic configuration files.

## Configuration and checkpoint loading

The CLI accepts the same **minimal SAE config** as the training launcher. It
follows `semantic_run.config_path` and composes the current semantic `model` and
`data` sections with the current `sae` section. An already-resolved LightningCLI
config containing `model` and `data` is also accepted.

The resulting `TopKSAESemanticTraining` is constructed with current code and
configuration. Analysis then loads only the `network` subtree from the full SAE
checkpoint. The encoder, task decoder, and SAE are restored, while current task
specs, losses, metrics, and other Lightning state remain defined by the config.
Historical checkpoint hyperparameters are never passed into the constructor:

```bash
python -m pathseg.sae.analysis \
    --config configs/sae_two_heads.yaml \
    --checkpoint runs/checkpoints/my-sae/last.ckpt \
    --output-dir runs/checkpoints/my-sae/analysis
```

Start with `--max-batches-per-loader 2` for a smoke test.  The CLI uses the
training module's `/255` input scaling and `window_imgs_semantic` validation
crop geometry.  It automatically reads task definitions from
`TopKSAESemanticTraining.task_specs` and searches the decoder for each task's
final linear head.  If the decoder layout is ambiguous, the library API below
lets you pass the heads explicitly.

Validation targets may be dense class maps or Mask2Former-style dictionaries
with `masks: [N,H,W]` and `labels: [N]`. Dictionary targets are converted using
the reconstructed Lightning module's `_targets_to_per_pixel()` method. The
standalone `segmenter=` API has a matching fallback: the output starts at
`ignore_idx`, masks are assigned their class labels in order, and pixels outside
every mask remain ignored.

The semantic checkpoint referenced by `semantic_run.checkpoint_path` is not
opened during analysis. The completed SAE checkpoint already contains the
encoder and semantic decoder weights. Only `semantic_run.config_path` is needed
to recover their current architecture and the datamodule configuration.

For use from your existing launcher, call:

```python
from pathseg.sae.analysis import TaskSpec, analyze_sae


result = analyze_sae(
    lightning_module=sae_training_module,
    data_module=data_module,
    heads={
        "anorak": anorak_linear_head,
        "ignite": ignite_linear_head,
    },
    output_dir="runs/sae-analysis/validation",
    precision="16-mixed",
)
```

Pass the actual final `nn.Linear` or `Conv2d(kernel_size=1)` layer for each
task in `heads`.  If the decoder wraps those layers, expose them in the launcher
rather than teaching the generic analyzer about one decoder implementation.

For statistics only, set `top_examples_per_latent=0`.  When calling with a bare
`segmenter` rather than the Lightning module, provide `tasks`, `ignore_idx`,
and the appropriate `input_scale` explicitly.

## Outputs

- `analysis.pt`: reusable tensors and exact head-attribution matrices.
- `summary.json`: L0, dead latents, task class mass, and dataset token mass.
- `latents.csv`: one row per latent with density, importance, decoder norm,
  dataset activation, and task summaries.
- `task_<name>_latents.csv`: one row per class and latent.
- `rankings.json`: strongest positive and negative contributors per class.
- `top_activations.jsonl`: image IDs and token coordinates for selected latent
  activations, suitable for a later visualization pass.

The top-activation pass keeps at most two examples per image by default.  Token
coordinates are stored together with the latent grid size so they can be mapped
back to the input image without assuming a fixed encoder patch size.

## Selecting concepts

A latent is identified by its integer SAE coordinate. Keep the semantic
interpretation as metadata in a YAML manifest so the same selection drives both
visual inspection and causal testing:

```yaml
latents:
  - latent_id: 1319
    task: anorak
    class_index: 4
    label: candidate pattern concept
  - latent_id: 234
    task: ignite
    class_index: 15
    label: candidate compartment concept
```

`task`, `class_index`, and `label` record why a latent was selected. Ablation
always means zeroing that integer SAE coordinate, independent of the metadata.
The equivalent repeatable command-line shorthand is
`--latent 1319:anorak:4 --latent 234:ignite:15`.

For contact sheets, images are selected automatically from the frozen
`top_activations.jsonl` produced by the statistics command. This is preferable
to choosing attractive images manually: the examples are ranked by activation,
limited per source image during collection, and exactly replayable by
`dataset_name + sample_id` (including `::crop_N`). If a task or class is in the
latent specification, the examples are restricted to that task/class. Use just
the latent ID to inspect its strongest examples across both datasets.

## Contact sheets

```bash
python -m pathseg.sae.analysis.contact_sheet \
    --config configs/sae_two_heads.yaml \
    --checkpoint runs/checkpoints/my-sae/last.ckpt \
    --analysis-dir runs/checkpoints/my-sae/analysis \
    --manifest configs/sae_concepts.yaml \
    --examples 16
```

This creates one PNG per manifest entry plus `contact_sheets.json` under
`ANALYSIS_DIR/contact_sheets`. Each example contains:

1. the replayed validation crop with the maximally active token marked;
2. the full latent activation heatmap, on one shared scale per sheet;
3. a local target-mask view around that token.

The stored and recomputed activation values are printed together. A discrepancy
is a useful warning that the checkpoint, transforms, crop geometry, or precision
does not match the original analysis run. If the original statistics were
collected with `--no-window-inputs`, pass that flag to the contact-sheet command
too.

## Full-image causal ablation

Start with one or two batches to validate memory and output:

```bash
python -m pathseg.sae.analysis.ablation \
    --config configs/sae_two_heads.yaml \
    --checkpoint runs/checkpoints/my-sae/last.ckpt \
    --output-dir runs/checkpoints/my-sae/analysis/ablation-smoke \
    --manifest configs/sae_concepts.yaml \
    --max-batches-per-loader 2
```

Then remove the batch limit for the real result:

```bash
python -m pathseg.sae.analysis.ablation \
    --config configs/sae_two_heads.yaml \
    --checkpoint runs/checkpoints/my-sae/last.ckpt \
    --output-dir runs/checkpoints/my-sae/analysis/ablation \
    --manifest configs/sae_concepts.yaml
```

Every selected latent is evaluated individually on both heads by default. Use
`--task anorak` or repeat `--task` to restrict the validation loaders. The
command forwards each crop through the encoder and SAE once. For latent `j`, it
computes the exact zero-ablation efficiently as

```text
ablated_reconstruction = reconstruction - activation[j] * decoder_column[j]
```

It then runs the real task decoder and the training module's normal stitching
function. The baseline is the SAE reconstruction, so `delta_miou` isolates the
latent intervention rather than mixing it with SAE reconstruction error.

Outputs:

- `ablation_overall.csv`: firing rate, mean active activation, prediction flip
  rate, baseline/ablated mIoU, `delta_miou`, and positive `miou_drop`.
- `ablation_per_class.csv`: the same causal comparison at class-IoU level.
- `ablation.json`: nested machine-readable results and the exact selection
  metadata.

mIoU matches the training module's
`MulticlassJaccardIndex(average=None).mean()`: all configured classes use one
fixed denominator, and a class with zero union contributes zero. The command
caps individual interventions at 64 by default because runtime grows linearly
with the number of latents; raise `--max-latents` explicitly after ranking a
smaller candidate set.
