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
