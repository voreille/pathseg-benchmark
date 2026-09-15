# Semantic segmenter refactor scaffold

This scaffold is intended to be copied over a branch of the existing `pathseg`
repository. It adds a map-first semantic model interface without modifying the
legacy Lightning modules.

## Included files

- `pathseg/models/decoders/linear.py`: one named 1x1-convolution head per task.
- `pathseg/models/semantic_segmenter.py`: encoder → feature maps → decoder.
- `pathseg/models/sae_semantic_segmenter.py`: reconstructs the final feature map
  with an SAE before semantic decoding.
- `pathseg/models/builders_semantic.py`: class-path encoder/SAE construction and
  decoder construction.
- `pathseg/models/checkpoints.py`: strict submodule loading from trusted
  Lightning checkpoints.
- `pathseg/training/semantic_common.py`: task parsing, losses, tiling, metrics,
  prediction, and optimizer setup shared by both training recipes.
- `pathseg/training/semantic.py`: one trainer for single- and multi-task semantic
  segmentation.
- `pathseg/training/sae_semantic.py`: SAE reconstruction training with matched
  original-versus-reconstructed semantic evaluation.
- `pathseg/training/sae_launcher.py`: expands a minimal SAE run into a regular
  LightningCLI fit config by reusing a saved semantic run.
- `configs/semantic_two_heads.yaml`: migration of the pasted two-head behavior.
- `configs/sae_two_heads.yaml`: minimal SAE-only configuration.

## Required encoder contract

The encoder class named by `encoder_class_path` must expose:

```python
encoder.forward_feature_maps(imgs) -> tuple[Tensor, ...]
encoder.out_channels  # int or tuple[int, ...]
```

The return type is deliberately a tuple even for a single level. Every map is
NCHW and maps are ordered shallow-to-deep. A map already carries its runtime
resolution in `map.shape[-2:]`, so returning a second copy of `(height, width)`
would create metadata that can disagree with the tensor.

For a future pyramid encoder, expose static construction metadata separately:

```python
encoder.out_channels = (96, 192, 384, 768)
encoder.feature_strides = (4, 8, 16, 32)
```

The decoder uses `out_channels`/`feature_strides` to construct its layers and
the actual tensor shapes for runtime resizing and skip alignment. The current
linear decoder only uses the final value of `out_channels`.

For a ViT, make the token-to-map conversion explicit inside the encoder. The
runtime grid must travel with the tokens until they are reshaped; do not infer
a square grid with `sqrt(num_tokens)` and do not use the configured training
image size for a variable-size input:

```python
tokens, (grid_h, grid_w) = encoder.forward_tokens_with_grid(imgs)
if tokens.shape[1] != grid_h * grid_w:
    raise ValueError("token count does not match the runtime patch grid")
feature_map = tokens.transpose(1, 2).reshape(
    tokens.shape[0], tokens.shape[2], grid_h, grid_w
)
return (feature_map,)
```

If the backbone already returns NHWC or NCHW features, obtain `(grid_h,
grid_w)` from that tensor before converting it to tokens. For the current ViT
encoder, `forward_feature_maps()` returns a one-element tuple containing the
final patch-token map.

The scaffold uses fully-qualified class paths instead of adding every encoder to
a central registry. This keeps Lightning checkpoint hyperparameters
serializable and lets the existing CLI reconstruct the architecture. If you
prefer the repository's existing registry, only `builders_semantic.py` needs to
change.

## Required SAE contract

The pure SAE supplied by `sae_class_path` must expose:

```python
def forward_with_aux(self, tokens):
    return {
        "reconstructed_tokens": reconstructed_tokens,
        "latents": latents,
    }
```

When `normalize_decoder: true`, it must also expose
`normalize_decoder_()`. Adapt the method names in
`SAESemanticSegmenter.forward_sae()` if the current `TopKSAE` uses a different
return convention.

## SAE training from a semantic run

The saved semantic LightningCLI `config.yaml` is the source of truth for:

- encoder and semantic decoder construction;
- task heads, tiler, and output-resolution policy;
- the complete datamodule, dataset, and transform configuration.

The semantic checkpoint supplies the trained encoder and decoder weights. The
minimal SAE configuration therefore contains only the two semantic-run paths,
the new Trainer settings, and the pure SAE definition:

```yaml
semantic_run:
  config_path: runs/semantic/config.yaml
  checkpoint_path: runs/semantic/checkpoints/best.ckpt

trainer:
  precision: "16-mixed"
  max_steps: 40000
  val_check_interval: 1000

sae:
  class_path: pathseg.sae.topk.TopKSAE
  init_args:
    input_dim: 768
    num_latents: 3072
    k: 32
  lr: 3.0e-4
  weight_decay: 0.0
  normalize_decoder: true
```

Run it from the repository root:

```bash
python -m pathseg.training.sae_launcher \
  --config configs/sae_two_heads.yaml
```

Paths in the minimal config are resolved relative to the working directory,
matching the usual LightningCLI invocation from the repository root. Inspect
the generated standard LightningCLI config without starting training with:

```bash
python -m pathseg.training.sae_launcher \
  --config configs/sae_two_heads.yaml \
  --print-resolved-config
```

Extra LightningCLI overrides are forwarded after `--`:

```bash
python -m pathseg.training.sae_launcher \
  --config configs/sae_two_heads.yaml -- \
  --trainer.fast_dev_run=true
```

The baseline Trainer block is intentionally not inherited: it often contains
baseline-specific W&B tags, job type, step counts, and callbacks. Declare the
SAE Trainer/logger settings explicitly in the minimal config. The launcher
does inherit the semantic model and data sections.

The launcher expands the semantic architecture into ordinary
`TopKSAESemanticTraining` constructor arguments. That module saves those
resolved arguments but excludes initialization checkpoint paths from its
hyperparameters. Its completed checkpoint therefore contains the encoder,
semantic decoder, and SAE weights and can reload without the original semantic
checkpoint or configuration file.

## Batch and task behavior

The multi-task training path preserves the legacy mixed-batch format:

```python
(imgs, targets, source_ids, image_ids)
```

Each named head is supervised only on the subset matching its configured
`source_id`. A zero-valued graph-connected loss is used when a mixed batch has
no sample for a head, which is safer under DDP than returning a detached zero.

`eval_task_names` maps validation/test dataloader indices to heads. For example:

```yaml
eval_task_names: [a, b]
```

means dataloader 0 evaluates task `a` and dataloader 1 evaluates task `b`.

The supplied IGNITE/ANORAK config follows the pasted datamodule exactly:
training uses its mixed tensor/list collate, while validation uses the ordered
per-dataset loaders. Keep `eval_task_names` in the same order as `datasets`.

## All-head diagnostic figures

For the first batch of every validation/test dataloader, semantic evaluation
requests and stitches every configured head. The logged figure contains:

1. the input image;
2. exactly one GT mask, labelled with the current dataloader task;
3. one prediction panel for every semantic head.

Only the matching head contributes to IoU/F1 metrics. The other predictions are
diagnostic outputs, which makes unexpected cross-dataset behavior visible
without treating another taxonomy as ground truth.

SAE evaluation logs two such figures: one using the original encoder features
and one using the reconstructed features. This distinguishes a pre-existing
cross-dataset prediction from a change introduced by SAE reconstruction. All
later batches request only the matching head, so the diagnostic does not add
all-head stitching throughout the full validation epoch.

## Explicit output-resolution policy

The decoder only produces logits; `SemanticSegmenter` owns the final spatial
contract because it knows the actual input size. The default is fail-fast:

```yaml
upsample_logits: false
```

With that setting, logits whose spatial size differs from the input raise an
error. The 1x1 linear head intentionally uses:

```yaml
upsample_logits: true
interpolation_mode: bilinear
```

which explicitly authorizes upsampling smaller patch-grid logits to the input
size. It never silently downsamples oversized logits. The Lightning modules do
not perform any additional interpolation.

## Suggested integration order

1. Copy these files into the refactor branch without deleting legacy modules.
2. Add the explicit `forward_feature_maps()` contract to the current encoder.
3. Update the class paths and class counts in `semantic_two_heads.yaml`.
4. Run the model tests:

   ```bash
   pytest -q \
     tests/test_semantic_models.py \
     tests/test_semantic_diagnostics.py \
     tests/test_sae_launcher.py
   ```

5. Train/validate the new `SemanticTraining` and compare against one legacy
   checkpoint and fixed batch.
6. Only after semantic parity, launch `TopKSAESemanticTraining` from the saved
   semantic run with `sae_launcher.py`.

The old Mask2Former path is intentionally untouched. It can later implement a
`semantic_logits()` conversion while retaining its dedicated training recipe.

The SAE wrapper currently reconstructs only the final feature map. If a future
UNETR decoder consumes several intermediate maps, those earlier maps bypass the
SAE unless the wrapper is extended to reconstruct them as well.
