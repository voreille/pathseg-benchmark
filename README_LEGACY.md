# Legacy checkpoint export branch

This branch preserves the code required to load and export Lightning checkpoints created before the model-construction refactor.

## Purpose

Use this branch only when an old checkpoint must be loaded and converted into the standalone `nn.Module` format used for inference.

New experiments and training runs should use the current `main` branch.

## Compatibility

This branch corresponds to the immutable Git tag:

```text
checkpoint-export-v1
```

Legacy checkpoints may require their original experiment configuration because the network and tiler are reconstructed through `LightningCLI` before loading the checkpoint weights.

## Typical workflow

```bash
git switch legacy/checkpoint-export-v1

python <export-script> \
    --config <original-config.yaml> \
    --checkpoint <legacy-checkpoint.ckpt> \
    --output <exported-model.pt>
```

After exporting the model:

```bash
git switch main
```

## Important files

* Original experiment YAML configuration
* Legacy checkpoint
* Model export script
* Compatible dependency or environment specification

Do not remove or substantially refactor the legacy loading and export code on this branch unless required to fix checkpoint export compatibility.
