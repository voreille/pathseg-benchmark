# training/callbacks/debug_aug.py
import torch
from lightning.pytorch.callbacks import Callback
import wandb


class DebugAugCallback(Callback):
    """Logs a panel (image / target / current pred) for a few train batches each epoch."""

    def __init__(self, num_batches: int = 2, freq_epochs: int = 1):
        self.num_batches = num_batches
        self.freq_epochs = freq_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        if trainer.current_epoch % self.freq_epochs != 0:
            return
        if batch_idx >= self.num_batches:
            return

        imgs, targets = batch  # already augmented by your dataset
        imgs = imgs.to(pl_module.device)
        with torch.no_grad():
            logits = pl_module(imgs)
            if isinstance(logits, dict):
                logits = logits["main"]
            logits = logits.detach().cpu()

        # one sample is enough to spot issues; take 0
        tgt_pp = pl_module.to_per_pixel_targets_semantic(
            [targets[0]], pl_module.ignore_idx)[0].cpu()
        panel = pl_module.plot_semantic(imgs[0].cpu(),
                                        tgt_pp,
                                        logits=logits[0])

        if hasattr(trainer.logger, "experiment") and isinstance(
                trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            trainer.logger.experiment.log({
                "debug/augmented_panel": [wandb.Image(panel)],
                "debug/epoch":
                trainer.current_epoch,
            })
