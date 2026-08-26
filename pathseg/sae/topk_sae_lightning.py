from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.models.builders import build_decoder, build_tiler
from pathseg.sae.encoder import EncoderWithTopKSAE
from pathseg.training.lightning_module import LightningModule


class TopKSAETraining(LightningModule):
    """Train a Top-k SAE on the patch tokens of a frozen image encoder.

    The dataloader may return ``(images, targets)`` or any tuple/list whose
    first element is the image batch. Supervised targets are intentionally
    ignored.

    ``EncoderWithTopKSAE.forward`` must return a dictionary containing:

    - ``tokens``: original patch tokens, shape ``[B, N, D]``;
    - ``reconstructed_tokens``: SAE reconstructions, shape ``[B, N, D]``;
    - ``latents``: sparse codes, shape ``[B, N, M]``.

    When ``normalize_decoder`` is enabled, ``network.sae`` must provide an
    in-place ``normalize_decoder_()`` method.
    """

    def __init__(
        self,
        encoder_id: str,
        num_latents: int,
        k: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = True,
        normalize_decoder: bool = True,
        tiler_name: Optional[str] = None,
        tiler_init_args: dict[str, Any] | None = None,
    ):
        network = EncoderWithTopKSAE(
            encoder_id=encoder_id,
            num_latents=num_latents,
            k=k,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )
        tiler = build_tiler(
            tiler_name=tiler_name,
            tiler_init_args=tiler_init_args,
        )
 
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            tiler=tiler,
        )

        if normalize_decoder and not hasattr(self.network.sae, "normalize_decoder_"):
            raise AttributeError(
                "normalize_decoder=True requires TopKSAE.normalize_decoder_()."
            )

        self.poly_lr_decay_power = poly_lr_decay_power
        self.normalize_decoder = normalize_decoder
        self.save_hyperparameters()

    @staticmethod
    def _images_from_batch(batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            if "imgs" in batch:
                return batch["imgs"]
            if "images" in batch:
                return batch["images"]
            raise KeyError("A dictionary batch must contain 'imgs' or 'images'.")

        if isinstance(batch, (tuple, list)):
            return batch[0]

        return batch

    @staticmethod
    def _reconstruction_metrics(
        output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Stop gradients through the reconstruction target. For the baseline,
        # the ViT should additionally be frozen.
        tokens = output["tokens"].detach()
        reconstructed = output["reconstructed_tokens"]
        latents = output["latents"]

        residual = reconstructed - tokens
        residual_ss = residual.square().sum()

        reconstruction_loss = F.mse_loss(reconstructed, tokens)

        relative_mse = residual_ss / tokens.square().sum().clamp_min(1e-12)

        centered_tokens = tokens - tokens.mean(dim=(0, 1), keepdim=True)
        fvu = residual_ss / centered_tokens.square().sum().clamp_min(1e-12)

        cosine_similarity = F.cosine_similarity(
            reconstructed,
            tokens,
            dim=-1,
        ).mean()

        l0 = (latents != 0).sum(dim=-1).float().mean()
        latent_density = l0 / latents.shape[-1]

        return {
            "loss": reconstruction_loss,
            "relative_mse": relative_mse,
            "fvu": fvu,
            "cosine_similarity": cosine_similarity,
            "l0": l0,
            "latent_density": latent_density,
        }

    def _shared_step(
        self,
        batch: Any,
        log_prefix: str,
    ) -> torch.Tensor:
        imgs = self._images_from_batch(batch)
        output = self(imgs)
        metrics = self._reconstruction_metrics(output)

        is_training = log_prefix == "train"
        self.log_dict(
            {
                f"{log_prefix}_loss_total": metrics["loss"],
                f"{log_prefix}_relative_mse": metrics["relative_mse"],
                f"{log_prefix}_fvu": metrics["fvu"],
                f"{log_prefix}_cosine_similarity": metrics["cosine_similarity"],
                f"{log_prefix}_l0": metrics["l0"],
                f"{log_prefix}_latent_density": metrics["latent_density"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=imgs.shape[0],
        )

        if is_training:
            self.log(
                "train_loss_total_step",
                metrics["loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=imgs.shape[0],
            )

        return metrics["loss"]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def eval_step(
        self,
        batch: Any,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
        log_prefix: str | None = None,
        is_notebook: bool = False,
    ):
        prefix = log_prefix or "val"

        if dataloader_idx is not None:
            prefix = f"{prefix}_{dataloader_idx}"

        if is_notebook:
            imgs = self._images_from_batch(batch)
            return self(imgs)

        return self._shared_step(batch, prefix)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure=None,
    ) -> None:
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )

        if self.normalize_decoder:
            self.network.sae.normalize_decoder_()

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                total_iters=max(1, int(self.trainer.estimated_stepping_batches)),
                power=self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
