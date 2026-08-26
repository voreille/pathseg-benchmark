from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from pathseg.models.builders import build_tiler
from pathseg.sae.encoder import EncoderWithTopKSAE
from pathseg.training.lightning_module import LightningModule


class TopKSAETraining(LightningModule):
    def __init__(
        self,
        encoder_id: str,
        num_latents: int,
        k: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
        tiler_name: Optional[str] = None,
        tiler_init_args: dict[str, Any] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = True,
        interpolate_latents: bool = False,
        normalize_decoder: bool = True,
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

        self.interpolate_latents = interpolate_latents
        self.poly_lr_decay_power = poly_lr_decay_power
        self.normalize_decoder = normalize_decoder
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        imgs, _, _, _ = batch

        output = self(imgs)
        tokens = output["tokens"].detach()
        reconstructed_tokens = output["reconstructed_tokens"]
        latents = output["latents"]

        residual = reconstructed_tokens - tokens
        residual_ss = residual.square().sum()

        loss_total = F.mse_loss(reconstructed_tokens, tokens)
        relative_mse = residual_ss / tokens.square().sum().clamp_min(1e-12)

        centered_tokens = tokens - tokens.mean(dim=(0, 1), keepdim=True)
        fvu = residual_ss / centered_tokens.square().sum().clamp_min(1e-12)

        cosine_similarity = F.cosine_similarity(
            reconstructed_tokens,
            tokens,
            dim=-1,
        ).mean()

        l0 = (latents != 0).sum(dim=-1).float().mean()

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_relative_mse", relative_mse, sync_dist=True)
        self.log("train_fvu", fvu, sync_dist=True)
        self.log("train_cosine_similarity", cosine_similarity, sync_dist=True)
        self.log("train_l0", l0, sync_dist=True)

        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, _, _, _ = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_output = self(crops)

        tokens = crop_output["tokens"].detach()
        reconstructed_tokens = crop_output["reconstructed_tokens"]
        sparse_latents = crop_output["latents"]

        residual = reconstructed_tokens - tokens
        residual_ss = residual.square().sum()

        loss_total = F.mse_loss(reconstructed_tokens, tokens)
        relative_mse = residual_ss / tokens.square().sum().clamp_min(1e-12)

        centered_tokens = tokens - tokens.mean(dim=(0, 1), keepdim=True)
        fvu = residual_ss / centered_tokens.square().sum().clamp_min(1e-12)

        cosine_similarity = F.cosine_similarity(
            reconstructed_tokens,
            tokens,
            dim=-1,
        ).mean()

        l0 = (sparse_latents != 0).sum(dim=-1).float().mean()

        self.log(
            f"{log_prefix}_{dataloader_idx}_loss_total",
            loss_total,
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_{dataloader_idx}_relative_mse",
            relative_mse,
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_{dataloader_idx}_fvu",
            fvu,
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_{dataloader_idx}_cosine_similarity",
            cosine_similarity,
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_{dataloader_idx}_l0",
            l0,
            sync_dist=True,
        )

        return loss_total

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
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, _, _, img_ids = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_output = self(crops)

        return crop_output["tokens"], crop_output["reconstructed_tokens"], crop_output["latents"], img_ids
