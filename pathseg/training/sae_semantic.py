from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from pathseg.models.builders import build_tiler
from pathseg.models.builders_semantic import build_sae_semantic_segmenter
from pathseg.models.checkpoints import (
    load_checkpoint_submodule,
    load_lightning_state_dict,
    load_submodule_state_dict,
)
from pathseg.training.semantic_common import (
    SemanticLightningModule,
    parse_task_specs,
)


class TopKSAESemanticTraining(SemanticLightningModule):
    """Train an SAE while monitoring a frozen semantic decoder.

    The encoder and decoder are frozen. Only ``network.sae`` is optimized.
    Validation reports both original-token and reconstructed-token semantic
    metrics using the same decoder and the same image batches.
    """

    def __init__(
        self,
        encoder_class_path: str,
        decoder_name: str,
        sae_class_path: str,
        tasks: dict[str, dict[str, Any]],
        ignore_idx: int,
        img_size: tuple[int, int],
        encoder_init_args: dict[str, Any] | None = None,
        decoder_init_args: dict[str, Any] | None = None,
        sae_init_args: dict[str, Any] | None = None,
        semantic_init_checkpoint_path: str | None = None,
        semantic_encoder_prefix: str = "network.encoder",
        semantic_decoder_prefix: str = "network.decoder",
        sae_init_checkpoint_path: str | None = None,
        sae_prefix: str = "network.sae",
        tiler_name: Optional[str] = None,
        tiler_init_args: dict[str, Any] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        poly_lr_decay_power: float = 0.9,
        upsample_logits: bool = False,
        interpolation_mode: str = "bilinear",
        normalize_decoder: bool = True,
    ) -> None:
        task_specs = parse_task_specs(tasks)
        network = build_sae_semantic_segmenter(
            encoder_class_path=encoder_class_path,
            encoder_init_args=encoder_init_args,
            decoder_name=decoder_name,
            decoder_init_args=decoder_init_args,
            num_classes_by_task={
                name: spec.num_classes for name, spec in task_specs.items()
            },
            sae_class_path=sae_class_path,
            sae_init_args=sae_init_args,
            upsample_logits=upsample_logits,
            interpolation_mode=interpolation_mode,
        )

        if semantic_init_checkpoint_path is not None:
            # A semantic checkpoint can be large. Read it once, then select the
            # encoder and decoder prefixes independently.
            semantic_state_dict = load_lightning_state_dict(
                semantic_init_checkpoint_path
            )
            load_submodule_state_dict(
                network.encoder,
                semantic_state_dict,
                source_prefix=semantic_encoder_prefix,
                strict=True,
            )
            load_submodule_state_dict(
                network.decoder,
                semantic_state_dict,
                source_prefix=semantic_decoder_prefix,
                strict=True,
            )
            del semantic_state_dict

        if sae_init_checkpoint_path is not None:
            load_checkpoint_submodule(
                network.sae,
                sae_init_checkpoint_path,
                source_prefix=sae_prefix,
                strict=True,
            )

        network.encoder.requires_grad_(False)
        network.decoder.requires_grad_(False)
        network.sae.requires_grad_(True)

        tiler = build_tiler(
            tiler_name=tiler_name,
            tiler_init_args=tiler_init_args,
        )

        super().__init__(
            network=network,
            tasks=tasks,
            ignore_idx=ignore_idx,
            img_size=img_size,
            freeze_encoder=True,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=1.0,
            poly_lr_decay_power=poly_lr_decay_power,
            tiler=tiler,
        )

        if normalize_decoder and not hasattr(self.network.sae, "normalize_decoder_"):
            raise AttributeError(
                "normalize_decoder=True requires sae.normalize_decoder_()."
            )

        self.normalize_decoder = bool(normalize_decoder)
        self.original_iou_metrics, self.original_f1_metrics = (
            self._make_metric_streams()
        )

        # The launcher expands the semantic run config into these ordinary
        # architecture arguments. Initialization paths are deliberately omitted
        # so a completed SAE checkpoint can reconstruct itself without the
        # original semantic checkpoint or config file.
        self.save_hyperparameters(
            ignore=[
                "semantic_init_checkpoint_path",
                "sae_init_checkpoint_path",
            ]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning recursively calls train(); keep both evaluation instruments
        # deterministic while the SAE remains trainable.
        self.network.encoder.eval()
        self.network.decoder.eval()
        return self

    @staticmethod
    def _sae_statistics(output: dict[str, torch.Tensor]):
        tokens = output["tokens"].detach()
        reconstructed_tokens = output["reconstructed_tokens"]
        latents = output["latents"]

        residual = reconstructed_tokens - tokens
        residual_ss = residual.square().sum()
        loss = F.mse_loss(reconstructed_tokens, tokens)
        relative_mse = residual_ss / tokens.square().sum().clamp_min(1e-12)

        centered_tokens = tokens - tokens.mean(dim=(0, 1), keepdim=True)
        fvu = residual_ss / centered_tokens.square().sum().clamp_min(1e-12)
        cosine_similarity = F.cosine_similarity(
            reconstructed_tokens,
            tokens,
            dim=-1,
        ).mean()
        l0 = (latents != 0).sum(dim=-1).float().mean()

        return {
            "loss_total": loss,
            "relative_mse": relative_mse,
            "fvu": fvu,
            "cosine_similarity": cosine_similarity,
            "l0": l0,
        }

    def _log_sae_statistics(
        self,
        statistics: dict[str, torch.Tensor],
        prefix: str,
        *,
        batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        for name, value in statistics.items():
            self.log(
                f"{prefix}_{name}",
                value,
                sync_dist=True,
                prog_bar=prog_bar and name == "loss_total",
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        imgs, _targets, _source_ids, _image_ids = self.unpack_batch(batch)
        output = self.network.forward_sae(imgs / 255.0)
        statistics = self._sae_statistics(output)
        self._log_sae_statistics(
            statistics, "train", batch_size=imgs.shape[0], prog_bar=True
        )
        return statistics["loss_total"]

    def eval_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        log_prefix: str,
    ):
        imgs, targets, _source_ids, _image_ids = self.unpack_batch(batch)
        task = self.eval_task_names[dataloader_idx]

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        plot_all_heads = batch_idx == 0
        output = self.network.forward_with_aux(
            crops / 255.0,
            task=None if plot_all_heads else task,
            include_original_logits=True,
        )

        statistics = self._sae_statistics(output)
        self._log_sae_statistics(
            statistics,
            f"{log_prefix}_{dataloader_idx}_{task}",
            batch_size=len(imgs),
        )

        reconstructed_logits_by_task = self.stitch_logits_by_task(
            output["logits"],
            origins,
            img_sizes,
        )
        original_logits_by_task = self.stitch_logits_by_task(
            output["original_logits"],
            origins,
            img_sizes,
        )
        reconstructed_logits = reconstructed_logits_by_task[task]
        original_logits = original_logits_by_task[task]
        target_maps = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        self.update_metric_stream(
            self.iou_metrics,
            self.f1_metrics,
            reconstructed_logits,
            target_maps,
            dataloader_idx,
        )
        self.update_metric_stream(
            self.original_iou_metrics,
            self.original_f1_metrics,
            original_logits,
            target_maps,
            dataloader_idx,
        )

        if plot_all_heads:
            reconstructed_plot = self.plot_semantic_heads(
                imgs[0],
                target_maps[0],
                {
                    head_task: head_logits[0]
                    for head_task, head_logits in (reconstructed_logits_by_task.items())
                },
                target_task=task,
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_{task}_reconstructed_all_heads_pred",
                reconstructed_plot,
                commit=False,
            )

            original_plot = self.plot_semantic_heads(
                imgs[0],
                target_maps[0],
                {
                    head_task: head_logits[0]
                    for head_task, head_logits in original_logits_by_task.items()
                },
                target_task=task,
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_{task}_original_all_heads_pred",
                original_plot,
                commit=False,
            )

        return {
            "reconstructed_logits": reconstructed_logits,
            "original_logits": original_logits,
        }

    def on_validation_epoch_end(self) -> None:
        self.finish_metric_stream(
            "val_reconstructed",
            self.iou_metrics,
            self.f1_metrics,
        )
        self.finish_metric_stream(
            "val_original",
            self.original_iou_metrics,
            self.original_f1_metrics,
        )

    def on_test_epoch_end(self) -> None:
        self.finish_metric_stream(
            "test_reconstructed",
            self.iou_metrics,
            self.f1_metrics,
        )
        self.finish_metric_stream(
            "test_original",
            self.original_iou_metrics,
            self.original_f1_metrics,
        )

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
