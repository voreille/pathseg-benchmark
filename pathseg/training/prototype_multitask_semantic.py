from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from pathseg.training.lightning_module import LightningModule
from pathseg.training.tiler import Tiler


class TwoHeadSemanticWithSupport(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        source_id_a: int,
        source_id_b: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        tiler: Optional[Tiler] = None,
        class_weights_a: Optional[Sequence[float]] = None,
        class_weights_b: Optional[Sequence[float]] = None,
        loss_weight_a: float = 1.0,
        loss_weight_b: float = 1.0,
        support_every_n_steps: int = 1,
        refresh_support_on_val: bool = False,
        num_classes_b: Optional[int] = None,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            tiler=tiler,
        )
        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = int(ignore_idx)
        self.poly_lr_decay_power = float(poly_lr_decay_power)
        self.source_id_a = int(source_id_a)
        self.source_id_b = int(source_id_b)
        self.loss_weight_a = float(loss_weight_a)
        self.loss_weight_b = float(loss_weight_b)
        self.support_every_n_steps = int(support_every_n_steps)
        self.refresh_support_on_val = bool(refresh_support_on_val)

        self.num_classes_a = network.num_compartments
        self.num_classes_b = num_classes_b
        if self.num_classes_b is None:
            raise ValueError("network.num_classes_b must be defined.")

        self.init_metrics_semantic(
            [self.num_classes_a, self.num_classes_b],
            self.ignore_idx,
            num_metrics,
        )

        w_a = (
            torch.tensor(class_weights_a, dtype=torch.float32)
            if class_weights_a is not None
            else None
        )
        w_b = (
            torch.tensor(class_weights_b, dtype=torch.float32)
            if class_weights_b is not None
            else None
        )

        self.criterion_a = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=w_a)
        self.register_buffer(
            "class_weights_b_tensor",
            w_b if w_b is not None else torch.empty(0),
            persistent=False,
        )

        self._support_iter_train = None
        self._support_iter_val = None
        self._support_iter_test = None

        self._cached_train_ctx = None
        self._cached_val_ctx = None
        self._cached_test_ctx = None

    def forward(self, x, ctx=None):
        return self.network(x / 255.0, ctx=ctx)

    # ------------------------------------------------------------------
    # Support helpers
    # ------------------------------------------------------------------
    def _get_support_loader(self, stage: str):
        dm = self.trainer.datamodule
        name = {
            "train": "train_support_dataloader",
            "val": "val_support_dataloader",
            "test": "test_support_dataloader",
        }[stage]

        if not hasattr(dm, name):
            raise AttributeError(f"Datamodule has no {name}()")
        return getattr(dm, name)()

    def _reset_support_iter(self, stage: str):
        loader = self._get_support_loader(stage)
        if stage == "train":
            self._support_iter_train = iter(loader)
        elif stage == "val":
            self._support_iter_val = iter(loader)
        elif stage == "test":
            self._support_iter_test = iter(loader)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _next_support_batch(self, stage: str):
        attr = {
            "train": "_support_iter_train",
            "val": "_support_iter_val",
            "test": "_support_iter_test",
        }[stage]

        support_iter = getattr(self, attr)
        if support_iter is None:
            self._reset_support_iter(stage)
            support_iter = getattr(self, attr)

        try:
            return next(support_iter)
        except StopIteration:
            self._reset_support_iter(stage)
            return next(getattr(self, attr))

    def _infer_support_target_mode(self, targets) -> str:
        if len(targets) == 0:
            raise ValueError("Empty support targets")

        first = targets[0]

        if isinstance(first, dict):
            return "semantic"

        if isinstance(first, (int,)):
            return "class"

        if torch.is_tensor(first) and first.ndim == 0:
            return "class"

        raise TypeError(f"Unsupported support target type: {type(first)}")

    def _prepare_support_pattern_targets(self, targets):
        mode = self._infer_support_target_mode(targets)

        if mode == "semantic":
            h_out, w_out = self.network.grid_size
            return self.preprocess_support_pattern_masks(
                targets=list(targets),
                out_hw=(h_out, w_out),
                num_classes=self.num_classes_b,
            )

        if mode == "class":
            return [int(t) for t in targets]

        raise RuntimeError(f"Unhandled mode: {mode}")

    @torch.compiler.disable
    def preprocess_support_pattern_masks(
        self,
        targets: list[dict],
        out_hw: tuple[int, int],
        num_classes: int,
    ) -> list[torch.Tensor]:
        h_out, w_out = out_hw
        support_targets = []

        for target in targets:
            masks = target["masks"]  # [N, H, W] bool
            labels = target["labels"]  # [N]
            device = labels.device

            h, w = masks.shape[-2:]
            weights = torch.zeros(
                (num_classes, h, w),
                dtype=torch.float32,
                device=device,
            )

            for mask, label in zip(masks, labels):
                class_id = int(label.item())
                weights[class_id] += mask.float()

            if (h, w) != (h_out, w_out):
                weights = F.interpolate(
                    weights.unsqueeze(0),
                    size=(h_out, w_out),
                    mode="area",
                ).squeeze(0)

            support_targets.append(weights)

        return support_targets

    @torch.no_grad()
    def _build_ctx(self, stage: str):
        batch = self._next_support_batch(stage)

        if len(batch) == 3:
            images, targets, _image_ids = batch
        elif len(batch) == 2:
            images, targets = batch
        else:
            raise ValueError(f"Unexpected support batch format: len={len(batch)}")

        pattern_targets = self._prepare_support_pattern_targets(targets)

        return self.network.fit_prototypes(
            images=list(images),
            pattern_targets=pattern_targets,
        )

    def _get_ctx(self, stage: str):
        if stage == "train":
            if (
                self._cached_train_ctx is None
                or self.global_step % self.support_every_n_steps == 0
            ):
                self._cached_train_ctx = self._build_ctx("train")
            return self._cached_train_ctx

        if stage == "val":
            if self._cached_val_ctx is None or self.refresh_support_on_val:
                self._cached_val_ctx = self._build_ctx("val")
            return self._cached_val_ctx

        if stage == "test":
            if self._cached_test_ctx is None:
                self._cached_test_ctx = self._build_ctx("test")
            return self._cached_test_ctx

        raise ValueError(f"Unknown stage: {stage}")

    def on_train_start(self):
        self._reset_support_iter("train")
        self._cached_train_ctx = None

    def on_validation_start(self):
        self._cached_val_ctx = None
        try:
            self._reset_support_iter("val")
        except AttributeError:
            self._support_iter_val = None

    def on_test_start(self):
        self._cached_test_ctx = None
        try:
            self._reset_support_iter("test")
        except AttributeError:
            self._support_iter_test = None

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _select(self, x, idx: torch.Tensor):
        if torch.is_tensor(x):
            return x[idx]
        return [v for v, m in zip(x, idx.tolist()) if m]

    def _targets_to_tensor(self, targets_list, device):
        targets_pp = self.to_per_pixel_targets_semantic(targets_list, self.ignore_idx)
        return torch.stack(targets_pp).long().to(device)

    def _loss_on_subset_fixed(
        self,
        logits: torch.Tensor,
        targets_list,
        subset_mask: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        if subset_mask.sum().item() == 0:
            return torch.zeros((), device=logits.device)

        logits_s = logits[subset_mask]
        targets_s = self._select(targets_list, subset_mask)
        targets_pp = self._targets_to_tensor(targets_s, logits.device)

        return criterion(logits_s, targets_pp)

    def _remap_targets_to_local_labels(
        self,
        targets_pp: torch.Tensor,
        local_label_ids: torch.Tensor,
    ) -> torch.Tensor:
        remapped = torch.full_like(targets_pp, fill_value=self.ignore_idx)
        for local_idx, global_label in enumerate(local_label_ids.tolist()):
            remapped[targets_pp == int(global_label)] = local_idx
        return remapped

    def _loss_on_subset_proto(
        self,
        logits_b: torch.Tensor,
        label_ids_b: torch.Tensor,
        targets_list,
        subset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if subset_mask.sum().item() == 0 or logits_b.shape[1] == 0:
            return torch.zeros((), device=logits_b.device)

        logits_s = logits_b[subset_mask]
        targets_s = self._select(targets_list, subset_mask)
        targets_pp = self._targets_to_tensor(targets_s, logits_b.device)
        local_targets = self._remap_targets_to_local_labels(targets_pp, label_ids_b)

        weight = None
        if self.class_weights_b_tensor.numel() > 0:
            weight = self.class_weights_b_tensor[label_ids_b].to(logits_b.device)

        return F.cross_entropy(
            logits_s,
            local_targets,
            ignore_index=self.ignore_idx,
            weight=weight,
        )

    def _expand_proto_logits_to_global(
        self,
        logits_b: torch.Tensor,
        label_ids_b: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = logits_b.shape
        full = logits_b.new_full(
            (int(b), int(self.num_classes_b), int(h), int(w)),
            -1e4,
        )
        full[:, label_ids_b, :, :] = logits_b
        return full

    def _select_eval_logits(self, out: dict, source_id: int) -> torch.Tensor:
        if source_id == self.source_id_a:
            return out["logits_a"]
        return self._expand_proto_logits_to_global(
            out["logits_b"],
            out["label_ids_b"],
        )

    # ------------------------------------------------------------------
    # Training / eval
    # ------------------------------------------------------------------
    def _get_loss_weight_b(self):
        step = self.global_step

        warmup_start = 1000
        warmup_end = 5000

        if step < warmup_start:
            return 0.0

        if step > warmup_end:
            return 1.0

        # linear ramp
        return (step - warmup_start) / (warmup_end - warmup_start)

    def training_step(self, batch, batch_idx):
        w_b = self._get_loss_weight_b() * self.loss_weight_b

        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        w_b = self._get_loss_weight_b()

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b
        if w_b == 0.0:
            out = self(imgs, ctx=None)
            logits_a = F.interpolate(out["logits_a"], self.img_size, mode="bilinear")

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = torch.zeros((), device=imgs.device)
            loss_total = self.loss_weight_a * loss_a
            label_ids_b = []
        else:
            ctx = self._get_ctx("train")
            out = self(imgs, ctx=ctx)

            logits_a = F.interpolate(out["logits_a"], self.img_size, mode="bilinear")
            logits_b = F.interpolate(out["logits_b"], self.img_size, mode="bilinear")
            label_ids_b = out["label_ids_b"]

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = self._loss_on_subset_proto(logits_b, label_ids_b, targets, m_b)
            loss_total = self.loss_weight_a * loss_a + w_b * loss_b

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True)
        self.log("train_loss_b", loss_b, sync_dist=True)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True)
        self.log(
            "train_num_support_labels",
            torch.tensor(len(label_ids_b), device=imgs.device, dtype=torch.float32),
            sync_dist=True,
        )
        self.log("loss_weight_b", w_b, prog_bar=False, sync_dist=True)

        return loss_total

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, targets, source_ids, image_ids = batch
        sid0 = int(source_ids[0])

        ctx = self._get_ctx("val" if log_prefix == "val" else "test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)
        crop_logits = self._select_eval_logits(crop_out, sid0)
        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")

        logits_list = self.revert_window_logits_semantic(
            crop_logits, origins, img_sizes
        )

        if is_notebook:
            return logits_list

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        self.update_metrics(logits_list, targets_pp, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(imgs[0], targets_pp[0], logits=logits_list[0])
            self.log_wandb_image(name, plot, commit=False)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_semantic("test")

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
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

    def init_metrics_semantic(
        self,
        num_classes: Union[int, Sequence[int]],
        ignore_idx: int,
        num_metrics: int | None = None,
    ) -> None:
        if isinstance(num_classes, int):
            if num_metrics is None:
                num_metrics = 1
            num_classes_list = [int(num_classes)] * int(num_metrics)
        else:
            num_classes_list = [int(x) for x in num_classes]
            if num_metrics is not None and int(num_metrics) != len(num_classes_list):
                raise ValueError(
                    f"num_metrics ({num_metrics}) must match len(num_classes) "
                    f"({len(num_classes_list)}) when num_classes is a list."
                )

        self.iou_metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=nc,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for nc in num_classes_list
            ]
        )
        self.f1_metrics = nn.ModuleList(
            [
                MulticlassF1Score(
                    num_classes=nc,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for nc in num_classes_list
            ]
        )

    @torch.compiler.disable
    def update_metrics(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_idx: int,
    ) -> None:
        for p, t in zip(preds, targets):
            p_in = p.unsqueeze(0)
            t_in = t.unsqueeze(0)
            self.iou_metrics[dataloader_idx].update(p_in, t_in)
            self.f1_metrics[dataloader_idx].update(p_in, t_in)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # support either:
        # (imgs, targets, source_ids, img_ids)
        # or (imgs, img_ids)
        # or (imgs, class_ids, img_ids)

        if len(batch) == 4:
            imgs, targets, source_ids, img_ids = batch
        elif len(batch) == 3:
            imgs, targets, img_ids = batch
            source_ids = None
        elif len(batch) == 2:
            imgs, img_ids = batch
            targets = None
            source_ids = None
        else:
            raise ValueError(f"Unexpected predict batch format: len={len(batch)}")

        imgs = torch.stack(list(imgs)).to(self.device)
        img_ids = list(img_ids)

        ctx = self._get_ctx("test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        if source_ids is None:
            crop_logits = self._expand_proto_logits_to_global(
                crop_out["logits_b"],
                crop_out["label_ids_b"],
            )
        else:
            sid0 = int(source_ids[0])
            crop_logits = self._select_eval_logits(crop_out, sid0)

        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits_list = self.revert_window_logits_semantic(
            crop_logits, origins, img_sizes
        )

        outs = []
        for i, logits in enumerate(logits_list):
            pred = torch.argmax(logits, dim=0)
            out_i = {
                "img_id": img_ids[i],
                "logits": logits.detach().cpu(),
                "pred": pred.detach().cpu(),
            }

            if "label_ids_b" in crop_out:
                out_i["label_ids_b"] = crop_out["label_ids_b"].detach().cpu()

            if targets is not None:
                t = targets[i]
                if torch.is_tensor(t):
                    out_i["target"] = t.detach().cpu()
                else:
                    out_i["target"] = t

            if source_ids is not None:
                out_i["source_id"] = int(source_ids[i])

            outs.append(out_i)

        return outs
