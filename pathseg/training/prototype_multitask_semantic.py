from __future__ import annotations

import io
from typing import Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from pathseg.training.histo_loss import BCEDiceLoss, CrossEntropyDiceLoss
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
        loss_weight_b_aux: float = 1.0,
        support_every_n_steps: int = 1,
        refresh_support_on_val: bool = False,
        num_classes_b: Optional[int] = None,
        use_fixed_loss_weight_b: Optional[float] = None,
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
        self.loss_weight_b_aux = float(loss_weight_b_aux)
        self.support_every_n_steps = int(support_every_n_steps)
        self.refresh_support_on_val = bool(refresh_support_on_val)
        self.use_fixed_loss_weight_b = (
            float(use_fixed_loss_weight_b)
            if use_fixed_loss_weight_b is not None
            else None
        )

        self.num_classes_a = network.num_compartments
        self.num_classes_b = num_classes_b
        if self.num_classes_b is None:
            raise ValueError("num_classes_b must be defined.")

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

        self.criterion_a = CrossEntropyDiceLoss(
            ignore_index=self.ignore_idx, weight=w_a
        )
        self.criterion_b_aux = CrossEntropyDiceLoss(
            ignore_index=self.ignore_idx, weight=w_b
        )

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
    def _infer_support_mode(self, targets) -> str:
        if len(targets) == 0:
            raise ValueError("Empty support targets")

        first = targets[0]

        if isinstance(first, dict):
            return "semantic"

        if isinstance(first, int):
            return "image_label"

        if torch.is_tensor(first) and first.ndim == 0:
            return "image_label"

        raise TypeError(f"Unsupported support target type: {type(first)}")

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

    @torch.compiler.disable
    def preprocess_support_pattern_masks(
        self,
        targets: list[dict],
        out_hw: tuple[int, int],
        num_classes: int,
    ) -> torch.Tensor:
        """
        Convert support annotations to dense soft masks [S, K, H, W].

        Each target is expected to contain:
          - target["masks"]:  [N, H, W] bool or float
          - target["labels"]: [N]
        """
        h_out, w_out = out_hw
        support_targets = []

        for target in targets:
            masks = target["masks"]  # [N,H,W]
            labels = target["labels"]  # [N]
            device = labels.device

            if masks.ndim != 3:
                raise ValueError("target['masks'] must have shape [N,H,W]")
            if labels.ndim != 1:
                raise ValueError("target['labels'] must have shape [N]")

            h, w = masks.shape[-2:]
            weights = torch.zeros(
                (num_classes, h, w),
                dtype=torch.float32,
                device=device,
            )

            for mask, label in zip(masks, labels):
                class_id = int(label.item())
                if not (0 <= class_id < num_classes):
                    raise ValueError(
                        f"support label {class_id} outside [0, {num_classes - 1}]"
                    )
                weights[class_id] += mask.float()

            if (h, w) != (h_out, w_out):
                weights = F.interpolate(
                    weights.unsqueeze(0),
                    size=(h_out, w_out),
                    mode="area",
                ).squeeze(0)

            support_targets.append(weights)

        return torch.stack(support_targets, dim=0)  # [S,K,H,W]

    @torch.no_grad()
    def _build_ctx(self, stage: str):
        batch = self._next_support_batch(stage)

        if len(batch) == 3:
            images, targets, _image_ids = batch
        elif len(batch) == 2:
            images, targets = batch
        else:
            raise ValueError(f"Unexpected support batch format: len={len(batch)}")

        if not torch.is_tensor(images):
            images = torch.stack(list(images), dim=0)
        images = images.to(self.device)

        mode = self._infer_support_mode(targets)

        if mode == "semantic":
            h_out, w_out = self.network.grid_size
            pattern_targets = self.preprocess_support_pattern_masks(
                targets=list(targets),
                out_hw=(h_out, w_out),
                num_classes=self.num_classes_b,
            )
            return self.network.fit_prototypes(
                support_images=images,
                pattern_targets=pattern_targets.to(self.device),
            )

        if mode == "image_label":
            image_labels = torch.as_tensor(
                targets, dtype=torch.long, device=self.device
            )

            return self.network.fit_prototypes_from_image_labels(
                images=images,
                image_labels=image_labels,
            )

        raise RuntimeError(f"Unhandled support mode: {mode}")

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
        logits_b: torch.Tensor,  # [B, num_classes_b, H, W]
        label_ids_b: torch.Tensor,  # supported labels this episode
        targets_list,
        subset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if subset_mask.sum().item() == 0:
            return torch.zeros((), device=logits_b.device)

        logits_s = logits_b[subset_mask]
        targets_s = self._select(targets_list, subset_mask)
        targets_pp = self._targets_to_tensor(targets_s, logits_b.device)

        supported = torch.zeros_like(targets_pp, dtype=torch.bool)
        for lab in label_ids_b.tolist():
            supported |= targets_pp == int(lab)

        # keep background too
        supported |= targets_pp == 0

        targets_masked = targets_pp.clone()
        targets_masked[~supported] = self.ignore_idx

        weight = None
        if self.class_weights_b_tensor.numel() > 0:
            weight = self.class_weights_b_tensor.to(logits_b.device)

        return F.cross_entropy(
            logits_s,
            targets_masked,
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

        return (step - warmup_start) / (warmup_end - warmup_start)

    def training_step(self, batch, batch_idx):
        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        if self.use_fixed_loss_weight_b is None:
            w_b = self._get_loss_weight_b() * self.loss_weight_b
        else:
            w_b = float(self.use_fixed_loss_weight_b)

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b

        if w_b == 0.0:
            out = self(imgs, ctx=None)
            logits_a = F.interpolate(
                out["logits_a"], self.img_size, mode="bilinear", align_corners=False
            )

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = torch.zeros((), device=imgs.device)
            loss_total = self.loss_weight_a * loss_a
            label_ids_b = torch.empty(0, dtype=torch.long, device=imgs.device)
        else:
            ctx = self._get_ctx("train")
            out = self(imgs, ctx=ctx)

            logits_a = F.interpolate(
                out["logits_a"], self.img_size, mode="bilinear", align_corners=False
            )
            logits_b = F.interpolate(
                out["logits_b"], self.img_size, mode="bilinear", align_corners=False
            )
            label_ids_b = out["label_ids_b"]

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = self._loss_on_subset_proto(logits_b, label_ids_b, targets, m_b)
            loss_total = self.loss_weight_a * loss_a + w_b * loss_b

        if out.get("logits_b_aux") is not None and m_b.any():
            logits_b_aux = F.interpolate(
                out["logits_b_aux"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            loss_b_aux = self._loss_on_subset_fixed(
                logits_b_aux,
                targets,
                m_b,
                self.criterion_b_aux,
            )
            self.log("train_loss_b_aux", loss_b_aux, sync_dist=True)
            loss_total = loss_total + self.loss_weight_b_aux * loss_b_aux

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True)
        self.log("train_loss_b", loss_b, sync_dist=True)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True)
        self.log(
            "train_num_support_labels",
            torch.tensor(
                label_ids_b.numel(),
                device=imgs.device,
                dtype=torch.float32,
            ),
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
    ):
        imgs, targets, source_ids, image_ids = batch
        sid0 = int(source_ids[0])

        ctx = self._get_ctx("val" if log_prefix == "val" else "test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)
        crop_logits_a = crop_out["logits_a"]
        crop_logits_b = crop_out["logits_b"]
        crop_logits_a = F.interpolate(
            crop_logits_a, self.img_size, mode="bilinear", align_corners=False
        )
        crop_logits_b = F.interpolate(
            crop_logits_b, self.img_size, mode="bilinear", align_corners=False
        )

        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a, origins, img_sizes
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b, origins, img_sizes
        )

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        if sid0 == self.source_id_a:
            self.update_metrics(logits_a_list, targets_pp, dataloader_idx)
        else:
            self.update_metrics(logits_b_list, targets_pp, dataloader_idx)

        if batch_idx == 0:
            img0 = imgs[0]
            tgt0 = targets_pp[0]

            plot = self.plot_two_head_semantic(
                img=img0,
                logits_a=logits_a_list[0],
                logits_b=None if logits_b_list is None else logits_b_list[0],
                target=tgt0,
                gt_head="a" if sid0 == self.source_id_a else "b",
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_two_head_{batch_idx}",
                plot,
                commit=False,
            )

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

        img_ids = list(img_ids)

        if source_ids is not None:
            if torch.is_tensor(source_ids):
                source_ids = source_ids.to(self.device)
            else:
                source_ids = torch.as_tensor(
                    source_ids, dtype=torch.long, device=self.device
                )

        ctx = self._get_ctx("test")

        # tile / crop
        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        # ---------------------------
        # Head B logits (task head)
        # ---------------------------
        if source_ids is None:
            crop_logits_b = self._expand_proto_logits_to_global(
                crop_out["logits_b"],
                crop_out["label_ids_b"],
            )
        else:
            sid0 = int(source_ids[0].item())
            crop_logits_b = self._select_eval_logits(crop_out, sid0)

        crop_logits_b = F.interpolate(
            crop_logits_b,
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b,
            origins,
            img_sizes,
        )

        # ---------------------------
        # Head A logits (compartments)
        # always stitch back to full image
        # ---------------------------
        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a,
            origins,
            img_sizes,
        )

        outs = []
        for i, (logits_b, logits_a) in enumerate(zip(logits_b_list, logits_a_list)):
            pred_b = torch.argmax(logits_b, dim=0)
            pred_a = torch.argmax(logits_a, dim=0)

            out_i = {
                "img_id": img_ids[i],
                "logits": logits_b.detach().cpu(),  # full-image head B logits
                "pred": pred_b.detach().cpu(),  # full-image head B prediction
                "logits_a": logits_a.detach().cpu(),  # full-image head A logits
                "pred_a": pred_a.detach().cpu(),  # full-image head A prediction
                "img_hw": tuple(int(x) for x in pred_b.shape[-2:]),
                "dataloader_idx": int(dataloader_idx),
                "img": imgs[i].detach().cpu(),
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
                out_i["source_id"] = int(source_ids[i].item())

            outs.append(out_i)

        return outs

    @torch.compiler.disable
    def plot_two_head_semantic(
        self,
        img: torch.Tensor,
        logits_a: torch.Tensor | None = None,
        logits_b: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        gt_head: str | None = None,
        cmap: str = "tab20",
    ):
        panels = [("Image", "img")]
        if target is not None:
            panels.append((f"GT ({gt_head})" if gt_head is not None else "GT", "gt"))
        if logits_a is not None:
            panels.append(("Head A", "a"))
        if logits_b is not None:
            panels.append(("Head B", "b"))

        ncols = len(panels)
        fig, axes = plt.subplots(
            1, ncols, figsize=(5 * ncols, 5), sharex=True, sharey=True
        )
        if ncols == 1:
            axes = [axes]

        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)

        target_np = None if target is None else target.detach().cpu().numpy()

        pred_a = None
        if logits_a is not None:
            pred_a = torch.argmax(logits_a, dim=0).detach().cpu().numpy()

        pred_b = None
        if logits_b is not None:
            pred_b = torch.argmax(logits_b, dim=0).detach().cpu().numpy()

        uniq_parts = []
        if target_np is not None:
            uniq_parts.append(np.unique(target_np))
        if pred_a is not None:
            uniq_parts.append(np.unique(pred_a))
        if pred_b is not None:
            uniq_parts.append(np.unique(pred_b))

        if len(uniq_parts) == 0:
            unique_classes = np.array([0], dtype=np.int64)
        else:
            unique_classes = np.unique(np.concatenate(uniq_parts))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))

        if self.ignore_idx in unique_classes:
            ignore_mask = unique_classes == self.ignore_idx
            colors[ignore_mask] = [0, 0, 0, 1]

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)

        def encode_mask(x: np.ndarray) -> np.ndarray:
            return np.digitize(x, unique_classes, right=False) - 1

        for ax, (title, kind) in zip(axes, panels):
            ax.set_title(title)
            ax.axis("off")

            if kind == "img":
                ax.imshow(img_np)
            elif kind == "gt":
                ax.imshow(
                    encode_mask(target_np),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind == "a":
                ax.imshow(
                    encode_mask(pred_a),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind == "b":
                ax.imshow(
                    encode_mask(pred_b),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]
        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)


class TwoHeadSemanticWithSupportWithPreseg(TwoHeadSemanticWithSupport):
    def __init__(
        self,
        *args,
        loss_weight_preseg: float = 1.0,
        fg_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss_weight_preseg = float(loss_weight_preseg)
        self.fg_threshold = float(fg_threshold)

        self.criterion_preseg = BCEDiceLoss(
            ignore_index=self.ignore_idx,
            smooth_labels=True,
        )
        self.criterion_b_proto = CrossEntropyDiceLoss(
            ignore_index=self.ignore_idx,
        )

    # ------------------------------------------------------------------
    # Targets / losses
    # ------------------------------------------------------------------
    def _make_pattern_only_targets(self, targets_pp: torch.Tensor) -> torch.Tensor:
        """
        Convert full semantic targets to pattern-only targets for head B.

        Input labels:
            0 = background
            1..K = pattern classes
            ignore_idx = ignore

        Output labels for head B:
            0..K-1 = pattern classes
            ignore_idx = background or ignore
        """
        targets_b = torch.full_like(targets_pp, fill_value=self.ignore_idx)

        valid_pattern = (targets_pp > 0) & (targets_pp != self.ignore_idx)
        targets_b[valid_pattern] = targets_pp[valid_pattern] - 1

        return targets_b

    def _loss_on_preseg(
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
        targets_pp = self._targets_to_tensor(targets_s, logits.device)  # [B,H,W]

        targets_bin = (targets_pp > 0).float()
        targets_bin[targets_pp == self.ignore_idx] = float(self.ignore_idx)
        targets_bin = targets_bin.unsqueeze(1)  # [B,1,H,W]

        return criterion(logits_s, targets_bin)

    def _loss_on_subset_proto_preseg(
        self,
        logits_b: torch.Tensor,  # [B, K, H, W], K = pattern classes only
        targets_list,
        subset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if subset_mask.sum().item() == 0:
            return torch.zeros((), device=logits_b.device)

        logits_s = logits_b[subset_mask]
        targets_s = self._select(targets_list, subset_mask)
        targets_pp = self._targets_to_tensor(targets_s, logits_b.device)  # [B,H,W]

        targets_b = self._make_pattern_only_targets(targets_pp)

        return self.criterion_b_proto(logits_s, targets_b)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _predict_b_from_logits(
        self,
        logits_b: torch.Tensor,  # [B,K,H,W]
        logits_preseg: torch.Tensor,  # [B,1,H,W]
        fg_threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Build final semantic prediction for task B:
            0 = background
            1..K = pattern classes
        """
        if fg_threshold is None:
            fg_threshold = self.fg_threshold

        fg_prob = torch.sigmoid(logits_preseg)  # [B,1,H,W]

        pred_patterns = torch.argmax(logits_b, dim=1) + 1  # [B,H,W]
        pred = pred_patterns.clone()
        pred[fg_prob[:, 0] < fg_threshold] = 0
        return pred

    def _predict_b_single(
        self,
        logits_b: torch.Tensor,  # [K,H,W]
        logits_preseg: torch.Tensor,  # [1,H,W]
        fg_threshold: float | None = None,
    ) -> torch.Tensor:
        if fg_threshold is None:
            fg_threshold = self.fg_threshold

        fg_prob = torch.sigmoid(logits_preseg)  # [1,H,W]
        pred_patterns = torch.argmax(logits_b, dim=0) + 1
        pred = pred_patterns.clone()
        pred[fg_prob[0] < fg_threshold] = 0
        return pred

    @torch.compiler.disable
    def update_metrics_with_predictions(
        self,
        preds: list[torch.Tensor],  # list of [H,W] predicted labels
        targets: list[torch.Tensor],  # list of [H,W] GT labels
        dataloader_idx: int,
    ) -> None:
        for p, t in zip(preds, targets):
            p_in = p.unsqueeze(0)
            t_in = t.unsqueeze(0)
            self.iou_metrics[dataloader_idx].update(p_in, t_in)
            self.f1_metrics[dataloader_idx].update(p_in, t_in)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        if self.use_fixed_loss_weight_b is None:
            w_b = self._get_loss_weight_b() * self.loss_weight_b
        else:
            w_b = float(self.use_fixed_loss_weight_b)

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b

        if w_b == 0.0:
            out = self(imgs, ctx=None)

            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = torch.zeros((), device=imgs.device)
            loss_total = self.loss_weight_a * loss_a
            label_ids_b = torch.empty(0, dtype=torch.long, device=imgs.device)
        else:
            ctx = self._get_ctx("train")
            out = self(imgs, ctx=ctx)

            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            logits_b = F.interpolate(
                out["logits_b"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )

            label_ids_b = out["label_ids_b"]

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = self._loss_on_subset_proto_preseg(logits_b, targets, m_b)
            loss_total = self.loss_weight_a * loss_a + w_b * loss_b

        if out.get("logits_b_aux") is not None and m_b.any():
            logits_b_aux = F.interpolate(
                out["logits_b_aux"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            loss_b_aux = self._loss_on_subset_fixed(
                logits_b_aux, targets, m_b, self.criterion_b_aux
            )
            self.log("train_loss_b_aux", loss_b_aux, sync_dist=True)
            loss_total = loss_total + self.loss_weight_b_aux * loss_b_aux

        logits_preseg = F.interpolate(
            out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        loss_preseg = self._loss_on_preseg(
            logits_preseg,
            targets,
            m_b,
            self.criterion_preseg,
        )
        self.log("train_loss_preseg", loss_preseg, sync_dist=True)

        loss_total = loss_total + self.loss_weight_preseg * loss_preseg

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True)
        self.log("train_loss_b", loss_b, sync_dist=True)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True)
        self.log(
            "train_num_support_labels",
            torch.tensor(
                label_ids_b.numel(),
                device=imgs.device,
                dtype=torch.float32,
            ),
            sync_dist=True,
        )
        self.log("loss_weight_b", w_b, prog_bar=False, sync_dist=True)

        return loss_total

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
    ):
        imgs, targets, source_ids, image_ids = batch
        sid0 = int(source_ids[0])

        ctx = self._get_ctx("val" if log_prefix == "val" else "test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_b = F.interpolate(
            crop_out["logits_b"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_preseg = F.interpolate(
            crop_out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a, origins, img_sizes
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b, origins, img_sizes
        )
        logits_preseg_list = self.revert_window_logits_semantic(
            crop_logits_preseg, origins, img_sizes
        )

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        if sid0 == self.source_id_a:
            self.update_metrics(logits_a_list, targets_pp, dataloader_idx)
        else:
            preds_b_list = [
                self._predict_b_single(lb, lp)
                for lb, lp in zip(logits_b_list, logits_preseg_list)
            ]
            self.update_metrics_with_predictions(
                preds_b_list, targets_pp, dataloader_idx
            )

        if batch_idx == 0:
            img0 = imgs[0]
            tgt0 = targets_pp[0]

            pred_b0 = self._predict_b_single(
                logits_b_list[0],
                logits_preseg_list[0],
            )
            plot = self.plot_two_head_semantic(
                img=img0,
                logits_a=logits_a_list[0],
                logits_b=None,
                pred_b=pred_b0,
                target=tgt0,
                gt_head="a" if sid0 == self.source_id_a else "b",
            )

            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_two_head_{batch_idx}",
                plot,
                commit=False,
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

        img_ids = list(img_ids)

        if source_ids is not None:
            if torch.is_tensor(source_ids):
                source_ids = source_ids.to(self.device)
            else:
                source_ids = torch.as_tensor(
                    source_ids, dtype=torch.long, device=self.device
                )

        ctx = self._get_ctx("test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_b = F.interpolate(
            crop_out["logits_b"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_preseg = F.interpolate(
            crop_out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a,
            origins,
            img_sizes,
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b,
            origins,
            img_sizes,
        )
        logits_preseg_list = self.revert_window_logits_semantic(
            crop_logits_preseg,
            origins,
            img_sizes,
        )

        outs = []
        for i, (logits_a, logits_b, logits_preseg) in enumerate(
            zip(logits_a_list, logits_b_list, logits_preseg_list)
        ):
            pred_a = torch.argmax(logits_a, dim=0)
            pred_b = self._predict_b_single(logits_b, logits_preseg)
            fg_prob = torch.sigmoid(logits_preseg)[0]

            out_i = {
                "img_id": img_ids[i],
                "logits_b": logits_b.detach().cpu(),
                "logits_preseg": logits_preseg.detach().cpu(),
                "fg_prob": fg_prob.detach().cpu(),
                "pred_b": pred_b.detach().cpu(),
                "logits_a": logits_a.detach().cpu(),
                "pred_a": pred_a.detach().cpu(),
                "img_hw": tuple(int(x) for x in pred_b.shape[-2:]),
                "dataloader_idx": int(dataloader_idx),
                "img": imgs[i].detach().cpu(),
            }

            if targets is not None:
                t = targets[i]
                out_i["target"] = t.detach().cpu() if torch.is_tensor(t) else t

            if source_ids is not None:
                out_i["source_id"] = int(source_ids[i].item())

            outs.append(out_i)

        return outs

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    @torch.compiler.disable
    def plot_two_head_semantic(
        self,
        img: torch.Tensor,
        logits_a: torch.Tensor | None = None,
        logits_b: torch.Tensor | None = None,
        pred_b: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        gt_head: str | None = None,
        cmap: str = "tab20",
    ):
        panels = [("Image", "img")]
        if target is not None:
            panels.append((f"GT ({gt_head})" if gt_head is not None else "GT", "gt"))
        if logits_a is not None:
            panels.append(("Head A", "a"))
        if pred_b is not None:
            panels.append(("Head B", "b"))
        elif logits_b is not None:
            panels.append(("Head B", "b_logits"))

        ncols = len(panels)
        fig, axes = plt.subplots(
            1, ncols, figsize=(5 * ncols, 5), sharex=True, sharey=True
        )
        if ncols == 1:
            axes = [axes]

        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)

        target_np = None if target is None else target.detach().cpu().numpy()

        pred_a = None
        if logits_a is not None:
            pred_a = torch.argmax(logits_a, dim=0).detach().cpu().numpy()

        pred_b_np = None
        if pred_b is not None:
            pred_b_np = pred_b.detach().cpu().numpy()
        elif logits_b is not None:
            pred_b_np = torch.argmax(logits_b, dim=0).detach().cpu().numpy()

        uniq_parts = []
        if target_np is not None:
            uniq_parts.append(np.unique(target_np))
        if pred_a is not None:
            uniq_parts.append(np.unique(pred_a))
        if pred_b_np is not None:
            uniq_parts.append(np.unique(pred_b_np))

        if len(uniq_parts) == 0:
            unique_classes = np.array([0], dtype=np.int64)
        else:
            unique_classes = np.unique(np.concatenate(uniq_parts))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))

        if self.ignore_idx in unique_classes:
            ignore_mask = unique_classes == self.ignore_idx
            colors[ignore_mask] = [0, 0, 0, 1]

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)

        def encode_mask(x: np.ndarray) -> np.ndarray:
            return np.digitize(x, unique_classes, right=False) - 1

        for ax, (title, kind) in zip(axes, panels):
            ax.set_title(title)
            ax.axis("off")

            if kind == "img":
                ax.imshow(img_np)
            elif kind == "gt":
                ax.imshow(
                    encode_mask(target_np),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind == "a":
                ax.imshow(
                    encode_mask(pred_a),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            elif kind in {"b", "b_logits"}:
                ax.imshow(
                    encode_mask(pred_b_np),
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]
        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)


class TwoHeadSemanticWithSupportWithPresegOpenSet(TwoHeadSemanticWithSupportWithPreseg):
    """
    Open-set episodic training with class dropout on the support set.

    Expected network API
    --------------------
    The network should be an open-set prototype decoder exposing:
      - fit_prototypes(...)
      - fit_prototypes_from_image_labels(...)
      - make_episode_pattern_targets_with_unknown(...)
      - forward(..., ctx=...) returning:
          out["logits_b"]        : [B, K_ep + 1, H, W]  (known classes + unknown)
          out["label_ids_b"]     : [K_ep] global labels kept in support
          out["unknown_index_b"] : scalar tensor = K_ep

    Semantics
    ---------
    - background is still handled by preseg
    - head B is trained only on foreground pixels:
        * supported fg classes -> mapped to episode-local labels [0..K_ep-1]
        * unsupported fg classes -> mapped to unknown class [K_ep]
        * background / ignore -> ignore_index
    """

    def __init__(
        self,
        *args,
        enable_class_dropout_train: bool = True,
        min_keep_classes: int = 2,
        keep_all_prob: float = 0.3,
        unknown_label_eval: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.enable_class_dropout_train = bool(enable_class_dropout_train)
        self.min_keep_classes = int(min_keep_classes)
        self.keep_all_prob = float(keep_all_prob)
        self.unknown_label_eval = int(unknown_label_eval)

        if self.min_keep_classes < 1:
            raise ValueError("min_keep_classes must be >= 1")

        if not hasattr(self.network, "make_episode_pattern_targets_with_unknown"):
            raise ValueError(
                "network must implement make_episode_pattern_targets_with_unknown()"
            )

    # ------------------------------------------------------------------
    # Support / class-dropout helpers
    # ------------------------------------------------------------------
    def _present_support_labels_from_pattern_targets(
        self,
        pattern_targets: torch.Tensor,  # [S, K+1, H, W], ch 0 = bg
    ) -> torch.Tensor:
        present = []
        num_classes = pattern_targets.shape[1]
        for c in range(1, num_classes):  # skip bg channel 0
            if torch.any(pattern_targets[:, c] > 0):
                present.append(c)
        if len(present) == 0:
            return torch.empty(0, dtype=torch.long, device=pattern_targets.device)
        return torch.tensor(present, dtype=torch.long, device=pattern_targets.device)

    def _sample_kept_support_labels(
        self,
        present_label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample which foreground classes remain visible in support for this episode.
        Sampling is done from the labels actually present in the current support batch.
        """
        m = int(present_label_ids.numel())
        if m == 0:
            return present_label_ids

        if (not self.enable_class_dropout_train) or (m <= self.min_keep_classes):
            return present_label_ids

        if torch.rand(()) < self.keep_all_prob:
            return present_label_ids

        min_keep = min(self.min_keep_classes, m)
        n_keep = int(torch.randint(low=min_keep, high=m, size=(1,)).item())

        perm = torch.randperm(m, device=present_label_ids.device)
        kept = present_label_ids[perm[:n_keep]]
        kept, _ = torch.sort(kept)
        return kept

    def _apply_support_class_dropout(
        self,
        pattern_targets: torch.Tensor,  # [S, K+1, H, W]
        kept_label_ids: torch.Tensor,  # global fg labels to keep
    ) -> torch.Tensor:
        """
        Zero out dropped foreground channels in support targets.
        Background channel 0 is kept unchanged.
        """
        dropped = pattern_targets.clone()

        keep_mask = torch.zeros(
            pattern_targets.shape[1],
            dtype=torch.bool,
            device=pattern_targets.device,
        )
        keep_mask[0] = True  # keep background channel
        if kept_label_ids.numel() > 0:
            keep_mask[kept_label_ids] = True

        dropped[:, ~keep_mask] = 0.0
        return dropped

    # ------------------------------------------------------------------
    # Override ctx building: class dropout only for train semantic support
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_ctx(self, stage: str):
        batch = self._next_support_batch(stage)

        if len(batch) == 3:
            images, targets, _image_ids = batch
        elif len(batch) == 2:
            images, targets = batch
        else:
            raise ValueError(f"Unexpected support batch format: len={len(batch)}")

        if not torch.is_tensor(images):
            images = torch.stack(list(images), dim=0)
        images = images.to(self.device)

        mode = self._infer_support_mode(targets)

        if mode == "semantic":
            h_out, w_out = self.network.grid_size
            pattern_targets = self.preprocess_support_pattern_masks(
                targets=list(targets),
                out_hw=(h_out, w_out),
                num_classes=self.num_classes_b,
            ).to(self.device)

            if stage == "train" and self.enable_class_dropout_train:
                present_label_ids = self._present_support_labels_from_pattern_targets(
                    pattern_targets
                )
                kept_label_ids = self._sample_kept_support_labels(present_label_ids)
                pattern_targets = self._apply_support_class_dropout(
                    pattern_targets=pattern_targets,
                    kept_label_ids=kept_label_ids,
                )

            ctx = self.network.fit_prototypes(
                support_images=images,
                pattern_targets=pattern_targets,
            )
            return ctx

        if mode == "image_label":
            # For image-label support, class dropout is also possible, but your current
            # setup seems semantic-mask based, so keep it simple here.
            image_labels = torch.as_tensor(
                targets, dtype=torch.long, device=self.device
            )

            if stage == "train" and self.enable_class_dropout_train:
                present = torch.unique(image_labels, sorted=True)
                kept = self._sample_kept_support_labels(present)
                keep_mask = torch.zeros_like(image_labels, dtype=torch.bool)
                for lab in kept.tolist():
                    keep_mask |= image_labels == int(lab)

                images = images[keep_mask]
                image_labels = image_labels[keep_mask]

            return self.network.fit_prototypes_from_image_labels(
                images=images,
                image_labels=image_labels,
            )

        raise RuntimeError(f"Unhandled support mode: {mode}")

    # ------------------------------------------------------------------
    # Open-set B loss
    # ------------------------------------------------------------------
    def _loss_on_subset_proto_preseg_open_set(
        self,
        logits_b: torch.Tensor,  # [B, K_ep + 1, H, W]
        label_ids_b: torch.Tensor,  # [K_ep] global support labels kept this episode
        targets_list,
        subset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if subset_mask.sum().item() == 0:
            return torch.zeros((), device=logits_b.device)

        logits_s = logits_b[subset_mask]
        targets_s = self._select(targets_list, subset_mask)
        targets_pp = self._targets_to_tensor(targets_s, logits_b.device)  # [B,H,W]

        # network remaps:
        #   supported fg -> 0..K_ep-1
        #   unsupported fg -> K_ep (unknown)
        #   background / ignore -> ignore_idx
        targets_local = self.network.make_episode_pattern_targets_with_unknown(
            pattern_targets=targets_pp,
            support_label_ids=label_ids_b,
            ignore_index=self.ignore_idx,
        )

        return self.criterion_b_proto(logits_s, targets_local)

    # ------------------------------------------------------------------
    # Prediction helpers for episodic open-set logits
    # ------------------------------------------------------------------
    def _predict_b_single_open_set(
        self,
        logits_b: torch.Tensor,  # [K_ep + 1, H, W]
        logits_preseg: torch.Tensor,  # [1, H, W]
        label_ids_b: torch.Tensor,  # [K_ep] global labels
        fg_threshold: float | None = None,
        unknown_label: int | None = None,
    ) -> torch.Tensor:
        """
        Returns global pattern labels:
          0 = background
          label_ids_b[i] = known class i
          unknown -> unknown_label (default self.unknown_label_eval)
        """
        if fg_threshold is None:
            fg_threshold = self.fg_threshold
        if unknown_label is None:
            unknown_label = self.unknown_label_eval

        fg_prob = torch.sigmoid(logits_preseg)[0]  # [H,W]
        pred_local = torch.argmax(logits_b, dim=0)  # [H,W]
        unknown_idx = logits_b.shape[0] - 1

        pred = torch.full_like(pred_local, fill_value=unknown_label)

        # map known local ids -> global labels
        for local_idx, global_lab in enumerate(label_ids_b.tolist()):
            pred[pred_local == int(local_idx)] = int(global_lab)

        # unknown channel
        pred[pred_local == unknown_idx] = int(unknown_label)

        # preseg decides background
        pred[fg_prob < fg_threshold] = 0
        return pred

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        if self.use_fixed_loss_weight_b is None:
            w_b = self._get_loss_weight_b() * self.loss_weight_b
        else:
            w_b = float(self.use_fixed_loss_weight_b)

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b

        if w_b == 0.0:
            out = self(imgs, ctx=None)

            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = torch.zeros((), device=imgs.device)
            loss_total = self.loss_weight_a * loss_a
            label_ids_b = torch.empty(0, dtype=torch.long, device=imgs.device)
        else:
            ctx = self._get_ctx("train")
            out = self(imgs, ctx=ctx)

            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            logits_b = F.interpolate(
                out["logits_b"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )  # [B, K_ep + 1, H, W]

            label_ids_b = out["label_ids_b"]

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = self._loss_on_subset_proto_preseg_open_set(
                logits_b=logits_b,
                label_ids_b=label_ids_b,
                targets_list=targets,
                subset_mask=m_b,
            )
            loss_total = self.loss_weight_a * loss_a + w_b * loss_b

        if out.get("logits_b_aux") is not None and m_b.any():
            logits_b_aux = F.interpolate(
                out["logits_b_aux"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            loss_b_aux = self._loss_on_subset_fixed(
                logits_b_aux,
                targets,
                m_b,
                self.criterion_b_aux,
            )
            self.log("train_loss_b_aux", loss_b_aux, sync_dist=True)
            loss_total = loss_total + self.loss_weight_b_aux * loss_b_aux

        logits_preseg = F.interpolate(
            out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        loss_preseg = self._loss_on_preseg(
            logits_preseg,
            targets,
            m_b,
            self.criterion_preseg,
        )
        self.log("train_loss_preseg", loss_preseg, sync_dist=True)

        loss_total = loss_total + self.loss_weight_preseg * loss_preseg

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True)
        self.log("train_loss_b", loss_b, sync_dist=True)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True)
        self.log(
            "train_num_support_labels",
            torch.tensor(
                label_ids_b.numel(),
                device=imgs.device,
                dtype=torch.float32,
            ),
            sync_dist=True,
        )
        self.log("loss_weight_b", w_b, prog_bar=False, sync_dist=True)

        return loss_total

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
    ):
        imgs, targets, source_ids, image_ids = batch
        sid0 = int(source_ids[0])

        # no class dropout here because _build_ctx only drops for stage=="train"
        ctx = self._get_ctx("val" if log_prefix == "val" else "test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_b = F.interpolate(
            crop_out["logits_b"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_preseg = F.interpolate(
            crop_out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a, origins, img_sizes
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b, origins, img_sizes
        )
        logits_preseg_list = self.revert_window_logits_semantic(
            crop_logits_preseg, origins, img_sizes
        )

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        if sid0 == self.source_id_a:
            self.update_metrics(logits_a_list, targets_pp, dataloader_idx)
        else:
            label_ids_b = crop_out["label_ids_b"]
            preds_b_list = [
                self._predict_b_single_open_set(lb, lp, label_ids_b=label_ids_b)
                for lb, lp in zip(logits_b_list, logits_preseg_list)
            ]
            self.update_metrics_with_predictions(
                preds_b_list, targets_pp, dataloader_idx
            )

        if batch_idx == 0:
            img0 = imgs[0]
            tgt0 = targets_pp[0]

            if sid0 == self.source_id_a:
                pred_b0 = None
            else:
                pred_b0 = self._predict_b_single_open_set(
                    logits_b_list[0],
                    logits_preseg_list[0],
                    label_ids_b=crop_out["label_ids_b"],
                )

            plot = self.plot_two_head_semantic(
                img=img0,
                logits_a=logits_a_list[0],
                logits_b=None,
                pred_b=pred_b0,
                target=tgt0,
                gt_head="a" if sid0 == self.source_id_a else "b",
            )

            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_two_head_{batch_idx}",
                plot,
                commit=False,
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

        img_ids = list(img_ids)

        if source_ids is not None:
            if torch.is_tensor(source_ids):
                source_ids = source_ids.to(self.device)
            else:
                source_ids = torch.as_tensor(
                    source_ids, dtype=torch.long, device=self.device
                )

        ctx = self._get_ctx("test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_b = F.interpolate(
            crop_out["logits_b"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        crop_logits_preseg = F.interpolate(
            crop_out["logits_preseg"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        logits_a_list = self.revert_window_logits_semantic(
            crop_logits_a, origins, img_sizes
        )
        logits_b_list = self.revert_window_logits_semantic(
            crop_logits_b, origins, img_sizes
        )
        logits_preseg_list = self.revert_window_logits_semantic(
            crop_logits_preseg, origins, img_sizes
        )

        label_ids_b = crop_out["label_ids_b"]

        outs = []
        for i, (logits_a, logits_b, logits_preseg) in enumerate(
            zip(logits_a_list, logits_b_list, logits_preseg_list)
        ):
            pred_a = torch.argmax(logits_a, dim=0)
            pred_b = self._predict_b_single_open_set(
                logits_b,
                logits_preseg,
                label_ids_b=label_ids_b,
            )
            fg_prob = torch.sigmoid(logits_preseg)[0]

            out_i = {
                "img_id": img_ids[i],
                "logits_b": logits_b.detach().cpu(),  # episodic logits incl unknown
                "logits_preseg": logits_preseg.detach().cpu(),
                "fg_prob": fg_prob.detach().cpu(),
                "pred_b": pred_b.detach().cpu(),  # global labels + unknown_label_eval
                "label_ids_b": label_ids_b.detach().cpu(),
                "logits_a": logits_a.detach().cpu(),
                "pred_a": pred_a.detach().cpu(),
                "img_hw": tuple(int(x) for x in pred_b.shape[-2:]),
                "dataloader_idx": int(dataloader_idx),
                "img": imgs[i].detach().cpu(),
            }

            if targets is not None:
                t = targets[i]
                out_i["target"] = t.detach().cpu() if torch.is_tensor(t) else t

            if source_ids is not None:
                out_i["source_id"] = int(source_ids[i].item())

            outs.append(out_i)

        return outs


class TwoHeadSemanticWithSupportTumorOnly(TwoHeadSemanticWithSupport):
    def __init__(
        self,
        *args,
        tumor_compartment_ids: Optional[list[int]] = None,
        tumor_threshold: float = 0.5,
        use_fixed_loss_weight_b: Optional[float] = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if tumor_compartment_ids is None:
            tumor_compartment_ids = [1]

        self.tumor_compartment_ids = torch.tensor(
            tumor_compartment_ids, dtype=torch.long
        )
        self.tumor_threshold = float(tumor_threshold)
        self.use_fixed_loss_weight_b = use_fixed_loss_weight_b

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_tumor_prob(
        self,
        logits_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits_a:
            [B, Ca, H, W] -> returns [B, H, W]
            [Ca, H, W]    -> returns [H, W]
        """
        tumor_ids = self.tumor_compartment_ids.to(logits_a.device)

        if logits_a.ndim == 4:
            probs_a = F.softmax(logits_a, dim=1)
            return probs_a[:, tumor_ids].sum(dim=1)

        if logits_a.ndim == 3:
            probs_a = F.softmax(logits_a, dim=0)
            return probs_a[tumor_ids].sum(dim=0)

        raise ValueError(f"Unexpected logits_a shape: {tuple(logits_a.shape)}")

    def _loss_on_subset_proto_conditioned(
        self,
        logits_a: torch.Tensor,
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

        # no grad to head A from loss B
        prob_tumor = self._get_tumor_prob(logits_a.detach())[subset_mask]

        targets_pp = targets_pp.clone()
        targets_pp[targets_pp == 0] = self.ignore_idx
        local_targets = self._remap_targets_to_local_labels(targets_pp, label_ids_b)

        weight = None
        if self.class_weights_b_tensor.numel() > 0:
            weight = self.class_weights_b_tensor[label_ids_b].to(logits_b.device)

        loss = F.cross_entropy(
            logits_s,
            local_targets,
            ignore_index=self.ignore_idx,
            weight=weight,
            reduction="none",
        )

        loss = loss * prob_tumor
        denom = prob_tumor.sum().clamp_min(1e-6)
        return loss.sum() / denom

    def _force_bg_outside_tumor(
        self,
        logits_b: torch.Tensor,  # [C,H,W] or [B,C,H,W], global classes, bg=0
        logits_a: torch.Tensor,  # [Ca,H,W] or [B,Ca,H,W]
    ) -> torch.Tensor:
        logits_b = logits_b.clone()
        prob_tumor = self._get_tumor_prob(logits_a)
        outside = prob_tumor < self.tumor_threshold

        if logits_b.ndim == 3:
            logits_b[1:, outside] = -1e4
            logits_b[0, outside] = 0.0
            return logits_b

        if logits_b.ndim == 4:
            logits_b[:, 1:, :, :] = logits_b[:, 1:, :, :].masked_fill(
                outside.unsqueeze(1), -1e4
            )
            bg = logits_b[:, 0]
            bg[outside] = 0.0
            logits_b[:, 0] = bg
            return logits_b

        raise ValueError(f"Expected 3D or 4D logits_b, got {tuple(logits_b.shape)}")

    def _stitch_head_a_logits(
        self,
        crop_out: dict,
        origins,
        img_sizes,
    ) -> list[torch.Tensor]:
        crop_logits_a = F.interpolate(
            crop_out["logits_a"],
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.revert_window_logits_semantic(crop_logits_a, origins, img_sizes)

    def _stitch_head_b_logits(
        self,
        crop_out: dict,
        origins,
        img_sizes,
    ) -> list[torch.Tensor]:
        crop_logits_b = self._expand_proto_logits_to_global(
            crop_out["logits_b"],
            crop_out["label_ids_b"],
        )
        crop_logits_b = F.interpolate(
            crop_logits_b,
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.revert_window_logits_semantic(crop_logits_b, origins, img_sizes)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        imgs, targets, source_ids, _image_ids = batch
        source_ids = source_ids.to(imgs.device)

        if self.use_fixed_loss_weight_b is None:
            w_b = self._get_loss_weight_b() * self.loss_weight_b
        else:
            w_b = float(self.use_fixed_loss_weight_b)

        m_a = source_ids == self.source_id_a
        m_b = source_ids == self.source_id_b

        if w_b == 0.0:
            out = self(imgs, ctx=None)
            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = torch.zeros((), device=imgs.device)
            loss_total = self.loss_weight_a * loss_a
            label_ids_b = torch.empty(0, dtype=torch.long, device=imgs.device)
        else:
            ctx = self._get_ctx("train")
            out = self(imgs, ctx=ctx)

            logits_a = F.interpolate(
                out["logits_a"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            logits_b = F.interpolate(
                out["logits_b"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            label_ids_b = out["label_ids_b"]

            loss_a = self._loss_on_subset_fixed(
                logits_a, targets, m_a, self.criterion_a
            )
            loss_b = self._loss_on_subset_proto_conditioned(
                logits_a=logits_a,
                logits_b=logits_b,
                label_ids_b=label_ids_b,
                targets_list=targets,
                subset_mask=m_b,
            )
            loss_total = self.loss_weight_a * loss_a + w_b * loss_b

        if out.get("logits_b_aux") is not None and m_b.any():
            logits_b_aux = F.interpolate(
                out["logits_b_aux"],
                self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            loss_b_aux = self._loss_on_subset_fixed(
                logits_b_aux,
                targets,
                m_b,
                self.criterion_b_aux,
            )
            self.log("train_loss_b_aux", loss_b_aux, sync_dist=True)
            loss_total = loss_total + self.loss_weight_b_aux * loss_b_aux

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_a", loss_a, sync_dist=True)
        self.log("train_loss_b", loss_b, sync_dist=True)
        self.log("train_frac_a", m_a.float().mean(), sync_dist=True)
        self.log("train_frac_b", m_b.float().mean(), sync_dist=True)
        self.log(
            "train_num_support_labels",
            torch.tensor(
                label_ids_b.numel(),
                device=imgs.device,
                dtype=torch.float32,
            ),
            sync_dist=True,
        )
        self.log("loss_weight_b", w_b, prog_bar=False, sync_dist=True)

        return loss_total

    # ------------------------------------------------------------------
    # Eval / Test
    # ------------------------------------------------------------------
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook: bool = False,
    ):
        imgs, targets, source_ids, image_ids = batch
        source_ids = torch.as_tensor(source_ids, device=self.device)
        sid0 = int(source_ids[0].item())

        ctx = self._get_ctx("val" if log_prefix == "val" else "test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        logits_a_list = self._stitch_head_a_logits(crop_out, origins, img_sizes)

        logits_b_list = None
        if "logits_b" in crop_out and "label_ids_b" in crop_out:
            logits_b_list = self._stitch_head_b_logits(crop_out, origins, img_sizes)
            logits_b_list = [
                self._force_bg_outside_tumor(lb, la)
                for lb, la in zip(logits_b_list, logits_a_list)
            ]

        if is_notebook:
            return {
                "logits_a_list": logits_a_list,
                "logits_b_list": logits_b_list,
            }

        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        if sid0 == self.source_id_a:
            self.update_metrics(logits_a_list, targets_pp, dataloader_idx)
        else:
            if logits_b_list is None:
                raise RuntimeError(
                    "Expected head B logits during eval for source_id_b."
                )
            self.update_metrics(logits_b_list, targets_pp, dataloader_idx)

        if batch_idx == 0:
            img0 = imgs[0]
            tgt0 = targets_pp[0]

            plot = self.plot_two_head_semantic(
                img=img0,
                logits_a=logits_a_list[0],
                logits_b=None if logits_b_list is None else logits_b_list[0],
                target=tgt0,
                gt_head="a" if sid0 == self.source_id_a else "b",
            )
            self.log_wandb_image(
                f"{log_prefix}_{dataloader_idx}_two_head_{batch_idx}",
                plot,
                commit=False,
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

        img_ids = list(img_ids)

        if source_ids is not None:
            source_ids = torch.as_tensor(source_ids, device=self.device)

        ctx = self._get_ctx("test")

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_out = self(crops, ctx=ctx)

        logits_a_list = self._stitch_head_a_logits(crop_out, origins, img_sizes)
        logits_b_list = self._stitch_head_b_logits(crop_out, origins, img_sizes)
        logits_b_list = [
            self._force_bg_outside_tumor(lb, la)
            for lb, la in zip(logits_b_list, logits_a_list)
        ]

        outs = []
        for i, (logits_b, logits_a_i) in enumerate(zip(logits_b_list, logits_a_list)):
            pred_b = torch.argmax(logits_b, dim=0)
            pred_a = torch.argmax(logits_a_i, dim=0)

            out_i = {
                "img_id": img_ids[i],
                "logits": logits_b.detach().cpu(),
                "pred": pred_b.detach().cpu(),
                "logits_a": logits_a_i.detach().cpu(),
                "pred_a": pred_a.detach().cpu(),
                "img_hw": tuple(int(x) for x in pred_b.shape[-2:]),
                "dataloader_idx": int(dataloader_idx),
                "img": imgs[i].detach().cpu(),
            }

            if "label_ids_b" in crop_out:
                out_i["label_ids_b"] = crop_out["label_ids_b"].detach().cpu()

            if targets is not None:
                t = targets[i]
                out_i["target"] = t.detach().cpu() if torch.is_tensor(t) else t

            if source_ids is not None:
                out_i["source_id"] = int(source_ids[i].item())

            outs.append(out_i)

        return outs
