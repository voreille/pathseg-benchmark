from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from lightning.pytorch.callbacks import Callback


def _make_discrete_cmap(num_classes: int) -> ListedColormap:
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.arange(num_classes))
    return ListedColormap(colors)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)


def _confusion_from_arrays(
    pred: np.ndarray,
    tgt: np.ndarray,
    num_classes: int,
    ignore_idx: int | None = None,
) -> np.ndarray:
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, tgt={tgt.shape}")

    valid = np.ones_like(tgt, dtype=bool)
    if ignore_idx is not None:
        valid &= tgt != ignore_idx

    p = pred[valid].astype(np.int64)
    t = tgt[valid].astype(np.int64)

    keep = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    p = p[keep]
    t = t[keep]

    inds = t * num_classes + p
    binc = np.bincount(inds, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes)


def _metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)

    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    total = cm.sum()

    fp = pred - tp
    fn = gt - tp
    union = tp + fp + fn

    iou = _safe_div(tp, union)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    pixel_acc = float(tp.sum() / total) if total > 0 else float("nan")

    present = gt > 0
    macro_iou = float(iou[present].mean()) if np.any(present) else float("nan")
    macro_f1 = float(f1[present].mean()) if np.any(present) else float("nan")
    macro_precision = (
        float(precision[present].mean()) if np.any(present) else float("nan")
    )
    macro_recall = float(recall[present].mean()) if np.any(present) else float("nan")

    return {
        "pixel_accuracy": pixel_acc,
        "per_class_iou": iou.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "macro_iou": macro_iou,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "support_gt": gt.astype(np.int64).tolist(),
        "support_pred": pred.astype(np.int64).tolist(),
    }


@dataclass
class _HeadStats:
    num_classes: int
    cm: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, tgt: np.ndarray, ignore_idx: int | None):
        self.cm += _confusion_from_arrays(
            pred=pred,
            tgt=tgt,
            num_classes=self.num_classes,
            ignore_idx=ignore_idx,
        )


@dataclass
class SaveTwoHeadEvalCallback(Callback):
    out_dir: str
    num_classes_a: int
    num_classes_b: int

    ignore_idx: int | None = None

    dpi: int = 200
    figsize: tuple[float, float] = (18.0, 5.0)

    save_figures: bool = True
    save_pred_masks: bool = False
    save_logits_pt: bool = False
    write_per_image_csv: bool = True
    write_summary_json: bool = True

    class_names_a: list[str] | None = None
    class_names_b: list[str] | None = None

    def _setup(self, stage: str, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.stage = stage
        self.root = Path(self.out_dir) / stage
        self.root.mkdir(parents=True, exist_ok=True)

        self.cmap_a = _make_discrete_cmap(self.num_classes_a)
        self.cmap_b = _make_discrete_cmap(self.num_classes_b)

        self.ignore = (
            self.ignore_idx
            if self.ignore_idx is not None
            else getattr(pl_module, "ignore_idx", None)
        )

        # global per-head stats
        self.global_stats = {
            "a": _HeadStats(self.num_classes_a),
            "b": _HeadStats(self.num_classes_b),
        }

        # per dataloader per head stats
        self.by_loader = defaultdict(dict)

        self._csv_file = None
        self._csv_writer = None
        if self.write_per_image_csv:
            self._csv_file = open(
                self.root / "per_image_metrics.csv", "w", newline="", encoding="utf-8"
            )
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "stage",
                    "dataloader_idx",
                    "img_id",
                    "gt_head",
                    "pixel_accuracy",
                    "macro_iou",
                    "macro_f1",
                    "per_class_iou",
                    "per_class_f1",
                ],
            )
            self._csv_writer.writeheader()

    def _teardown(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self._save_all_summaries()

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def on_validation_start(self, trainer, pl_module):
        self._setup("val", trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        self._setup("test", trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._teardown(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self._teardown(trainer, pl_module)

    def _get_loader_head_stats(self, dataloader_idx: int, head: str) -> _HeadStats:
        if head not in self.by_loader[dataloader_idx]:
            n = self.num_classes_a if head == "a" else self.num_classes_b
            self.by_loader[dataloader_idx][head] = _HeadStats(n)
        return self.by_loader[dataloader_idx][head]

    def _tensor_img_to_numpy(self, img_t: torch.Tensor) -> np.ndarray:
        img = img_t.detach().cpu()
        if img.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got {tuple(img.shape)}")
        img = img.permute(1, 2, 0).float().numpy()
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _save_figure(
        self,
        out_path: Path,
        img_id: str,
        img: np.ndarray | None,
        target: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        gt_head: str,
        metrics: dict,
    ) -> None:
        fig = plt.figure(figsize=self.figsize)
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        for ax in (ax1, ax2, ax3, ax4):
            ax.axis("off")

        ax1.set_title("Image")
        ax2.set_title(f"GT ({gt_head})")
        ax3.set_title("Pred A")
        ax4.set_title("Pred B")

        if img is not None:
            ax1.imshow(img)
        else:
            ax1.text(0.5, 0.5, "no image", ha="center", va="center")

        if gt_head == "a":
            ax2.imshow(
                target,
                cmap=self.cmap_a,
                vmin=-0.5,
                vmax=self.num_classes_a - 0.5,
                interpolation="nearest",
            )
        else:
            ax2.imshow(
                target,
                cmap=self.cmap_b,
                vmin=-0.5,
                vmax=self.num_classes_b - 0.5,
                interpolation="nearest",
            )

        ax3.imshow(
            pred_a,
            cmap=self.cmap_a,
            vmin=-0.5,
            vmax=self.num_classes_a - 0.5,
            interpolation="nearest",
        )
        ax4.imshow(
            pred_b,
            cmap=self.cmap_b,
            vmin=-0.5,
            vmax=self.num_classes_b - 0.5,
            interpolation="nearest",
        )

        lines = [
            f"img_id: {img_id}",
            f"GT head: {gt_head}",
            f"pixel_acc: {metrics['pixel_accuracy']:.4f}",
            f"macro_iou: {metrics['macro_iou']:.4f}",
            f"macro_f1: {metrics['macro_f1']:.4f}",
        ]

        fig.suptitle("\n".join(lines), fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _write_confusion_outputs(
        self,
        out_dir: Path,
        cm: np.ndarray,
        title: str,
        class_names: list[str] | None,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        k = cm.shape[0]

        metrics = _metrics_from_confusion(cm)

        row_sums = cm.sum(axis=1, keepdims=True)
        row_norm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

        labels = (
            class_names
            if class_names is not None and len(class_names) == k
            else [str(i) for i in range(k)]
        )

        # counts csv
        with open(out_dir / "confusion_matrix_counts.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["GT \\ Pred"] + labels)
            for i in range(k):
                writer.writerow([labels[i]] + list(cm[i]))

        # row-normalized csv
        with open(out_dir / "confusion_matrix_row_norm.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["GT \\ Pred"] + labels)
            for i in range(k):
                writer.writerow([labels[i]] + [f"{v:.6f}" for v in row_norm[i]])

        # per-class metrics csv
        with open(out_dir / "per_class_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "class_id",
                    "class_name",
                    "support_gt",
                    "support_pred",
                    "iou",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
            for i in range(k):
                writer.writerow(
                    [
                        i,
                        labels[i],
                        metrics["support_gt"][i],
                        metrics["support_pred"][i],
                        f"{metrics['per_class_iou'][i]:.6f}",
                        f"{metrics['per_class_precision'][i]:.6f}",
                        f"{metrics['per_class_recall'][i]:.6f}",
                        f"{metrics['per_class_f1'][i]:.6f}",
                    ]
                )

        # summary json
        if self.write_summary_json:
            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": title,
                        "num_classes": k,
                        "ignore_idx": self.ignore,
                        "macro_iou": metrics["macro_iou"],
                        "macro_f1": metrics["macro_f1"],
                        "macro_precision": metrics["macro_precision"],
                        "macro_recall": metrics["macro_recall"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "class_names": labels,
                    },
                    f,
                    indent=2,
                )

        # heatmap
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(row_norm, cmap="viridis")

        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        for i in range(k):
            for j in range(k):
                v = row_norm[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix_heatmap.png", dpi=self.dpi)
        plt.close(fig)

        return metrics

    def _save_all_summaries(self):
        # global
        self._write_confusion_outputs(
            out_dir=self.root / "head_a_global",
            cm=self.global_stats["a"].cm,
            title=f"{self.stage.upper()} - Head A confusion matrix",
            class_names=self.class_names_a,
        )
        self._write_confusion_outputs(
            out_dir=self.root / "head_b_global",
            cm=self.global_stats["b"].cm,
            title=f"{self.stage.upper()} - Head B confusion matrix",
            class_names=self.class_names_b,
        )

        # per dataloader
        for dl_idx, head_map in self.by_loader.items():
            for head, stats in head_map.items():
                self._write_confusion_outputs(
                    out_dir=self.root / f"dl{dl_idx}" / f"head_{head}",
                    cm=stats.cm,
                    title=f"{self.stage.upper()} - DL{dl_idx} - Head {head.upper()}",
                    class_names=self.class_names_a if head == "a" else self.class_names_b,
                )

    def _process_outputs(self, outputs, dataloader_idx: int):
        if outputs is None:
            return

        for o in outputs:
            img_id = str(o["img_id"])
            gt_head = str(o["gt_head"])

            if gt_head not in {"a", "b"}:
                continue

            pred_a_t = o["pred_a"]
            pred_b_t = o["pred_b"]
            tgt_t = o["target"]
            img_t = o.get("img", None)

            pred_a = pred_a_t.numpy().astype(np.int32)
            pred_b = pred_b_t.numpy().astype(np.int32)
            tgt = tgt_t.numpy().astype(np.int32)

            pred_supervised = pred_a if gt_head == "a" else pred_b
            num_classes = self.num_classes_a if gt_head == "a" else self.num_classes_b

            cm_img = _confusion_from_arrays(
                pred=pred_supervised,
                tgt=tgt,
                num_classes=num_classes,
                ignore_idx=self.ignore,
            )
            metrics_img = _metrics_from_confusion(cm_img)

            self.global_stats[gt_head].update(pred_supervised, tgt, self.ignore)
            self._get_loader_head_stats(dataloader_idx, gt_head).update(
                pred_supervised, tgt, self.ignore
            )

            out_dir = self.root / f"dl{dataloader_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.save_figures:
                img_np = None
                if isinstance(img_t, torch.Tensor):
                    img_np = self._tensor_img_to_numpy(img_t)

                self._save_figure(
                    out_path=out_dir / f"{img_id}.png",
                    img_id=img_id,
                    img=img_np,
                    target=tgt,
                    pred_a=pred_a,
                    pred_b=pred_b,
                    gt_head=gt_head,
                    metrics=metrics_img,
                )

            if self.save_pred_masks:
                torch.save(pred_a_t, out_dir / f"{img_id}_pred_a.pt")
                torch.save(pred_b_t, out_dir / f"{img_id}_pred_b.pt")
                torch.save(tgt_t, out_dir / f"{img_id}_target.pt")

            if self.save_logits_pt:
                torch.save(o["logits_a"], out_dir / f"{img_id}_logits_a.pt")
                torch.save(o["logits_b"], out_dir / f"{img_id}_logits_b.pt")

            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    {
                        "stage": self.stage,
                        "dataloader_idx": dataloader_idx,
                        "img_id": img_id,
                        "gt_head": gt_head,
                        "pixel_accuracy": metrics_img["pixel_accuracy"],
                        "macro_iou": metrics_img["macro_iou"],
                        "macro_f1": metrics_img["macro_f1"],
                        "per_class_iou": json.dumps(
                            [round(x, 8) for x in metrics_img["per_class_iou"]]
                        ),
                        "per_class_f1": json.dumps(
                            [round(x, 8) for x in metrics_img["per_class_f1"]]
                        ),
                    }
                )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.is_global_zero:
            return
        self._process_outputs(outputs, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.is_global_zero:
            return
        self._process_outputs(outputs, dataloader_idx)