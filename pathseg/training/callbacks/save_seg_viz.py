from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch


def _make_discrete_cmap(num_classes: int) -> ListedColormap:
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.arange(num_classes))
    return ListedColormap(colors)


def _per_class_iou(
    pred: np.ndarray,
    tgt: np.ndarray,
    num_classes: int,
    ignore_idx: int | None,
    present_only: bool = True,
    exclude_labels: set[int] | None = None,
) -> dict[int, float]:
    """
    Returns {class_id: iou}.
    present_only=True -> compute only for classes present in GT (excluding ignore/excluded).
    """
    exclude_labels = exclude_labels or set()

    if ignore_idx is not None:
        valid = tgt != ignore_idx
        pred = pred[valid]
        tgt = tgt[valid]

    if present_only:
        classes = np.unique(tgt.reshape(-1))
    else:
        classes = np.arange(num_classes)

    out: dict[int, float] = {}
    for c in classes:
        c = int(c)
        if c < 0 or c >= num_classes:
            continue
        if (ignore_idx is not None and c == ignore_idx) or (c in exclude_labels):
            continue

        inter = np.logical_and(pred == c, tgt == c).sum()
        union = np.logical_or(pred == c, tgt == c).sum()
        if union > 0:
            out[c] = float(inter / union)
    return out


@dataclass
class SaveSegVizCallback(Callback):
    out_dir: str
    num_classes: int

    dpi: int = 200
    figsize: tuple[float, float] = (16.0, 4.0)  # <--- NEW
    write_csv: bool = True
    save_pred_mask: bool = False

    ignore_idx: int | None = None  # if None, will read from pl_module.ignore_idx

    show_iou_per_class: bool = True
    iou_present_only: bool = True

    exclude_bg_from_miou: bool = True  # <--- NEW
    bg_label: int = 0  # <--- NEW

    def _update_cm(
        self,
        pred: np.ndarray,
        tgt: np.ndarray,
        ignore_idx: int | None,
        exclude_labels: set[int],
    ):
        if ignore_idx is not None:
            valid = tgt != ignore_idx
        else:
            valid = np.ones_like(tgt, dtype=bool)

        if exclude_labels:
            for lab in exclude_labels:
                valid &= tgt != lab

        p = pred[valid].astype(np.int64)
        t = tgt[valid].astype(np.int64)

        # fast bincount trick
        k = self.num_classes
        inds = t * k + p
        binc = np.bincount(inds, minlength=k * k)
        self.cm += binc.reshape(k, k)

    def on_predict_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.root = Path(self.out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cmap = _make_discrete_cmap(self.num_classes)
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        self.csv_path = self.root / "per_image_metrics.csv"
        self._csv_file = None
        self._csv_writer = None

        if self.write_csv:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=["split", "img_id", "miou_no_bg", "iou_per_class"],
            )
            self._csv_writer.writeheader()

    def save_cm(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        # --- counts ---
        counts = self.cm
        k = self.num_classes

        # Row-normalized (GT -> prediction distribution)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_norm = np.divide(
            counts,
            row_sums,
            out=np.zeros_like(counts, dtype=float),
            where=row_sums != 0,
        )

        # ---------- Save CSV (counts) ----------
        counts_csv = self.root / "confusion_matrix_counts.csv"
        with open(counts_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["GT \\ Pred"] + [str(i) for i in range(k)]
            writer.writerow(header)
            for i in range(k):
                writer.writerow([str(i)] + list(counts[i]))

        # ---------- Save CSV (row-normalized) ----------
        row_norm_csv = self.root / "confusion_matrix_row_norm.csv"
        with open(row_norm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["GT \\ Pred"] + [str(i) for i in range(k)]
            writer.writerow(header)
            for i in range(k):
                writer.writerow([str(i)] + [f"{v:.6f}" for v in row_norm[i]])

        # ---------- Save heatmap figure ----------
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(row_norm, cmap="viridis")

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_title("Confusion Matrix (row-normalized)")

        # annotate values
        for i in range(k):
            for j in range(k):
                val = row_norm[i, j]
                if val > 0.01:  # only annotate meaningful cells
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.root / "confusion_matrix_heatmap.png", dpi=200)
        plt.close(fig)

        # ---------- Top confusions (off-diagonal) ----------
        rows = []
        for i in range(k):
            if row_sums[i, 0] == 0:
                continue

            tmp = row_norm[i].copy()
            tmp[i] = -1  # ignore correct predictions
            j = int(tmp.argmax())
            rows.append((i, j, row_norm[i, j]))

        top_csv = self.root / "top_confusions.csv"
        with open(top_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_class", "pred_class", "rate"])
            for i, j, rate in rows:
                writer.writerow([i, j, f"{rate:.6f}"])

    def on_predict_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.save_cm(trainer, pl_module)
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not trainer.is_global_zero:
            return

        ignore = (
            self.ignore_idx
            if self.ignore_idx is not None
            else getattr(pl_module, "ignore_idx", None)
        )
        exclude = {self.bg_label} if self.exclude_bg_from_miou else set()

        for o in outputs:
            split = str(o.get("split", f"dl{dataloader_idx}"))
            img_id = str(o["img_id"])

            pred_t = o["pred"]
            tgt_t = o.get("tgt", None)
            img_t = o.get("img", None)

            pred = pred_t.numpy().astype(np.int32)
            H, W = int(pred.shape[-2]), int(pred.shape[-1])

            miou_no_bg = np.nan
            per_cls = {}
            if tgt_t is not None:
                tgt = tgt_t.numpy().astype(np.int32)
                per_cls = _per_class_iou(
                    pred=pred,
                    tgt=tgt,
                    num_classes=self.num_classes,
                    ignore_idx=ignore,
                    present_only=self.iou_present_only,
                    exclude_labels=exclude,  # excludes bg from per-class IoUs used in mIoU
                )
                miou_no_bg = (
                    float(np.mean(list(per_cls.values()))) if len(per_cls) else np.nan
                )
                self._update_cm(pred, tgt, ignore_idx=ignore, exclude_labels=set())

            # ---------- figure ----------
            fig = plt.figure(figsize=self.figsize)
            ax1 = fig.add_subplot(1, 4, 1)
            ax2 = fig.add_subplot(1, 4, 2)
            ax3 = fig.add_subplot(1, 4, 3)
            ax4 = fig.add_subplot(1, 4, 4)

            ax1.set_title("Image")
            ax2.set_title("GT")
            ax3.set_title("Pred")
            ax4.set_title("Error")
            for ax in (ax1, ax2, ax3, ax4):
                ax.axis("off")

            # Image
            if (
                img_t is not None
                and isinstance(img_t, torch.Tensor)
                and img_t.ndim == 3
            ):
                ax1.imshow(img_t.permute(1, 2, 0))
            else:
                ax1.text(0.5, 0.5, "no image", ha="center", va="center")

            # GT / Pred with consistent colormap
            if tgt_t is not None:
                tgt = tgt_t.numpy().astype(np.int32)

                ax2.imshow(
                    tgt,
                    cmap=self.cmap,
                    vmin=-0.5,
                    vmax=self.num_classes - 0.5,
                    interpolation="nearest",
                )
                ax3.imshow(
                    pred,
                    cmap=self.cmap,
                    vmin=-0.5,
                    vmax=self.num_classes - 0.5,
                    interpolation="nearest",
                )

                err = pred != tgt
                if ignore is not None:
                    err = err & (tgt != ignore)
                ax4.imshow(err.astype(np.uint8), interpolation="nearest")

                # Metrics text box
                lines = []
                if not np.isnan(miou_no_bg):
                    lines.append(f"mIoU (no bg={self.bg_label}): {miou_no_bg:.3f}")

                if self.show_iou_per_class and len(per_cls):
                    for c in sorted(per_cls.keys()):
                        lines.append(f"IoU[{c}]: {per_cls[c]:.3f}")

                if lines:
                    ax4.text(
                        1.02,
                        0.98,
                        "\n".join(lines),
                        transform=ax4.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", alpha=0.85),
                    )
            else:
                ax2.text(0.5, 0.5, "no GT", ha="center", va="center")
                ax3.imshow(
                    pred,
                    cmap=self.cmap,
                    vmin=-0.5,
                    vmax=self.num_classes - 0.5,
                    interpolation="nearest",
                )
                ax4.text(0.5, 0.5, "no GT", ha="center", va="center")

            # Save
            out_dir = self.root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{img_id}.png"

            title = f"{img_id}"
            if not np.isnan(miou_no_bg):
                title += f"  mIoU(no-bg)={miou_no_bg:.3f}"
            title += f" (img_size={W}x{H})"
            fig.suptitle(title)

            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            # Optional: save pred mask (uint8) — safe only if num_classes <= 256
            if self.save_pred_mask:
                pred_path = out_dir / f"{img_id}_pred.png"
                plt.imsave(pred_path, pred.astype(np.uint8))

            # CSV row
            if self._csv_writer is not None:
                iou_str = ";".join([f"{c}:{v:.4f}" for c, v in sorted(per_cls.items())])
                self._csv_writer.writerow(
                    {
                        "split": split,
                        "img_id": img_id,
                        "miou_no_bg": miou_no_bg,
                        "iou_per_class": iou_str,
                    }
                )
