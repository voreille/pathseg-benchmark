from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from lightning.pytorch.callbacks import Callback


def _make_discrete_cmap(num_classes: int) -> ListedColormap:
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.arange(num_classes))
    return ListedColormap(colors)


def _area_stats_from_pred(
    pred: np.ndarray,
    num_classes: int,
) -> tuple[list[int], list[float]]:
    total = int(pred.size)
    counts = np.bincount(pred.reshape(-1), minlength=num_classes)[:num_classes]
    counts = counts.astype(np.int64)
    fracs = (counts / max(total, 1)).astype(np.float64)
    return counts.tolist(), fracs.tolist()


def _extract_scalar_label(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        if x.ndim == 0:
            return int(x.item())
        return None

    if isinstance(x, (int, float, str)):
        return int(x)

    # segmentation dict or anything else
    return None


@dataclass
class SavePrototypeROIPredictionsCallback(Callback):
    out_dir: str
    num_classes_b: int
    num_classes_a: int

    dpi: int = 200
    figsize: tuple[float, float] = (16.0, 5.0)

    save_figures: bool = True
    save_pred_masks: bool = True
    save_logits_pt: bool = False
    save_compartment_logits_pt: bool = False
    save_numpy_masks: bool = False
    write_csv: bool = True

    def on_predict_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.root = Path(self.out_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.cmap_b = _make_discrete_cmap(self.num_classes_b)
        self.cmap_a = _make_discrete_cmap(self.num_classes_a)

        self.csv_path = self.root / "roi_predictions.csv"
        self._csv_file = None
        self._csv_writer = None

        if self.write_csv:
            fieldnames = [
                "dataloader_idx",
                "img_id",
                "gt_label",
                "pred_label_max_area",
                "height",
                "width",
                "label_ids_b",
                "area_px_b",
                "area_frac_b",
                "area_px_a",
                "area_frac_a",
            ]
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()

    def on_predict_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _tensor_img_to_numpy(self, img_t: torch.Tensor) -> np.ndarray:
        img = img_t.detach().cpu()
        if img.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got {tuple(img.shape)}")

        img = img.permute(1, 2, 0).float().numpy()

        # robust display scaling
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def _save_figure(
        self,
        out_path: Path,
        img_id: str,
        img: np.ndarray | None,
        pred_b: np.ndarray,
        pred_a: np.ndarray,
        gt_label: int | None,
        pred_label: int,
        area_frac_b: list[float],
        label_ids_b: list[int] | None,
    ) -> None:
        fig = plt.figure(figsize=self.figsize)

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # ---- Titles ----
        ax1.set_title("Image")
        ax2.set_title("Head B (patterns)")
        ax3.set_title("Head A (compartments)")

        for ax in (ax1, ax2, ax3):
            ax.axis("off")

        # ---- Image ----
        if img is not None:
            ax1.imshow(img)
        else:
            ax1.text(0.5, 0.5, "no image", ha="center", va="center")

        # ---- Head B ----
        im_b = ax2.imshow(
            pred_b,
            cmap=self.cmap_b,
            vmin=-0.5,
            vmax=self.num_classes_b - 0.5,
            interpolation="nearest",
        )

        # ---- Head A ----
        im_a = ax3.imshow(
            pred_a,
            cmap=self.cmap_a,
            vmin=-0.5,
            vmax=self.num_classes_a - 0.5,
            interpolation="nearest",
        )

        # ---- Colorbars ----
        cbar_b = fig.colorbar(im_b, ax=ax2, fraction=0.046, pad=0.04)
        cbar_b.set_ticks(range(self.num_classes_b))
        cbar_b.ax.set_title("B", fontsize=8)

        cbar_a = fig.colorbar(im_a, ax=ax3, fraction=0.046, pad=0.04)
        cbar_a.set_ticks(range(self.num_classes_a))
        cbar_a.ax.set_title("A", fontsize=8)

        # ---- Clean text formatting ----
        header = [
            f"{img_id}",
            f"GT: {gt_label}" if gt_label is not None else "GT: None",
            f"Pred (max area): {pred_label}",
        ]

        if label_ids_b is not None:
            header.append(f"Active labels B: {label_ids_b}")

        # compact area display (only non-zero)
        area_lines = []
        for i, v in enumerate(area_frac_b):
            if v > 0.01:  # threshold for readability
                area_lines.append(f"{i}: {v:.2f}")

        area_text = "Area frac (B):\n" + ", ".join(area_lines)

        # ---- Title + side box ----
        fig.suptitle("\n".join(header), fontsize=12)

        ax3.text(
            1.05,
            0.5,
            area_text,
            transform=ax3.transAxes,
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.85),
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def on_predict_batch_end(
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

        split_dir = self.root / f"dl{dataloader_idx}"
        split_dir.mkdir(parents=True, exist_ok=True)

        for o in outputs:
            img_id = str(o["img_id"])

            pred_b_t = o["pred"]
            pred_a_t = o["pred_a"]
            logits_b_t = o["logits"]
            logits_a_t = o["logits_a"]
            img_t = o.get("img", None)

            pred_b = pred_b_t.numpy().astype(np.int32)
            pred_a = pred_a_t.numpy().astype(np.int32)

            img_np = None
            if isinstance(img_t, torch.Tensor):
                img_np = self._tensor_img_to_numpy(img_t)

            h, w = int(pred_b.shape[-2]), int(pred_b.shape[-1])

            area_px_b, area_frac_b = _area_stats_from_pred(pred_b, self.num_classes_b)
            area_px_a, area_frac_a = _area_stats_from_pred(pred_a, self.num_classes_a)

            pred_label = int(np.argmax(np.asarray(area_frac_b)))

            gt_label = _extract_scalar_label(o.get("gt_label", None))
            label_ids_b = o.get("label_ids_b", None)
            if torch.is_tensor(label_ids_b):
                label_ids_b = [int(x) for x in label_ids_b.tolist()]

            if self.save_pred_masks:
                torch.save(pred_b_t, split_dir / f"{img_id}_pred_b.pt")
                torch.save(pred_a_t, split_dir / f"{img_id}_pred_a.pt")

            if self.save_numpy_masks:
                np.save(split_dir / f"{img_id}_pred_b.npy", pred_b)
                np.save(split_dir / f"{img_id}_pred_a.npy", pred_a)

            if self.save_logits_pt:
                torch.save(logits_b_t, split_dir / f"{img_id}_logits_b.pt")

            if self.save_compartment_logits_pt:
                torch.save(logits_a_t, split_dir / f"{img_id}_logits_a.pt")

            summary = {
                "img_id": img_id,
                "gt_label": gt_label,
                "pred_label_max_area": pred_label,
                "height": h,
                "width": w,
                "label_ids_b": label_ids_b,
                "area_px_b": area_px_b,
                "area_frac_b": area_frac_b,
                "area_px_a": area_px_a,
                "area_frac_a": area_frac_a,
            }
            with (split_dir / f"{img_id}_summary.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, indent=2)

            if self.save_figures:
                self._save_figure(
                    out_path=split_dir / f"{img_id}.png",
                    img_id=img_id,
                    img=img_np,
                    pred_b=pred_b,
                    pred_a=pred_a,
                    gt_label=gt_label,
                    pred_label=pred_label,
                    area_frac_b=area_frac_b,
                    label_ids_b=label_ids_b,
                )

            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    {
                        "dataloader_idx": dataloader_idx,
                        "img_id": img_id,
                        "gt_label": gt_label,
                        "pred_label_max_area": pred_label,
                        "height": h,
                        "width": w,
                        "label_ids_b": json.dumps(label_ids_b),
                        "area_px_b": json.dumps(area_px_b),
                        "area_frac_b": json.dumps([round(x, 8) for x in area_frac_b]),
                        "area_px_a": json.dumps(area_px_a),
                        "area_frac_a": json.dumps([round(x, 8) for x in area_frac_a]),
                    }
                )
