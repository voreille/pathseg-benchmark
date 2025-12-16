from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np


class SavePredictions(BasePredictionWriter):

    def __init__(self, out_dir: str, write_interval="batch"):
        super().__init__(write_interval=write_interval)
        self.out_dir = Path(out_dir)
        self._m = {"val": [], "test": []}

    def setup(self, trainer, pl_module, stage):
        (self.out_dir / "val").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "test").mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
        **kwargs,
    ):
        for gi, item in zip(batch_indices, prediction):
            split = item["split"]  # comes from predict_step
            item["pil"].save(self.out_dir / split / f"{item['img_id']}.png")
            self._m.setdefault(split, []).append(item["miou"])

    def on_predict_end(self, trainer, pl_module):
        for split, vals in self._m.items():
            if not vals:
                continue
            mean_iou = float(np.nanmean(vals))
            (self.out_dir / split / f"{split}_summary.txt"
             ).write_text(f"Mean IoU: {mean_iou:.6f}\nCount: {len(vals)}\n")
