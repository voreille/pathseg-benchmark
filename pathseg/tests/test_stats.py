import time
from pathlib import Path

from datasets.stats import compute_class_weights_from_ids
from datasets.anorak import ANORAK


def main():
    masks_dir = Path(
        "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK/mask")
    roi_ids = [p.stem for p in Path(masks_dir).glob("*.png")]
    num_classes = 7
    ignore_idx = 255
    t0 = time.perf_counter()
    class_weights = compute_class_weights_from_ids(roi_ids, masks_dir,
                                                   num_classes, ignore_idx)
    t1 = time.perf_counter()
    print(f"Computed class weights in {t1 - t0:.2f} seconds:")
    print(class_weights)
    # Set up data module to load validation data
    for fold in range(5):
        print(f"Fold {fold}:")
        data_module = ANORAK(
            root="./data/ANORAK",  # Adjust path to your data
            devices=1,
            num_workers=4,
            batch_size=1,
            img_size=(448, 448),
            num_classes=7,
            fold=fold,
            num_metrics=1)

        # Setup the data module
        data_module.setup()
        class_weights_dm = data_module.compute_class_weights()
        print("Class weights from data module:")
        print(class_weights_dm)

if __name__ == "__main__":
    main()
