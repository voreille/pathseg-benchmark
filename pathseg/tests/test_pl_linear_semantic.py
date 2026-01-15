import time

import torch
from tqdm import tqdm

from datasets.anorak import ANORAK
from pathseg.models.linear_decoder import LinearDecoder
from training.linear_semantic import LinearSemantic


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def batch_size_of(batch):
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    return int(getattr(x, "shape", [1])[0]) if hasattr(x, "shape") else 1


def main():
    device = torch.device("cuda:0")
    max_batches = 2
    pl_dm = ANORAK(
        "/home/valentin/workspaces/benchmark-vfm-ss/data/ANORAK",
        devices=1,
        num_workers=0,
        batch_size=1,
        img_size=(448, 448),
    )
    pl_dm.setup()

    loader = pl_dm.val_dataloader()
    n, samples = 0, 0
    encoder = LinearDecoder("hf-hub:MahmoodLab/UNI2-h", 7, (448, 448))
    encoder.eval()
    encoder.to(device)
    pl_model = LinearSemantic(encoder,
                              num_metrics=1,
                              ignore_idx=255,
                              num_classes=7,
                              img_size=(448, 448),
                              lr=1e-3,
                              weight_decay=1e-4,
                              lr_multiplier_encoder=0.1,
                              freeze_encoder=True)

    t0 = time.perf_counter()
    for n, batch in tqdm(enumerate(loader, start=1), total=max_batches):
        if device:
            batch = move_to_device(batch, device)
        samples += batch_size_of(batch)
        imgs, target = batch
        with torch.no_grad():
            crops, origins, img_sizes = pl_model.window_imgs_semantic(imgs)

        if n >= max_batches:
            break
    t1 = time.perf_counter()
    print(f"Processed {samples} samples in {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
