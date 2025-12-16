import time

import torch
from tqdm import tqdm

from datasets.ade20k import ADE20K


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
    max_batches = 1000
    pl_dm = ADE20K("/home/valentin/workspaces/benchmark-vfm-ss/data", devices=1, num_workers=8)
    pl_dm.setup()

    loader = pl_dm.train_dataloader()
    n, samples = 0, 0
    t0 = time.perf_counter()
    for n, batch in tqdm(enumerate(loader, start=1), total=max_batches):
        if device:
            _ = move_to_device(batch, device)
        samples += batch_size_of(batch)
        if n >= max_batches:
            break
    t1 = time.perf_counter()
    print(f"Processed {samples} samples in {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
