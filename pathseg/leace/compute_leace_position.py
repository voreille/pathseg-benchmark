#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.anorak import ANORAKFewShot
from training.tiler import GridPadTiler
from pathseg.models.encoder import Encoder
from leace.leace import LeaceFitter


@torch.no_grad()
def accumulate_features_and_positions(
    encoder: Encoder,
    dataloader: DataLoader,
    img_tiler: GridPadTiler,
    grid_size: Tuple[int, int],
    device: torch.device,
):
    """Accumulate encoder token features, labels, and LEACE stats for positions."""
    Ht, Wt = grid_size

    encoder.eval()
    encoder.to(device)

    global_img_offset = 0
    tile_size = img_tiler.tile

    # positions in token grid, used as concept z
    gy, gx = torch.meshgrid(
        torch.arange(Ht, device=device),
        torch.arange(Wt, device=device),
        indexing="ij",
    )
    positions = torch.stack([gy, gx], dim=-1).reshape(-1, 2).float()  # [Q,2]
    pos_leace_fitter = LeaceFitter(
        x_dim=encoder.embed_dim,
        z_dim=2,
        device=device,
    )

    for imgs, _ in tqdm(dataloader, desc="collect tokens", leave=False):
        # your Dataset returns list-like; ANORAKFewShot uses batch_size>=1
        # but we tiled with GridPadTiler which expects list of images
        B_batch = len(imgs)
        assert B_batch == 1, "This script assumes batch_size=1 at dataloader level"

        img_shape = imgs[0].shape  # [C,H,W]
        if min(img_shape[1:]) < tile_size:
            continue  # skip too small images

        batch_img_ids = torch.arange(
            global_img_offset, global_img_offset + B_batch, dtype=torch.long
        )
        global_img_offset += B_batch

        # --- tile images ---
        img_crops, origins, img_sizes = img_tiler.window(imgs)
        img_crops = img_crops.to(device) / 255.0  # encoder expects [0,1]
        Nc = img_crops.shape[0]

        crop_img_ids = torch.empty(Nc, dtype=torch.long)
        for i, ori in enumerate(origins):
            b_idx = ori.img_idx
            crop_img_ids[i] = batch_img_ids[b_idx]

        tokens = encoder(img_crops)  # [Nc, Q, D]
        Nc, Q, D = tokens.shape
        assert Q == Ht * Wt, f"Q={Q} != Ht*Wt={Ht * Wt}"

        X_flat = tokens.reshape(Nc * Q, D)  # [Nc*Q, D]
        pos_expanded = positions.unsqueeze(0).expand(Nc, Q, 2).reshape(Nc * Q, 2)

        # update LEACE statistics with all tokens (no masking)
        pos_leace_fitter.update(X_flat, pos_expanded)

    pos_eraser = pos_leace_fitter.eraser
    return pos_eraser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, required=True, help="Root ANORAK_10x dir (with image/mask)"
    )
    parser.add_argument("--fewshot_csv", type=Path, required=True)
    parser.add_argument("--support_set", type=int, default=0)
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--encoder_id", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--tile", type=int, default=448)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--ignore_idx", type=int, default=255)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Must be 1 for this script (tiler expects list)",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the LEACE eraser (e.g. leace_pos.pt)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- DataModule (few-shot, but we use its train set as LEACE data) ---
    dm = ANORAKFewShot(
        root=args.root,
        devices=1,
        fewshot_csv=args.fewshot_csv,
        support_set=args.support_set,
        n_shot=args.n_shot,
        img_size=(args.tile, args.tile),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        num_iter_per_epoch=-1,
        drop_last=False,
    )
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    # --- Encoder ---
    encoder = Encoder(
        encoder_id=args.encoder_id,
        img_size=(args.tile, args.tile),
        sub_norm=False,
        ckpt_path=str(args.ckpt_path),
        discard_last_mlp=False,
    )

    # --- Tilers (image + target) ---
    grid_size = encoder.grid_size  # typically (Ht, Wt)
    img_tiler = GridPadTiler(
        tile=args.tile,
        stride=args.stride,
        weighted_blend=False,
        pad_mode="replicate",
        pad_value=0.0,
    )

    # --- Compute LEACE eraser for positional concept ---
    pos_eraser = accumulate_features_and_positions(
        encoder=encoder,
        dataloader=train_loader,
        img_tiler=img_tiler,
        grid_size=grid_size,
        device=device,
    )

    if pos_eraser is None:
        print("[LEACE] No eraser produced; exiting.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pos_eraser.save(args.output)
    print(f"[LEACE] Saved positional eraser to: {args.output}")


if __name__ == "__main__":
    main()
