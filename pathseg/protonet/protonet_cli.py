#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from datasets.anorak import ANORAKFewShot
from models.histo_protonet_decoder import ProtoNetDecoder
from training.tiler import GridPadTiler, Tiler

# ---------------------------
# helpers
# ---------------------------


def parse_devices(devs: List[int]) -> torch.device:
    if torch.cuda.is_available() and len(devs) > 0:
        torch.cuda.set_device(devs[0])
        return torch.device(f"cuda:{devs[0]}")
    return torch.device("cpu")


@torch.compiler.disable
def to_per_pixel_targets_semantic(
    targets: list[dict],
    ignore_idx: int,
) -> list[torch.Tensor]:
    """Convert list of instance masks into a single-channel semantic map per image."""
    out: list[torch.Tensor] = []
    for t in targets:
        h, w = t["masks"].shape[-2:]
        y = torch.full((h, w),
                       ignore_idx,
                       dtype=t["labels"].dtype,
                       device=t["labels"].device)
        for i, m in enumerate(t["masks"]):
            y[m] = t["labels"][i]
        out.append(y)  # [H,W] long
    return out


@torch.no_grad()
def compute_prototypes(
    decoder: ProtoNetDecoder,
    dataloader,
    img_tiler: Tiler,
    target_tiler: Tiler,
):
    decoder.eval()
    # 1) mean
    for imgs, targets in tqdm(dataloader, desc="mean", leave=False):
        # images -> [N,3,T,T], targets -> [N,1,T,T]
        img_crops, _, _ = img_tiler.window(imgs)  # [N,3,T,T]
        img_crops = img_crops / 255.0
        sem = to_per_pixel_targets_semantic(
            targets, decoder.ignore_idx)  # list of [H,W]
        sem = [y.unsqueeze(0) for y in sem]  # [1,H,W]
        tgt_crops, _, _ = target_tiler.window(sem)  # [N,1,T,T]
        decoder.update_mean_from_batch(img_crops, tgt_crops)

    # 2) prototypes
    for imgs, targets in tqdm(dataloader, desc="prototypes", leave=False):
        img_crops, _, _ = img_tiler.window(imgs)
        img_crops = img_crops / 255.0
        sem = to_per_pixel_targets_semantic(targets, decoder.ignore_idx)
        sem = [y.unsqueeze(0) for y in sem]
        tgt_crops, _, _ = target_tiler.window(sem)
        decoder.update_prototypes_from_batch(img_crops, tgt_crops)

    return decoder.head.prototypes.clone(), decoder.head.mean.clone()


@torch.no_grad()
def predict_fullres(
    decoder: ProtoNetDecoder,
    imgs: list[torch.Tensor],  # list of [3,H,W]
    img_tiler: Tiler,
) -> list[torch.Tensor]:
    """Tile -> run -> stitch. Returns per-image logits [C,H,W] (same size as input)."""
    crops, origins, img_sizes = img_tiler.window(imgs)  # [N,3,T,T]
    crops = crops / 255.0
    crop_logits = decoder(crops)  # [N,C,T,T]
    logits_full = img_tiler.stitch(crop_logits, origins,
                                   img_sizes)  # list of [C,H,W]
    return logits_full


# ---------------------------
# subcommands
# ---------------------------


def cmd_compute_prototypes(args: argparse.Namespace) -> None:
    device = parse_devices(args.device)

    # datamodule
    dm = ANORAKFewShot(
        args.root_dir,
        args.device,
        num_workers=args.num_workers,
        fold=args.fold,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_metrics=1,
        ignore_idx=args.ignore_idx,
        prefetch_factor=args.prefetch_factor,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # model/decoder
    decoder = ProtoNetDecoder(
        encoder_name=args.encoder_name,
        num_classes=args.num_classes,
        img_size=tuple(args.img_size),  # ViT grid target (not forcing resize)
        sub_norm=False,
        patch_size=args.patch_size,
        pretrained=not args.no_pretrained,
        ckpt_path=args.ckpt_path,
        metric=args.metric,
        center_feats=not args.no_center_feats,
        normalize_feats=not args.no_normalize_feats,
        ignore_idx=args.ignore_idx,
    ).to(device).eval()

    # tilers (image: replicate pad; targets: constant pad with ignore_idx)
    img_tiler = GridPadTiler(
        tile=args.tile,
        stride=args.stride,
        weighted_blend=args.weighted_blend,
        pad_mode="replicate",
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=args.tile,
        stride=args.stride,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=float(args.ignore_idx),
    )

    # compute
    prototypes, mean = compute_prototypes(decoder, train_loader, img_tiler,
                                          tgt_tiler)

    # save
    out = {
        "prototypes": prototypes.cpu(),
        "mean": mean.cpu(),
        "meta": {
            "encoder_name": args.encoder_name,
            "ckpt_path": args.ckpt_path,
            "img_size": tuple(args.img_size),
            "grid_size": tuple(decoder.grid_size),
            "patch_size": args.patch_size,
            "num_classes": args.num_classes,
            "ignore_idx": args.ignore_idx,
            "metric": args.metric,
            "center_feats": not args.no_center_feats,
            "normalize_feats": not args.no_normalize_feats,
            "root_dir": os.path.abspath(args.root_dir),
            "fold": args.fold,
            "tile": args.tile,
            "stride": args.stride,
            "weighted_blend": args.weighted_blend,
        },
    }
    torch.save(out, args.save_path)
    print(
        f"[OK] saved to {args.save_path}  "
        f"prototypes={list(out['prototypes'].shape)}, mean={list(out['mean'].shape)}"
    )


def cmd_predict(args: argparse.Namespace) -> None:
    device = parse_devices(args.device)

    payload = torch.load(args.load_path, map_location="cpu")
    meta = payload["meta"]
    num_classes = int(meta["num_classes"])

    # build decoder & load prototypes
    decoder = ProtoNetDecoder(
        encoder_name=args.encoder_name,
        num_classes=num_classes,
        img_size=tuple(meta["img_size"]),
        sub_norm=False,
        patch_size=args.patch_size,
        pretrained=not args.no_pretrained,
        ckpt_path=args.ckpt_path,
        metric=meta["metric"],
        center_feats=meta["center_feats"],
        normalize_feats=meta["normalize_feats"],
        ignore_idx=meta["ignore_idx"],
    ).to(device).eval()
    with torch.no_grad():
        decoder.head.prototypes.copy_(payload["prototypes"].to(device))
        decoder.head.mean.copy_(payload["mean"].to(device))

    # tiler mirrors training tiler
    img_tiler = GridPadTiler(
        tile=int(meta["tile"]),
        stride=int(meta["stride"]),
        weighted_blend=bool(meta["weighted_blend"]),
        pad_mode="replicate",
        pad_value=0.0,
    )

    # load images as list of tensors [3,H,W]
    arr = np.load(args.imgs_npy, allow_pickle=False)  # (B,H,W,3)
    imgs = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    imgs = [im.float().to(device) for im in imgs]

    logits_full = predict_fullres(decoder, imgs, img_tiler)  # list of [C,H,W]
    labels = [lf.argmax(dim=0).cpu().numpy() for lf in logits_full]
    print(f"pred shapes: {[x.shape for x in labels]}")


# ---------------------------
# CLI
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="protonet_cli",
        description="ProtoNet prototype computation & predict (no PL).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # compute-prototypes
    pc = sub.add_parser("compute-prototypes",
                        help="Compute dataset-wide prototypes and mean.")
    pc.add_argument("--root-dir",
                    type=str,
                    required=True,
                    help="ANORAK dataset root")
    pc.add_argument("--fold", type=int, default=0)
    pc.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[448, 448],
        metavar=("H", "W"),
        help="ViT working size used inside the model (grid mapping).")
    pc.add_argument("--batch-size", type=int, default=1)
    pc.add_argument("--num-classes", type=int, default=7)
    pc.add_argument("--ignore-idx", type=int, default=255)
    pc.add_argument("--num-workers", type=int, default=0)
    pc.add_argument("--prefetch-factor", type=int, default=2)
    pc.add_argument("--persistent-workers", action="store_true")

    # tiler config (important for histopathology magnification preservation)
    pc.add_argument("--tile",
                    type=int,
                    default=448,
                    help="Crop size (pixels) at source resolution.")
    pc.add_argument("--stride",
                    type=int,
                    default=448,
                    help="Stride between crops.")
    pc.add_argument("--weighted-blend",
                    action="store_true",
                    help="Use Hann weighting during stitching.")

    # model/encoder
    pc.add_argument("--encoder-name",
                    type=str,
                    default="hf-hub:MahmoodLab/UNI2-h")
    pc.add_argument("--patch-size", type=int, default=14)
    pc.add_argument("--no-pretrained", action="store_true")
    pc.add_argument("--ckpt-path", type=str, default="")
    pc.add_argument("--metric",
                    type=str,
                    default="L2",
                    choices=["L2", "cosine"])
    pc.add_argument("--no-center-feats", action="store_true")
    pc.add_argument("--no-normalize-feats", action="store_true")

    pc.add_argument("--device",
                    type=int,
                    nargs="+",
                    default=[0],
                    help="GPU ids (first is used)")
    pc.add_argument("--save-path", type=str, default="prototypes.pt")

    # predict
    pp = sub.add_parser("predict",
                        help="Predict labels at full resolution with tiling.")
    pp.add_argument("--imgs-npy",
                    type=str,
                    required=True,
                    help="Path to numpy array (B,H,W,3).")
    pp.add_argument("--load-path",
                    type=str,
                    required=True,
                    help="Path to prototypes file (.pt).")
    pp.add_argument("--encoder-name",
                    type=str,
                    default="hf-hub:MahmoodLab/UNI2-h")
    pp.add_argument("--patch-size", type=int, default=14)
    pp.add_argument("--no-pretrained", action="store_true")
    pp.add_argument("--ckpt-path", type=str, default="")
    pp.add_argument("--device", type=int, nargs="+", default=[0])

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "compute-prototypes":
        cmd_compute_prototypes(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
