from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click
import torch

from models.histo_encoder import Encoder
from models.histo_protonet_decoder import (
    ProtoNetFitConfig,
    ProtoNetDecoderFitter,
    PCAMode,
)
from training.tiler import GridPadTiler
from training.protonet_utils import (
    parse_devices,
    build_train_loader_fewshot,
)


@click.command()
@click.option("--root-dir", type=str, required=True, help="ANORAK dataset root.")
@click.option(
    "--img-size",
    type=int,
    nargs=2,
    default=(448, 448),
    show_default=True,
    metavar="H W",
)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--num-classes", type=int, default=7, show_default=True)
@click.option("--ignore-idx", type=int, default=255, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--prefetch-factor", type=int, default=2, show_default=True)
@click.option("--num-metrics", type=int, default=1, show_default=True)
@click.option(
    "--weighted-blend",
    is_flag=True,
    default=False,
    help="Use Hann weighting during stitching (for images only).",
)
# model / encoder
@click.option(
    "--encoder-id",
    type=str,
    default="h0-mini",
    show_default=True,
    help="Backbone encoder id.",
)
@click.option("--ckpt-path", type=str, default="", show_default=True)
@click.option(
    "--metric",
    type=click.Choice(["L2", "cosine"], case_sensitive=False),
    default="L2",
    show_default=True,
)
@click.option("--no-center-feats", is_flag=True, default=False)
@click.option("--no-normalize-feats", is_flag=True, default=False)
# PCA / proto
@click.option(
    "--proj-dim",
    type=int,
    default=None,
    help="PCA dimension (n_components). If <=0 and --pca-evr<=0, PCA is disabled.",
)
@click.option(
    "--pca-evr",
    type=float,
    default=0.0,
    show_default=True,
    help="PCA explained variance ratio (0<evr<=1). Mutually exclusive with proj-dim>0.",
)
@click.option(
    "--max-tokens-per-class",
    type=int,
    default=0,
    show_default=True,
    help="Max tokens per class (balanced across images). <=0 means use all tokens.",
)
# device / output
@click.option(
    "--device",
    "-d",
    type=int,
    multiple=True,
    default=(0,),
    show_default=True,
    help="GPU ids (first is used for encoder).",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Where to save the ProtoNet bundle (.pt).",
)
@click.option(
    "--fewshot-csv",
    type=str,
    required=True,
    help="Path to the few-shot CSV file.",
)
@click.option(
    "--n-shot",
    type=int,
    default=1,
    show_default=True,
    help="Number of shots per class.",
)
@click.option(
    "--support-set",
    type=int,
    default=0,
    show_default=True,
    help="Support set identifier.",
)
def main(
    root_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    num_classes: int,
    ignore_idx: int,
    num_workers: int,
    prefetch_factor: int,
    num_metrics: int,
    weighted_blend: bool,
    encoder_id: str,
    ckpt_path: str,
    metric: str,
    no_center_feats: bool,
    no_normalize_feats: bool,
    proj_dim: int | None,
    pca_evr: float,
    max_tokens_per_class: int,
    device: tuple[int, ...],
    output_path: str,
    fewshot_csv: str,
    n_shot: int,
    support_set: int,
) -> None:
    """Fit ProtoNet head and save a bundle for ProtoNetDecoder."""
    devices = list(device)
    dev = parse_devices(devices)
    img_size = (int(img_size[0]), int(img_size[1]))

    if img_size[0] != img_size[1]:
        raise ValueError(f"--img-size must be square for now, got {img_size}.")
    tile_size = img_size[0]

    center_enabled = not no_center_feats
    norm_enabled = not no_normalize_feats

    # 1) Encoder
    encoder = (
        Encoder(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=False,
            ckpt_path=ckpt_path,
        )
        .to(dev)
        .eval()
    )
    grid_size = tuple(encoder.grid_size)

    # 2) Tilers
    img_tiler = GridPadTiler(
        tile=tile_size,
        stride=tile_size,
        weighted_blend=weighted_blend,
        pad_mode="replicate",
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=tile_size,
        stride=tile_size,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=float(ignore_idx),
    )

    # 3) Loader
    train_loader = build_train_loader_fewshot(
        root_dir=root_dir,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=num_classes,
        num_metrics=num_metrics,
        ignore_idx=ignore_idx,
        n_shot=n_shot,
        support_set=support_set,
        prefetch_factor=prefetch_factor,
        fewshot_csv=fewshot_csv,
    )

    # 4) Fit config
    if proj_dim is not None and proj_dim > 0 and pca_evr > 0.0:
        raise ValueError("Use either --proj-dim or --pca-evr, not both.")

    if proj_dim is not None and proj_dim > 0:
        pca_mode = PCAMode.N_COMPONENTS
        pca_n_components = proj_dim
        pca_evr_target = None
    elif pca_evr > 0.0:
        pca_mode = PCAMode.EVR_TARGET
        pca_n_components = None
        pca_evr_target = pca_evr
    else:
        pca_mode = PCAMode.NONE
        pca_n_components = None
        pca_evr_target = None

    fit_cfg = ProtoNetFitConfig(
        num_classes=num_classes,
        ignore_idx=ignore_idx,
        max_tokens_per_class=max_tokens_per_class,
        metric=metric,
        center_feats=center_enabled,
        normalize_feats=norm_enabled,
        pca_mode=pca_mode,
        pca_n_components=pca_n_components,
        pca_evr_target=pca_evr_target,
    )

    fitter = ProtoNetDecoderFitter(
        encoder=encoder,
        train_loader=train_loader,
        img_tiler=img_tiler,
        target_tiler=tgt_tiler,
        grid_size=grid_size,
        fit_cfg=fit_cfg,
        device=dev,
    )

    encoder_meta = {
        "encoder_id": encoder_id,
        "img_size": img_size,
        "ckpt_path": ckpt_path,
        "sub_norm": False,
    }

    output_path = Path(output_path)
    fitter.fit_and_save_bundle(output_path, encoder_meta=encoder_meta)
    click.echo(f"[OK] saved ProtoNet bundle to {output_path}")


if __name__ == "__main__":
    main()
