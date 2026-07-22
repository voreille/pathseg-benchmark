from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from pathseg.models.encoder import Encoder

from pathseg.ood.residual_pca_scorer import FullPCA
from pathseg.utils.load_experiment import load_experiment


class FixedCropsPerPatientSampler(Sampler[int]):
    """
    Sample exactly `crops_per_patient` dataset entries per patient.

    Dataset entries correspond to ROIs. Sampling is performed with
    replacement because each access produces a new random crop.
    """

    def __init__(
        self,
        dataset,
        metadata: pd.DataFrame,
        crops_per_patient: int,
        seed: int = 0,
    ) -> None:
        if crops_per_patient <= 0:
            raise ValueError("crops_per_patient must be positive.")

        if not hasattr(dataset, "image_ids"):
            raise TypeError(
                "The dataset must expose its sample IDs through `image_ids`."
            )

        required_columns = {"sample_id", "patient_id"}
        missing = required_columns - set(metadata.columns)

        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}.")

        id_to_index = {
            str(image_id): index for index, image_id in enumerate(dataset.image_ids)
        }

        patient_indices: dict[str, list[int]] = defaultdict(list)

        for row in metadata.itertuples(index=False):
            sample_id = str(row.sample_id)

            # This automatically restricts fitting to the exact samples
            # contained in the datamodule's training dataset.
            if sample_id in id_to_index:
                patient_indices[str(row.patient_id)].append(id_to_index[sample_id])

        if not patient_indices:
            raise RuntimeError("No metadata sample IDs matched the training dataset.")

        rng = np.random.default_rng(seed)
        indices: list[int] = []

        for patient_id in sorted(patient_indices):
            available_indices = patient_indices[patient_id]

            sampled = rng.choice(
                available_indices,
                size=crops_per_patient,
                replace=True,
            )

            indices.extend(sampled.tolist())

        rng.shuffle(indices)

        self.indices = indices
        self.num_patients = len(patient_indices)
        self.crops_per_patient = crops_per_patient

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_ood_fit_loader(
    datamodule,
    metadata: pd.DataFrame,
    *,
    crops_per_patient: int,
    seed: int,
    batch_size: int | None = None,
) -> DataLoader:
    """
    Reuse the training dataset and collate function, but replace its
    ordinary sampler with patient-balanced sampling.
    """
    original_loader = datamodule.train_dataloader()
    dataset = original_loader.dataset

    sampler = FixedCropsPerPatientSampler(
        dataset=dataset,
        metadata=metadata,
        crops_per_patient=crops_per_patient,
        seed=seed,
    )

    num_workers = original_loader.num_workers

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size or original_loader.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=original_loader.collate_fn,
        pin_memory=original_loader.pin_memory,
        persistent_workers=(
            original_loader.persistent_workers if num_workers > 0 else False
        ),
    )


def unpack_batch(batch):
    """
    Supports:
        images, targets
        images, targets, image_ids
    """
    if len(batch) == 2:
        images, targets = batch
    elif len(batch) == 3:
        images, targets, _ = batch
    else:
        raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")

    if isinstance(images, (list, tuple)):
        images = torch.stack(list(images))

    return images, targets


def targets_to_token_mask(
    targets,
    grid_size: tuple[int, int],
    *,
    min_valid_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Convert Mask2Former targets to a valid-token mask.

    A pixel is considered valid when it belongs to at least one returned
    semantic mask. This assumes `return_background=True`, so that all
    annotated non-ignore pixels are represented in `target["masks"]`.

    Returns:
        Boolean tensor [B, N_tokens].
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    token_masks = []

    for target in targets:
        masks = torch.as_tensor(target["masks"])

        if masks.ndim != 3:
            raise ValueError(
                f"Expected target masks with shape [C, H, W], got {tuple(masks.shape)}."
            )

        if masks.shape[0] == 0:
            valid_pixels = torch.zeros(
                masks.shape[-2:],
                dtype=torch.bool,
            )
        else:
            valid_pixels = masks.bool().any(dim=0)

        valid_fraction = F.adaptive_avg_pool2d(
            valid_pixels[None, None].float(),
            output_size=grid_size,
        )[0, 0]

        token_masks.append((valid_fraction >= min_valid_fraction).flatten())

    return torch.stack(token_masks)


@torch.inference_mode()
def iter_valid_tokens(
    network,
    dataloader: DataLoader,
    device: torch.device,
    *,
    min_valid_fraction: float,
) -> Iterator[torch.Tensor]:
    """
    Extract valid spatial tokens as matrices [M, D].

    Flattening and validity selection happen before transferring tokens
    to the PCA fitting device.
    """
    network.eval()

    for batch in dataloader:
        images, targets = unpack_batch(batch)

        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            # Bypass LinearDecoder.forward and directly obtain the
            # Encoder spatial tokens [B, N, D].
            tokens = Encoder.forward(network, images)

        valid_token_mask = targets_to_token_mask(
            targets=targets,
            grid_size=network.grid_size,
            min_valid_fraction=min_valid_fraction,
        ).to(tokens.device)

        valid_tokens = tokens[valid_token_mask]

        if valid_tokens.numel() > 0:
            yield valid_tokens.detach().cpu()



@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Experiment configuration file.",
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Trained segmentation checkpoint.",
)
@click.option(
    "--metadata",
    "metadata_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Dataset metadata CSV.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Output file for the fitted full PCA.",
)
@click.option(
    "--crops-per-patient",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of randomly sampled crops per training patient.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Override the training dataloader batch size.",
)
@click.option(
    "--min-valid-fraction",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="Minimum annotated fraction for retaining a token.",
)
@click.option(
    "--normalize-embeddings/--no-normalize-embeddings",
    default=False,
    show_default=True,
    help="L2-normalize embeddings before fitting PCA.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--device",
    default=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    show_default="cuda when available, otherwise cpu",
    help="Device used for encoder inference.",
)
@click.option(
    "--pca-device",
    default="cpu",
    show_default=True,
    help="Device used to accumulate and decompose the covariance matrix.",
)
def main(
    config_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    output_path: Path,
    crops_per_patient: int,
    batch_size: int | None,
    min_valid_fraction: float,
    normalize_embeddings: bool,
    seed: int,
    device: str,
    pca_device: str,
) -> None:
    """Fit a full PCA on patient-balanced training embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_device = torch.device(device)

    experiment = load_experiment(
        config_path=str(config_path),
        ckpt_path=str(checkpoint_path),
        stage="fit",
        device=torch_device,
        eval_mode=True,
    )

    model = experiment["model"]
    datamodule = experiment["datamodule"]
    network = model.network

    metadata = pd.read_csv(metadata_path)

    fit_loader = build_ood_fit_loader(
        datamodule=datamodule,
        metadata=metadata,
        crops_per_patient=crops_per_patient,
        seed=seed,
        batch_size=batch_size,
    )

    sampler = fit_loader.sampler

    click.echo(
        f"Sampling {len(sampler)} crops from "
        f"{sampler.num_patients} patients "
        f"({sampler.crops_per_patient} per patient)."
    )

    pca = FullPCA(
        feature_dim=network.embed_dim,
        normalize_embeddings=normalize_embeddings,
    )

    pca.fit(
        iter_valid_tokens(
            network=network,
            dataloader=fit_loader,
            device=torch_device,
            min_valid_fraction=min_valid_fraction,
        ),
        fit_device=pca_device,
        fit_dtype=torch.float64,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "pca_state_dict": pca.state_dict(),
            "feature_dim": pca.feature_dim,
            "normalize_embeddings": pca.normalize_embeddings,
            "n_fit_tokens": pca.n_fit_tokens,
            "crops_per_patient": crops_per_patient,
            "min_valid_fraction": min_valid_fraction,
            "seed": seed,
            "segmentation_checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
        },
        output_path,
    )

    click.echo(f"Saved PCA fitted on {pca.n_fit_tokens:,} tokens to {output_path}.")


if __name__ == "__main__":
    main()
