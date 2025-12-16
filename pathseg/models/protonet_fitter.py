from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

from models.histo_encoder import Encoder
from training.tiler import GridPadTiler
from leace.leace import LeaceEraser

# If these live elsewhere, adjust imports accordingly.
from .protonet_utils import (
    accumulate_features_and_labels,
    subsample_tokens_balanced_by_image,
)


# -------------------------
# Config dataclasses
# -------------------------


@dataclass
class ProtoNetHeadConfig:
    metric: str = "l2"
    center_feats: bool = True
    normalize_feats: bool = True
    num_classes: Optional[int] = None
    embedding_dim: Optional[int] = None  # D_proj


class PCAMode(Enum):
    NONE = auto()
    N_COMPONENTS = auto()
    EVR_TARGET = auto()  # explained variance ratio


@dataclass
class ProtoNetFitConfig:
    num_classes: int
    ignore_idx: int = 255
    max_tokens_per_class: int = 0  # 0 => use all tokens
    metric: str = "l2"
    center_feats: bool = True
    normalize_feats: bool = True
    pca_mode: PCAMode = PCAMode.NONE
    pca_n_components: Optional[int] = None
    pca_evr_target: Optional[float] = None


class ProtoNetDecoderFitter:
    """Runs the offline statistics pipeline to build a ProtoNetHead and save a bundle.

    Responsibilities:
      - accumulate encoder token features + labels
      - optional LEACE on positions
      - optional PCA
      - mean-centering & normalization
      - compute class prototypes
      - save bundle for ProtoNetDecoder.
    """

    def __init__(
        self,
        encoder: Encoder,
        train_loader: DataLoader,
        img_tiler: GridPadTiler,
        target_tiler: GridPadTiler,
        grid_size: Tuple[int, int],
        fit_cfg: ProtoNetFitConfig,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.train_loader = train_loader
        self.img_tiler = img_tiler
        self.target_tiler = target_tiler
        self.grid_size = grid_size
        self.cfg = fit_cfg
        self.device = device

    @torch.no_grad()
    def fit_head(self) -> tuple[ProtoNetHead, dict[str, Any]]:
        Ht, Wt = self.grid_size
        num_classes = self.cfg.num_classes
        ignore_idx = self.cfg.ignore_idx

        X_all, y_all, img_ids_all, pos_eraser = accumulate_features_and_labels(
            encoder=self.encoder,
            dataloader=self.train_loader,
            img_tiler=self.img_tiler,
            target_tiler=self.target_tiler,
            grid_size=self.grid_size,
            ignore_idx=ignore_idx,
            device=self.device,
        )

        if X_all.shape[0] == 0:
            raise RuntimeError("No valid tokens collected – check masks & ignore_idx.")

        # Balanced subsampling across images
        X_used, y_used, img_ids_used = subsample_tokens_balanced_by_image(
            X_all,
            y_all,
            img_ids_all,
            num_classes=num_classes,
            max_tokens_per_class=self.cfg.max_tokens_per_class,
        )

        if X_used.shape[0] == 0:
            raise RuntimeError("No tokens left after balancing – decrease constraints.")

        # LEACE on positions (if available)
        if pos_eraser is not None:
            X_used = pos_eraser(X_used)

        N_used, encoder_dim = X_used.shape

        # PCA
        pca_mode = self.cfg.pca_mode
        pca_n_components = self.cfg.pca_n_components
        pca_evr_target = self.cfg.pca_evr_target

        if pca_mode is not PCAMode.NONE:
            if PCA is None:
                raise ImportError(
                    "scikit-learn is required for PCA. "
                    "Install with `pip install scikit-learn`."
                )

            if pca_mode is PCAMode.N_COMPONENTS:
                if pca_n_components is None or pca_n_components <= 0:
                    raise ValueError(
                        "ProtoNetFitConfig.pca_n_components must be > 0 "
                        "when pca_mode == PCAMode.N_COMPONENTS."
                    )
                pca = PCA(n_components=pca_n_components)
                pca_mode_str = "n_components"
            else:
                if pca_evr_target is None or not (0.0 < pca_evr_target <= 1.0):
                    raise ValueError(
                        "ProtoNetFitConfig.pca_evr_target must be in (0,1] "
                        "when pca_mode == PCAMode.EVR_TARGET."
                    )
                pca = PCA(n_components=pca_evr_target)
                pca_mode_str = "explained_variance_ratio"

            X_np = X_used.cpu().numpy()
            pca.fit(X_np)
            proj_matrix = torch.from_numpy(pca.components_.T).float()  # [D_in, D_proj]
            mean = torch.from_numpy(pca.mean_).float()  # [D_in]
            X_centered = X_used - mean
            Z = X_centered @ proj_matrix
            final_proj_dim = proj_matrix.shape[1]
        else:
            pca_mode_str = "none"
            proj_matrix = torch.eye(encoder_dim, encoder_dim, dtype=torch.float32)
            if self.cfg.center_feats:
                mean = X_used.mean(dim=0)
                Z = X_used - mean
            else:
                mean = torch.zeros(encoder_dim, dtype=torch.float32)
                Z = X_used
            final_proj_dim = encoder_dim

        if self.cfg.normalize_feats:
            Z = F.normalize(Z, dim=-1)

        # Prototypes
        prototypes = torch.zeros(num_classes, final_proj_dim, dtype=torch.float32)
        class_counts = torch.zeros(num_classes, dtype=torch.float32)

        for c in range(num_classes):
            mask_c = y_used == c
            if mask_c.any():
                Zc = Z[mask_c]
                prototypes[c] = Zc.mean(dim=0)
                class_counts[c] = float(mask_c.sum().item())

        head_cfg = ProtoNetHeadConfig(
            metric=self.cfg.metric.lower(),
            center_feats=self.cfg.center_feats,
            normalize_feats=self.cfg.normalize_feats,
            num_classes=num_classes,
            embedding_dim=final_proj_dim,
        )

        head = ProtoNetHead(
            prototypes=prototypes,
            mean=mean,
            proj_matrix=proj_matrix,
            class_counts=class_counts,
            config=head_cfg,
            leace_eraser=pos_eraser,
        )

        fit_meta: dict[str, Any] = {
            "num_tokens_collected": int(X_all.shape[0]),
            "num_tokens_used": int(X_used.shape[0]),
            "encoder_dim": int(encoder_dim),
            "proj_dim": int(final_proj_dim),
            "pca_mode": pca_mode_str,
            "pca_n_components": int(pca_n_components) if pca_n_components else None,
            "pca_evr_target": float(pca_evr_target) if pca_evr_target else None,
            "max_tokens_per_class": int(self.cfg.max_tokens_per_class),
            "ignore_idx": int(self.cfg.ignore_idx),
            "metric": self.cfg.metric.lower(),
            "center_feats": bool(self.cfg.center_feats),
            "normalize_feats": bool(self.cfg.normalize_feats),
            "grid_size": [int(Ht), int(Wt)],
        }

        return head, fit_meta

    def save_bundle(
        self,
        path: str | PathLike[str],
        encoder_meta: dict[str, Any],
    ) -> None:
        """Fit a head, build a bundle, and save it to disk."""
        head, fit_meta = self.fit_head()
        Ht, Wt = self.grid_size

        bundle: dict[str, Any] = {
            "encoder_meta": encoder_meta,
            "head": head.to_payload(extra_meta=fit_meta),
            "grid_size": (int(Ht), int(Wt)),
            "ignore_idx": int(self.cfg.ignore_idx),
            "tile": int(self.img_tiler.tile),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)
