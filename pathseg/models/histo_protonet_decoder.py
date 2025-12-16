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
from training.protonet_utils import (
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


# -------------------------
# ProtoNetHead
# -------------------------


class ProtoNetHead(nn.Module):
    """Projection + prototypes + distance metric."""

    def __init__(
        self,
        prototypes: Tensor,  # [C, D_proj]
        mean: Tensor,  # [D_in]
        proj_matrix: Tensor,  # [D_in, D_proj]
        class_counts: Optional[Tensor] = None,
        config: Optional[ProtoNetHeadConfig] = None,
        leace_eraser: Optional[LeaceEraser] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = ProtoNetHeadConfig()
        self.config = config

        self.metric = config.metric.lower()
        assert self.metric in ("l2", "cosine")
        self.center_feats = config.center_feats
        self.normalize_feats = config.normalize_feats

        self.register_buffer("prototypes", prototypes, persistent=True)
        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("proj_matrix", proj_matrix, persistent=True)

        if class_counts is None:
            class_counts = torch.zeros(prototypes.shape[0], dtype=torch.long)
        self.register_buffer("class_counts", class_counts, persistent=True)

        self._leace = leace_eraser

    def _project(self, X: Tensor) -> Tensor:
        # 1) LEACE
        if self._leace is not None:
            eraser = self._leace.to(X.device)
            X = eraser(X)

        # 2) mean-centering
        if self.center_feats:
            X = X - self.mean

        # 3) linear projection
        Z = X @ self.proj_matrix

        # 4) normalization
        if self.normalize_feats:
            Z = F.normalize(Z, dim=-1)

        return Z

    def forward(self, X: Tensor) -> Tensor:
        """X: [B,Q,D_in] or [N,D_in] -> scores: [B,Q,C] or [N,C]."""
        if X.ndim == 3:
            B, Q, D = X.shape
            X_flat = X.reshape(B * Q, D)
        elif X.ndim == 2:
            X_flat = X
            B = Q = None
        else:
            raise ValueError("X must be [B,Q,D] or [N,D]")

        Z = self._project(X_flat)  # [N, D_proj]
        P = self.prototypes  # [C, D_proj]

        if self.metric == "l2":
            diff = Z.unsqueeze(1) - P.unsqueeze(0)  # [N, C, D_proj]
            scores = -diff.norm(dim=-1, p=2)
        else:
            Z_n = F.normalize(Z, dim=-1)
            P_n = F.normalize(P, dim=-1)
            scores = Z_n @ P_n.T

        if B is not None:
            scores = scores.view(B, Q, -1)
        return scores

    # ----------- serialization ------------

    def to_payload(self, extra_meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "config": asdict(self.config),
            "leace_position_eraser": (
                self._leace.state_dict() if self._leace is not None else None
            ),
            "meta": extra_meta or {},
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ProtoNetHead":
        cfg = ProtoNetHeadConfig(**payload["config"])
        sd: dict[str, Tensor] = payload["state_dict"]

        leace = None
        leace_state = payload.get("leace_position_eraser", None)
        if leace_state is not None:
            leace = LeaceEraser.from_state_dict(leace_state)

        head = cls(
            prototypes=sd["prototypes"],
            mean=sd["mean"],
            proj_matrix=sd["proj_matrix"],
            class_counts=sd.get("class_counts", None),
            config=cfg,
            leace_eraser=leace,
        )
        head.load_state_dict(sd, strict=False)
        return head

    def save(
        self,
        path: str | PathLike[str],
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        torch.save(self.to_payload(extra_meta=extra_meta), path)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        map_location: str | torch.device | None = "cpu",
    ) -> "ProtoNetHead":
        payload = torch.load(path, map_location=map_location)
        return cls.from_payload(payload)


# -------------------------
# ProtoNetDecoderFitter
# -------------------------


class ProtoNetDecoderFitter:
    """Offline statistics pipeline to build a ProtoNetHead and save a bundle."""

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

        X_used, y_used, img_ids_used = subsample_tokens_balanced_by_image(
            X_all,
            y_all,
            img_ids_all,
            num_classes=num_classes,
            max_tokens_per_class=self.cfg.max_tokens_per_class,
        )

        if X_used.shape[0] == 0:
            raise RuntimeError("No tokens left after balancing – decrease constraints.")

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

    def fit_and_save_bundle(
        self,
        path: str | PathLike[str],
        encoder_meta: dict[str, Any],
    ) -> None:
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
        print(f"Proj shape of fitted ProtoNetHead: {head.proj_matrix.shape}")


# -------------------------
# ProtoNetDecoder (runtime)
# -------------------------


class ProtoNetDecoder(Encoder):
    """
    Encoder → token features → ProtoNetHead → per-patch logits.

    Expected bundle format (saved by ProtoNetDecoderFitter.save_bundle):
      {
        "encoder_meta": {...},
        "head": ProtoNetHead.to_payload(...),
        "grid_size": (Ht, Wt),
        "ignore_idx": int,
        "tile": int,
        ...
      }
    """

    def __init__(
        self,
        prototypes_path: str | Path,
        encoder_id: Optional[str] = None,
        img_size: Optional[Tuple[int, int]] = None,
        ckpt_path: Optional[str] = None,
        sub_norm: Optional[bool] = None,
    ) -> None:
        prototypes_path = Path(prototypes_path)
        bundle = torch.load(prototypes_path, map_location="cpu")

        enc_meta = bundle.get("encoder_meta", {})

        enc_id = encoder_id if encoder_id is not None else enc_meta.get("encoder_id")
        if enc_id is None:
            raise ValueError(
                "encoder_id must be provided or present in bundle['encoder_meta']."
            )

        img_sz = img_size or tuple(enc_meta.get("img_size", (448, 448)))
        ckpt = ckpt_path if ckpt_path is not None else enc_meta.get("ckpt_path", "")
        subn = sub_norm if sub_norm is not None else enc_meta.get("sub_norm", False)

        super().__init__(
            encoder_id=enc_id,
            img_size=img_sz,
            ckpt_path=ckpt,
            sub_norm=subn,
        )

        self.head = ProtoNetHead.from_payload(bundle["head"])
        self.grid_size = tuple(bundle.get("grid_size", tuple(self.grid_size)))
        self.num_classes = int(self.head.prototypes.shape[0])
        self.ignore_idx = int(bundle.get("ignore_idx", 255))
        self.tile = int(bundle.get("tile", img_sz[0]))

    @torch.no_grad()
    def tokens_from_images(self, imgs: Tensor) -> Tensor:
        return super().forward(imgs)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, C, H, W] → logits: [B, num_classes, Ht, Wt]
        """
        tokens = self.tokens_from_images(x)  # [B, Q, D_in]
        B, Q, D = tokens.shape
        Ht, Wt = self.grid_size
        if Q != Ht * Wt:
            raise ValueError(f"Q={Q} must equal Ht*Wt={Ht * Wt}.")

        scores = self.head(tokens)  # [B, Q, C]
        scores = scores.transpose(1, 2)  # [B, C, Q]
        return scores.reshape(B, self.num_classes, Ht, Wt)
