from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder
from pathseg.models.refiner_layers import (
    ProtoProjLite,
    LayerNorm2d,
    ProtoProjGN,
    ProtoProjDilatedSpatialScaleAttn,
    ProtoProjPSPAttention,
)


def build_protoproj(
    protoproj_id: str,
    embed_dim: int,
    proto_dim: int,
    hidden_dim: int,
    protoproj_kwargs: dict | None = None,
) -> nn.Module:
    if protoproj_kwargs is None:
        protoproj_kwargs = {}
    if protoproj_id == "linear":
        return nn.Conv2d(
            embed_dim,
            proto_dim,
            kernel_size=1,
            bias=False,
        )
    elif protoproj_id == "protoproj_gn":
        return ProtoProjGN(
            embed_dim,
            proto_dim,
            hidden_dim=hidden_dim,
        )
    elif protoproj_id == "protoproj_dilated_spatial_scale_attn":
        return ProtoProjDilatedSpatialScaleAttn(
            embed_dim,
            proto_dim,
        )
    elif protoproj_id == "psp_attention":
        return ProtoProjPSPAttention(
            in_dim=embed_dim,
            out_dim=proto_dim,
            pool_bins=protoproj_kwargs.get("pool_bins", (1, 2, 4, 8)),
            use_residual_refine=protoproj_kwargs.get("use_residual_refine", True),
        )

    elif protoproj_id == "none":
        if proto_dim != embed_dim:
            print(
                f"Warning: protoproj_id='none' but proto_dim={proto_dim} != embed_dim={embed_dim}. Ignoring proto_dim and using embed_dim as proto_dim."
            )
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported protoproj_id={protoproj_id}")


class TwoStagesCompartmentPrototypeDecoder(Encoder):
    """
    Two-stage prototype decoder.

    Stage 1
    -------
    Head A predicts tissue compartments.

    A presegmentation head takes Head A logits and predicts a single-channel
    foreground logit. Its sigmoid gives a soft foreground probability map
    indicating regions likely to contain pattern annotations.

    Stage 2
    -------
    Pattern prototypes are fitted only from foreground regions, using the
    foreground probability as a soft weight together with the support masks.

    Head B outputs logits only for pattern classes (no background class).
    Background handling is delegated to the presegmentation head at inference.
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        center_per_image: bool = False,
        num_compartments: int = 16,
        proto_dim: int = 256,
        num_pattern_classes: int = 6,  # only used for the auxiliary head if use_aux_b_head=True, otherwise main head has no fixed num_classes
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        detach_preseg_input: bool = True,
        detach_proto_features: bool = True,
        protoproj_id: str = "linear",
        protoproj_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_compartments = int(num_compartments)
        self.num_pattern_classes = int(num_pattern_classes)

        if center and center_per_image:
            raise ValueError("center and center_per_image cannot both be True")
        self.center = bool(center)
        self.center_per_image = bool(center_per_image)

        self.eps = float(eps)

        self.detach_preseg_input = bool(detach_preseg_input)
        self.detach_proto_features = bool(detach_proto_features)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)

        # Single-channel FG logit from compartment logits
        self.presegmentation_head = nn.Conv2d(
            self.num_compartments,
            1,
            kernel_size=1,
        )

        self.proto_proj = build_protoproj(
            protoproj_id=protoproj_id,
            embed_dim=self.embed_dim,
            proto_dim=proto_dim,
            hidden_dim=protoproj_kwargs.get("hidden_dim", 512)
            if protoproj_kwargs is not None
            else 512,
        )

        self.head_b_aux = (
            nn.Conv2d(self.embed_dim, self.num_pattern_classes + 1, kernel_size=1)
            if use_aux_b_head
            else None
        )

        log_temp = torch.log(torch.tensor(float(temperature_init)))
        if learnable_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

    # --------------------------------------------------------
    # utilities
    # --------------------------------------------------------
    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))

    def _empty_ctx(
        self,
        device: torch.device,
        proto_dim: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "prototypes": torch.empty(0, proto_dim, device=device),
            "label_ids": torch.empty(0, dtype=torch.long, device=device),
        }

    # --------------------------------------------------------
    # inference helper
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_segmentation(
        self,
        out: dict[str, torch.Tensor],
        fg_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Convert model outputs to a full semantic prediction.

        Returns:
            pred: [B,H,W]
                0 = background
                1..K = pattern classes
        """
        fg_prob = out["fg_prob"]  # [B,1,H,W]
        logits_b = out["logits_b"]  # [B,K,H,W]

        pred_patterns = torch.argmax(logits_b, dim=1) + 1  # [B,H,W], shift by +1
        pred = pred_patterns.clone()
        pred[fg_prob[:, 0] < fg_threshold] = 0
        return pred

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.forward_features_map(x)

        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        preseg_input = logits_a.detach() if self.detach_preseg_input else logits_a
        logits_preseg = self.presegmentation_head(preseg_input)  # [B,1,H,W]
        fg_prob = torch.sigmoid(logits_preseg)  # [B,1,H,W]

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
            "logits_preseg": logits_preseg,
            "fg_prob": fg_prob,
        }

        if self.head_b_aux is not None:
            out["logits_b_aux"] = self.head_b_aux(feat)

        if ctx is None:
            return out

        b, _, h, w = feat.shape
        num_fg_classes = int(ctx["num_fg_classes"].item())
        logits_b = feat.new_zeros((b, num_fg_classes, h, w))

        proto_input = feat.detach() if self.detach_proto_features else feat
        proto_feat = self.proto_proj(proto_input)

        if self.center_per_image:
            proto_feat = proto_feat - proto_feat.mean(dim=[2, 3], keepdim=True)

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        pair_logits = torch.einsum("bdhw,pd->bphw", proto_feat, ctx["prototypes"])
        pair_logits = pair_logits * torch.exp(self.log_temp)

        for p, lab in enumerate(ctx["label_ids"].tolist()):
            logits_b[:, int(lab) - 1] = logits_b[:, int(lab) - 1] + pair_logits[:, p]

        out["logits_b"] = logits_b
        out["label_ids_b"] = torch.unique(ctx["label_ids"], sorted=True)
        out["pair_label_ids_b"] = ctx["label_ids"]
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S,K+1,H,W], K = pattern classes only and +1 for bg
    ) -> dict[str, torch.Tensor]:
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)
        logits_a = self.head_a(feat)

        preseg_input = logits_a.detach() if self.detach_preseg_input else logits_a
        logits_preseg = self.presegmentation_head(preseg_input)
        fg_prob = torch.sigmoid(logits_preseg)  # [S,1,H,W]

        proto_feat = self.proto_proj(feat)

        s, d, h, w = proto_feat.shape
        num_fg_classes = pattern_targets.shape[1] - 1

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if self.center_per_image:
            proto_feat = proto_feat - proto_feat.mean(dim=[2, 3], keepdim=True)

        if self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(dim=0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        patt_flat = pattern_targets.reshape(s, num_fg_classes + 1, h * w)  # [S,K+1,N]
        fg_flat = fg_prob.reshape(s, h * w)  # [S,N]

        prototypes_list: list[torch.Tensor] = []
        label_ids_list: list[int] = []

        for k in range(1, num_fg_classes + 1):
            weights = patt_flat[:, k] * fg_flat
            weight_sum = weights.sum()

            if weight_sum <= self.eps:
                continue

            proto_sum = torch.einsum("sn,snd->d", weights, feat_flat)
            prototype = proto_sum / weight_sum.clamp_min(self.eps)
            prototype = F.normalize(prototype, dim=0)

            prototypes_list.append(prototype)
            label_ids_list.append(int(k))

        if len(prototypes_list) == 0:
            ctx = self._empty_ctx(device=device, proto_dim=d)
            if center is not None:
                ctx["center"] = center
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes_list, dim=0),  # [P,D]
            "label_ids": torch.tensor(label_ids_list, dtype=torch.long, device=device),
            "num_fg_classes": torch.tensor(
                num_fg_classes, dtype=torch.long, device=device
            ),
        }
        if center is not None:
            ctx["center"] = center
        return ctx

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = images.device
        images = images.to(device=device)
        image_labels = image_labels.to(device=device)

        self._validate_image_labels(
            images=images,
            image_labels=image_labels,
        )

        pattern_targets, label_mapping = self._make_full_image_pattern_targets(
            images=images,
            image_labels=image_labels,
        )

        ctx = self.fit_prototypes(
            support_images=images,
            pattern_targets=pattern_targets,
        )
        ctx["label_mapping"] = label_mapping
        return ctx

    # --------------------------------------------------------
    # label utilities
    # --------------------------------------------------------
    def _make_full_image_pattern_targets(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, int]]:
        s, _, h, w = images.shape
        device = images.device
        image_labels = image_labels.to(device=device, dtype=torch.long)
        label_mapping = {
            idx + 1: lab.item() for idx, lab in enumerate(image_labels.unique())
        }
        image_labels_one_hot_idx = torch.zeros_like(image_labels, dtype=torch.long)
        for idx, lab in label_mapping.items():
            image_labels_one_hot_idx[image_labels == lab] = idx

        num_fg_classes = len(label_mapping)

        pattern_targets = torch.zeros(
            s,
            int(num_fg_classes) + 1,
            h,
            w,
            dtype=torch.float32,
            device=device,
        )
        pattern_targets[
            torch.arange(s, device=device),
            image_labels_one_hot_idx,
        ] = 1.0
        return pattern_targets, label_mapping

    def _validate_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [S,C,H,W], got {tuple(images.shape)}"
            )
        if image_labels.ndim != 1:
            raise ValueError(
                f"image_labels must have shape [S], got {tuple(image_labels.shape)}"
            )
        if images.shape[0] != image_labels.shape[0]:
            raise ValueError("images and image_labels must have the same batch size")


class TwoStagesCompartmentPrototypeDecoderLocalKMeans(
    TwoStagesCompartmentPrototypeDecoder
):
    """
    Variant of TwoStagesCompartmentPrototypeDecoder with:
    - linear prototype projection (1x1 conv)
    - multiple local prototypes per class from weighted k-means on support pixels
    - per-class aggregation with logsumexp instead of summing one prototype/class

    Notes
    -----
    - Background handling is unchanged: still delegated to presegmentation.
    - K-means is done independently for each foreground class on support pixels.
    - Prototypes are stored in ctx["prototypes"] with their class IDs in ctx["label_ids"].
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        center_per_image: bool = False,
        num_compartments: int = 16,
        proto_dim: int = 256,
        num_pattern_classes: int = 6,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        detach_preseg_input: bool = True,
        detach_proto_features: bool = True,
        num_local_prototypes: int = 4,
        kmeans_iters: int = 10,
        kmeans_sample_limit: int = 2048,
        kmeans_weight_thresh: float = 1e-3,
        protoproj_id: str = "linear",
        class_logit_pool: str = "logsumexp",
        protoproj_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            center_per_image=center_per_image,
            num_compartments=num_compartments,
            proto_dim=proto_dim,
            num_pattern_classes=num_pattern_classes,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
            use_aux_b_head=use_aux_b_head,
            detach_preseg_input=detach_preseg_input,
            detach_proto_features=detach_proto_features,
            protoproj_id=protoproj_id,
            protoproj_kwargs=protoproj_kwargs,
        )

        self.num_local_prototypes = int(num_local_prototypes)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_sample_limit = int(kmeans_sample_limit)
        self.kmeans_weight_thresh = float(kmeans_weight_thresh)
        self.class_logit_pool = str(class_logit_pool)

        if self.num_local_prototypes < 1:
            raise ValueError("num_local_prototypes must be >= 1")
        if self.kmeans_iters < 1:
            raise ValueError("kmeans_iters must be >= 1")
        if self.class_logit_pool not in {"logsumexp", "max", "mean"}:
            raise ValueError("class_logit_pool must be one of: logsumexp, max, mean")

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _empty_ctx(
        self,
        device: torch.device,
        proto_dim: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "prototypes": torch.empty(0, proto_dim, device=device),
            "label_ids": torch.empty(0, dtype=torch.long, device=device),
            "num_fg_classes": torch.tensor(0, dtype=torch.long, device=device),
        }

    def _aggregate_class_logits(self, pair_logits: torch.Tensor) -> torch.Tensor:
        """
        pair_logits: [B, M, H, W] for the M prototypes of one class
        returns: [B, H, W]
        """
        if pair_logits.shape[1] == 0:
            raise ValueError("pair_logits must contain at least one prototype")

        if self.class_logit_pool == "logsumexp":
            return torch.logsumexp(pair_logits, dim=1)
        if self.class_logit_pool == "max":
            return pair_logits.max(dim=1).values
        if self.class_logit_pool == "mean":
            return pair_logits.mean(dim=1)
        raise RuntimeError(f"Unsupported class_logit_pool={self.class_logit_pool}")

    def _subsample_weighted(
        self,
        feats: torch.Tensor,  # [N, D]
        weights: torch.Tensor,  # [N]
        max_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = feats.shape[0]
        if n <= max_samples:
            return feats, weights

        probs = weights / weights.sum().clamp_min(self.eps)
        idx = torch.multinomial(probs, num_samples=max_samples, replacement=False)
        return feats[idx], weights[idx]

    def _init_kmeans_centers_weighted(
        self,
        feats: torch.Tensor,  # [N, D], already normalized
        weights: torch.Tensor,  # [N]
        num_centers: int,
    ) -> torch.Tensor:
        """
        Weighted farthest-point style initialization.
        """
        n, d = feats.shape
        if num_centers == 1:
            center = (weights[:, None] * feats).sum(dim=0) / weights.sum().clamp_min(
                self.eps
            )
            return F.normalize(center[None], dim=1)

        probs = weights / weights.sum().clamp_min(self.eps)
        first_idx = torch.multinomial(probs, num_samples=1).item()
        centers = [feats[first_idx]]

        min_dist = 1.0 - (feats @ centers[0].unsqueeze(1)).squeeze(1)  # cosine dist

        for _ in range(1, num_centers):
            score = min_dist.clamp_min(0.0) * weights
            if float(score.sum()) <= self.eps:
                next_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                next_idx = torch.argmax(score).item()
            centers.append(feats[next_idx])
            dist_new = 1.0 - (feats @ centers[-1].unsqueeze(1)).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_new)

        centers = torch.stack(centers, dim=0)  # [K, D]
        return F.normalize(centers, dim=1)

    def _weighted_kmeans(
        self,
        feats: torch.Tensor,  # [N, D], normalized
        weights: torch.Tensor,  # [N]
        num_centers: int,
        num_iters: int,
    ) -> torch.Tensor:
        """
        Weighted spherical-ish k-means:
        - assignment by cosine similarity
        - centroid update by weighted average then renormalization
        """
        n, d = feats.shape
        k = min(int(num_centers), n)

        if k == 0:
            return feats.new_empty((0, d))
        if k == 1:
            center = (weights[:, None] * feats).sum(dim=0) / weights.sum().clamp_min(
                self.eps
            )
            return F.normalize(center[None], dim=1)

        centers = self._init_kmeans_centers_weighted(feats, weights, k)

        for _ in range(num_iters):
            sims = feats @ centers.t()  # [N, K]
            assign = sims.argmax(dim=1)  # [N]

            new_centers = []
            for j in range(k):
                mask = assign == j
                if not mask.any():
                    # Re-seed empty cluster with high-weight far point.
                    fallback_idx = torch.argmax(weights).item()
                    c = feats[fallback_idx]
                else:
                    wj = weights[mask]
                    fj = feats[mask]
                    c = (wj[:, None] * fj).sum(dim=0) / wj.sum().clamp_min(self.eps)
                new_centers.append(c)

            centers = F.normalize(torch.stack(new_centers, dim=0), dim=1)

        return centers

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.forward_features_map(x)

        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        preseg_input = logits_a.detach() if self.detach_preseg_input else logits_a
        logits_preseg = self.presegmentation_head(preseg_input)
        fg_prob = torch.sigmoid(logits_preseg)

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
            "logits_preseg": logits_preseg,
            "fg_prob": fg_prob,
        }

        if self.head_b_aux is not None:
            out["logits_b_aux"] = self.head_b_aux(feat)

        if ctx is None:
            return out

        b, _, h, w = feat.shape
        num_fg_classes = int(ctx["num_fg_classes"].item())
        logits_b = feat.new_zeros((b, num_fg_classes, h, w))

        proto_input = feat.detach() if self.detach_proto_features else feat
        proto_feat = self.proto_proj(proto_input)

        if self.center_per_image:
            proto_feat = proto_feat - proto_feat.mean(dim=(2, 3), keepdim=True)

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        prototypes = ctx["prototypes"]  # [P, D]
        label_ids = ctx["label_ids"]  # [P]

        if prototypes.numel() == 0:
            out["logits_b"] = logits_b
            out["label_ids_b"] = torch.empty(0, dtype=torch.long, device=feat.device)
            out["pair_label_ids_b"] = label_ids
            return out

        pair_logits = torch.einsum("bdhw,pd->bphw", proto_feat, prototypes)
        pair_logits = pair_logits * torch.exp(self.log_temp)

        unique_labs = torch.unique(label_ids, sorted=True)
        for lab in unique_labs.tolist():
            idx = torch.where(label_ids == lab)[0]
            class_pair_logits = pair_logits[:, idx]  # [B, M, H, W]
            logits_b[:, int(lab) - 1] = self._aggregate_class_logits(class_pair_logits)

        out["logits_b"] = logits_b
        out["label_ids_b"] = unique_labs
        out["pair_label_ids_b"] = label_ids
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S,K+1,H,W]
    ) -> dict[str, torch.Tensor]:
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)
        logits_a = self.head_a(feat)

        preseg_input = logits_a.detach() if self.detach_preseg_input else logits_a
        logits_preseg = self.presegmentation_head(preseg_input)
        fg_prob = torch.sigmoid(logits_preseg)  # [S,1,H,W]

        proto_feat = self.proto_proj(feat)

        s, d, h, w = proto_feat.shape
        num_fg_classes = pattern_targets.shape[1] - 1

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if self.center_per_image:
            proto_feat = proto_feat - proto_feat.mean(dim=(2, 3), keepdim=True)
            center = None
        elif self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(dim=0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s * h * w, d)  # [SN, D]
        patt_flat = pattern_targets.reshape(s, num_fg_classes + 1, h * w)
        patt_flat = patt_flat.permute(0, 2, 1).reshape(s * h * w, num_fg_classes + 1)
        fg_flat = fg_prob.reshape(s * h * w)  # [SN]

        prototypes_list: list[torch.Tensor] = []
        label_ids_list: list[int] = []

        for k in range(1, num_fg_classes + 1):
            class_mask = patt_flat[:, k] > 0.5
            if not class_mask.any():
                continue

            weights_k = patt_flat[class_mask, k] * fg_flat[class_mask]
            valid = weights_k > self.kmeans_weight_thresh
            if not valid.any():
                continue

            feats_k = feat_flat[class_mask][valid]  # [Nk, D]
            weights_k = weights_k[valid]  # [Nk]

            if feats_k.shape[0] == 0:
                continue

            feats_k, weights_k = self._subsample_weighted(
                feats_k,
                weights_k,
                max_samples=self.kmeans_sample_limit,
            )

            num_centers_k = min(self.num_local_prototypes, feats_k.shape[0])

            centers_k = self._weighted_kmeans(
                feats=feats_k,
                weights=weights_k,
                num_centers=num_centers_k,
                num_iters=self.kmeans_iters,
            )  # [Mk, D]

            for c in centers_k:
                prototypes_list.append(F.normalize(c, dim=0))
                label_ids_list.append(int(k))

        if len(prototypes_list) == 0:
            ctx = self._empty_ctx(device=device, proto_dim=d)
            if center is not None:
                ctx["center"] = center
            ctx["num_fg_classes"] = torch.tensor(
                num_fg_classes, dtype=torch.long, device=device
            )
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes_list, dim=0),  # [P, D]
            "label_ids": torch.tensor(label_ids_list, dtype=torch.long, device=device),
            "num_fg_classes": torch.tensor(
                num_fg_classes, dtype=torch.long, device=device
            ),
        }
        if center is not None:
            ctx["center"] = center
        return ctx


class TwoStagesCompartmentPrototypeDecoderLocalKMeansOpenSet(
    TwoStagesCompartmentPrototypeDecoderLocalKMeans
):
    """
    Local-kmeans prototype decoder with an extra episodic unknown logit.

    For an episode containing K supported foreground classes:
        logits_b_known: [B, K, H, W]
        logits_b_open:  [B, K+1, H, W]
    where the last channel is the learned "unknown foreground" logit.

    Intended training regime:
    - class dropout on support
    - supported query fg pixels -> known class channels
    - unsupported query fg pixels -> unknown channel
    - background handled by preseg, ignored in head B loss
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        center_per_image: bool = False,
        num_compartments: int = 16,
        proto_dim: int = 256,
        num_pattern_classes: int = 6,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        detach_preseg_input: bool = True,
        detach_proto_features: bool = True,
        num_local_prototypes: int = 4,
        kmeans_iters: int = 10,
        kmeans_sample_limit: int = 2048,
        kmeans_weight_thresh: float = 1e-3,
        class_logit_pool: str = "logsumexp",
        unknown_mode: str = "max",
        unknown_init_bias: float = 0.0,
        unknown_init_scale: float = 1.0,
        protoproj_id: str = "linear",
        protoproj_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            center_per_image=center_per_image,
            num_compartments=num_compartments,
            proto_dim=proto_dim,
            num_pattern_classes=num_pattern_classes,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
            use_aux_b_head=use_aux_b_head,
            detach_preseg_input=detach_preseg_input,
            detach_proto_features=detach_proto_features,
            num_local_prototypes=num_local_prototypes,
            kmeans_iters=kmeans_iters,
            kmeans_sample_limit=kmeans_sample_limit,
            kmeans_weight_thresh=kmeans_weight_thresh,
            class_logit_pool=class_logit_pool,
            protoproj_id=protoproj_id,
            protoproj_kwargs=protoproj_kwargs,
        )

        if unknown_mode not in {"max", "max_margin"}:
            raise ValueError("unknown_mode must be 'max' or 'max_margin'")
        self.unknown_mode = unknown_mode

        # unknown_logit = bias - scale * confidence_term
        # scale is constrained positive
        self.unknown_bias = nn.Parameter(torch.tensor(float(unknown_init_bias)))
        self.unknown_log_scale = nn.Parameter(
            torch.log(torch.tensor(float(unknown_init_scale)))
        )

        if unknown_mode == "max_margin":
            self.unknown_margin_weight = nn.Parameter(torch.tensor(1.0))

    def _compute_unknown_logit(self, logits_b_known: torch.Tensor) -> torch.Tensor:
        """
        logits_b_known: [B, K, H, W]
        returns:
            unknown_logit: [B, 1, H, W]
        """
        if logits_b_known.shape[1] == 0:
            # no supported class in the episode -> everything is unknown fg
            b, _, h, w = logits_b_known.shape
            return logits_b_known.new_full((b, 1, h, w), self.unknown_bias)

        scale = F.softplus(self.unknown_log_scale)

        top1 = logits_b_known.max(dim=1, keepdim=True).values  # [B,1,H,W]
        unknown_logit = self.unknown_bias - scale * top1

        if self.unknown_mode == "max_margin" and logits_b_known.shape[1] >= 2:
            top2 = torch.topk(logits_b_known, k=2, dim=1).values
            margin = top2[:, 0:1] - top2[:, 1:2]  # [B,1,H,W]
            unknown_logit = unknown_logit - self.unknown_margin_weight * margin

        return unknown_logit

    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        out = super().forward(x, ctx=ctx)

        if ctx is None:
            return out

        logits_b_known = out["logits_b"]  # [B,K,H,W]
        unknown_logit = self._compute_unknown_logit(logits_b_known)  # [B,1,H,W]
        logits_b_open = torch.cat([logits_b_known, unknown_logit], dim=1)

        out["logits_b_known"] = logits_b_known
        out["logits_b_unknown"] = unknown_logit
        out["logits_b"] = logits_b_open
        out["unknown_index_b"] = torch.tensor(
            logits_b_open.shape[1] - 1,
            dtype=torch.long,
            device=logits_b_open.device,
        )
        return out

    @torch.no_grad()
    def predict_segmentation(
        self,
        out: dict[str, torch.Tensor],
        fg_threshold: float = 0.5,
        unknown_label: int | None = None,
    ) -> torch.Tensor:
        """
        Full semantic prediction.

        Returns:
            pred: [B,H,W]
                0 = background
                1..K = supported pattern classes (episode-local indexing shifted by +1)
                unknown_label if provided for unsupported fg, otherwise 0
        """
        fg_prob = out["fg_prob"]  # [B,1,H,W]
        logits_b = out["logits_b"]  # [B,K+1,H,W]
        unknown_idx = int(out["unknown_index_b"].item())

        pred_local = torch.argmax(logits_b, dim=1)  # [B,H,W], in [0..K]
        pred = pred_local + 1

        is_fg = fg_prob[:, 0] >= fg_threshold
        pred[~is_fg] = 0

        reject = is_fg & (pred_local == unknown_idx)
        pred[reject] = 0 if unknown_label is None else unknown_label

        return pred

    def make_episode_pattern_targets_with_unknown(
        self,
        pattern_targets: torch.Tensor,  # [B,H,W], 0=bg, 1..C=patterns
        support_label_ids: torch.Tensor,  # [K_supported], global labels present in support
        ignore_index: int,
    ) -> torch.Tensor:
        """
        Remap full query pattern targets into episode-local targets for logits_b_open.

        Input:
            pattern_targets:
                0 = background
                1..C = global pattern labels
                ignore_index = ignore
            support_label_ids:
                global labels present in support, e.g. tensor([1, 5])

        Output:
            local_targets:
                0..K-1   -> supported classes in the episode
                K        -> unknown foreground (GT fg class absent from support)
                ignore   -> background and ignore pixels
        """
        device = pattern_targets.device
        support_label_ids = support_label_ids.to(device=device, dtype=torch.long)

        num_supported = int(support_label_ids.numel())
        unknown_idx = num_supported

        local_targets = torch.full_like(pattern_targets, fill_value=ignore_index)

        valid_fg = (pattern_targets > 0) & (pattern_targets != ignore_index)

        if num_supported == 0:
            local_targets[valid_fg] = unknown_idx
            return local_targets

        for local_idx, global_lab in enumerate(support_label_ids.tolist()):
            mask = pattern_targets == int(global_lab)
            local_targets[mask] = int(local_idx)

        supported_mask = torch.zeros_like(valid_fg, dtype=torch.bool)
        for global_lab in support_label_ids.tolist():
            supported_mask |= pattern_targets == int(global_lab)

        unknown_fg_mask = valid_fg & (~supported_mask)
        local_targets[unknown_fg_mask] = unknown_idx

        return local_targets


class TwoStagesCompartmentPrototypeDecoderV2(TwoStagesCompartmentPrototypeDecoder):
    """
    Variant of TwoStagesCompartmentPrototypeDecoder with some modifications:
    - presegmentation head takes the original feature map instead of head A logits as input
    - option to detach presegmentation head input and proto features from gradient flow
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        num_compartments: int = 16,
        proto_dim: int = 256,
        num_pattern_classes: int = 6,  # only used for the auxiliary head if use_aux_b_head=True, otherwise main head has no fixed num_classes
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        detach_preseg_input: bool = True,
        detach_proto_features: bool = True,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_compartments = int(num_compartments)
        self.num_pattern_classes = int(num_pattern_classes)
        self.center = bool(center)
        self.eps = float(eps)

        self.detach_preseg_input = bool(detach_preseg_input)
        self.detach_proto_features = bool(detach_proto_features)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)

        # Single-channel FG logit from compartment logits
        self.presegmentation_head = nn.Conv2d(
            self.embed_dim,
            1,
            kernel_size=1,
        )

        self.proto_proj = ProtoProjLite(
            self.embed_dim,
            proto_dim,
            hidden_dim=proto_dim,
        )

        self.head_b_aux = (
            nn.Conv2d(self.embed_dim, self.num_pattern_classes + 1, kernel_size=1)
            if use_aux_b_head
            else None
        )

        log_temp = torch.log(torch.tensor(float(temperature_init)))
        if learnable_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

        # --------------------------------------------------------

    # forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.forward_features_map(x)

        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        preseg_input = feat.detach() if self.detach_preseg_input else feat
        logits_preseg = self.presegmentation_head(preseg_input)  # [B,1,H,W]
        fg_prob = torch.sigmoid(logits_preseg)  # [B,1,H,W]

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
            "logits_preseg": logits_preseg,
            "fg_prob": fg_prob,
        }

        if self.head_b_aux is not None:
            out["logits_b_aux"] = self.head_b_aux(feat)

        if ctx is None:
            return out

        b, _, h, w = feat.shape
        num_fg_classes = int(ctx["num_fg_classes"].item())
        logits_b = feat.new_zeros((b, num_fg_classes, h, w))

        proto_input = feat.detach() if self.detach_proto_features else feat
        proto_feat = self.proto_proj(proto_input)

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        pair_logits = torch.einsum("bdhw,pd->bphw", proto_feat, ctx["prototypes"])
        pair_logits = pair_logits * torch.exp(self.log_temp)

        for p, lab in enumerate(ctx["label_ids"].tolist()):
            logits_b[:, int(lab) - 1] = logits_b[:, int(lab) - 1] + pair_logits[:, p]

        out["logits_b"] = logits_b
        out["label_ids_b"] = torch.unique(ctx["label_ids"], sorted=True)
        out["pair_label_ids_b"] = ctx["label_ids"]
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S,K+1,H,W], K = pattern classes only and +1 for bg
    ) -> dict[str, torch.Tensor]:
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)

        preseg_input = feat.detach() if self.detach_preseg_input else feat
        logits_preseg = self.presegmentation_head(preseg_input)
        fg_prob = torch.sigmoid(logits_preseg)  # [S,1,H,W]

        proto_feat = self.proto_proj(feat)

        s, d, h, w = proto_feat.shape
        num_fg_classes = pattern_targets.shape[1] - 1

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(dim=0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        patt_flat = pattern_targets.reshape(s, num_fg_classes + 1, h * w)  # [S,K+1,N]
        fg_flat = fg_prob.reshape(s, h * w)  # [S,N]

        prototypes_list: list[torch.Tensor] = []
        label_ids_list: list[int] = []

        for k in range(1, num_fg_classes + 1):
            weights = patt_flat[:, k] * fg_flat
            weight_sum = weights.sum()

            if weight_sum <= self.eps:
                continue

            proto_sum = torch.einsum("sn,snd->d", weights, feat_flat)
            prototype = proto_sum / weight_sum.clamp_min(self.eps)
            prototype = F.normalize(prototype, dim=0)

            prototypes_list.append(prototype)
            label_ids_list.append(int(k))

        if len(prototypes_list) == 0:
            ctx = self._empty_ctx(device=device, proto_dim=d)
            if center is not None:
                ctx["center"] = center
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes_list, dim=0),  # [P,D]
            "label_ids": torch.tensor(label_ids_list, dtype=torch.long, device=device),
            "num_fg_classes": torch.tensor(
                num_fg_classes, dtype=torch.long, device=device
            ),
        }
        if center is not None:
            ctx["center"] = center
        return ctx


class TwoStagesCompartmentPrototypeDecoderRefined(TwoStagesCompartmentPrototypeDecoder):
    """
    Two-stage prototype decoder.

    Stage 1
    -------
    Head A predicts tissue compartments.

    A presegmentation head takes Head A logits and predicts a single-channel
    foreground logit. Its sigmoid gives a soft foreground probability map
    indicating regions likely to contain pattern annotations.

    Stage 2
    -------
    Pattern prototypes are fitted only from foreground regions, using the
    foreground probability as a soft weight together with the support masks.

    Head B outputs logits only for pattern classes (no background class).
    Background handling is delegated to the presegmentation head at inference.
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        num_compartments: int = 16,
        proto_dim: int = 256,
        num_pattern_classes: int = 6,  # only used for the auxiliary head if use_aux_b_head=True, otherwise main head has no fixed num_classes
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        detach_preseg_input: bool = True,
        detach_proto_features: bool = True,
        protoproj_hidden_dim: int = 256,
        protoproj_depth: int = 2,
        protoproj_kernel_size: int = 5,
        protoproj_mlp_ratio: float = 2.0,
        protoproj_num_groups: int = 8,
        protoproj_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_compartments = int(num_compartments)
        self.num_pattern_classes = int(num_pattern_classes)
        self.center = bool(center)
        self.eps = float(eps)

        self.detach_preseg_input = bool(detach_preseg_input)
        self.detach_proto_features = bool(detach_proto_features)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)

        # Single-channel FG logit from compartment logits
        self.presegmentation_head = nn.Conv2d(
            self.num_compartments,
            1,
            kernel_size=1,
        )

        self.proto_proj = ProtoProjGN(
            self.embed_dim,
            proto_dim,
            hidden_dim=protoproj_hidden_dim,
            depth=protoproj_depth,
            kernel_size=protoproj_kernel_size,
            mlp_ratio=protoproj_mlp_ratio,
            num_groups=protoproj_num_groups,
            dropout=protoproj_dropout,
        )

        self.head_b_aux = (
            nn.Conv2d(self.embed_dim, self.num_pattern_classes + 1, kernel_size=1)
            if use_aux_b_head
            else None
        )

        log_temp = torch.log(torch.tensor(float(temperature_init)))
        if learnable_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)


class LabelPrototypeDecoder(Encoder):
    """
    Simpler prototype decoder:
      - one prototype per class from support, including bg
      - head A predicts compartments
      - prototype projection is conditioned on head A outputs
      - no compartment-specific prototype bank
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        num_compartments: int = 16,
        num_classes_b: int = 7,
        proto_dim: int = 256,
        proto_hidden_dim: int | None = None,
        ctx_dim: int = 32,
        temperature_init: float = 10.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        prototype_threshold: float = 0.0,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_compartments = int(num_compartments)
        self.num_classes_b = int(num_classes_b)
        self.center = bool(center)
        self.eps = float(eps)
        self.prototype_threshold = float(prototype_threshold)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)

        # local context extracted from head A probabilities
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(
                self.num_compartments, ctx_dim, kernel_size=3, padding=1, bias=False
            ),
            LayerNorm2d(ctx_dim),
            nn.GELU(),
            nn.Conv2d(ctx_dim, ctx_dim, kernel_size=3, padding=1, bias=True),
        )

        # prototype projection sees both image features and compartment context
        self.proto_proj = ProtoProjLite(
            in_dim=self.embed_dim + ctx_dim,
            proto_dim=proto_dim,
            hidden_dim=proto_hidden_dim if proto_hidden_dim is not None else proto_dim,
        )

        self.head_b_aux = (
            nn.Conv2d(self.embed_dim, self.num_classes_b, kernel_size=1)
            if use_aux_b_head
            else None
        )

        log_temp = torch.log(torch.tensor(float(temperature_init)))
        if learnable_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

    # --------------------------------------------------------
    # utilities
    # --------------------------------------------------------
    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))

    def _conditioned_proto_features(
        self,
        feat: torch.Tensor,  # [B,D,H,W]
        probs_a: torch.Tensor,  # [B,C,H,W]
    ) -> torch.Tensor:
        ctx_map = self.ctx_proj(probs_a)  # [B,Cctx,H,W]
        proto_in = torch.cat([feat, ctx_map], dim=1)  # [B,D+Cctx,H,W]
        proto_feat = self.proto_proj(proto_in)  # [B,Pdim,H,W]
        return proto_feat

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.forward_features_map(x)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if self.head_b_aux is not None:
            out["logits_b_aux"] = self.head_b_aux(feat)

        if ctx is None:
            return out

        b, _, h, w = feat.shape

        if ctx["prototypes"].numel() == 0:
            logits_b = feat.new_full((b, self.num_classes_b, h, w), -1e4)
            logits_b[:, 0, :, :] = 0.0
            out["logits_b"] = logits_b
            out["label_ids_b"] = ctx["label_ids"]
            return out

        proto_feat = self._conditioned_proto_features(feat.detach(), probs_a.detach())

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        # one prototype per selected class
        logits_local = torch.einsum("bdhw,kd->bkhw", proto_feat, ctx["prototypes"])
        logits_local = logits_local * torch.exp(self.log_temp)

        # expand to global class space
        logits_b = logits_local.new_full((b, self.num_classes_b, h, w), -1e4)
        logits_b[:, 0, :, :] = 0.0

        logits_b[:, ctx["label_ids"], :, :] = logits_local

        out["logits_b"] = logits_b
        out["label_ids_b"] = ctx["label_ids"]
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S,K,H,W]
    ) -> dict[str, torch.Tensor]:
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        proto_feat = self._conditioned_proto_features(feat, probs_a)

        s, d, h, w = proto_feat.shape
        k_total = pattern_targets.shape[1]

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        patt_flat = pattern_targets.reshape(s, k_total, h * w)  # [S,K,N]

        prototypes_list: list[torch.Tensor] = []
        label_ids_list: list[int] = []

        for k in range(k_total):  # includes bg
            weights = patt_flat[:, k]  # [S,N]
            weight_sum = weights.sum()

            if weight_sum <= max(self.eps, self.prototype_threshold):
                continue

            proto_sum = torch.einsum("sn,snd->d", weights, feat_flat)
            prototype = proto_sum / weight_sum.clamp_min(self.eps)
            prototype = F.normalize(prototype, dim=0)

            prototypes_list.append(prototype)
            label_ids_list.append(int(k))

        if len(prototypes_list) == 0:
            ctx = {
                "prototypes": torch.empty(0, d, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
            }
            if center is not None:
                ctx["center"] = center
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes_list, dim=0),  # [K',D]
            "label_ids": torch.tensor(label_ids_list, dtype=torch.long, device=device),
        }
        if center is not None:
            ctx["center"] = center
        return ctx

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
        device = self.pixel_mean.device
        images = images.to(device=device)
        image_labels = image_labels.to(device=device)

        self._validate_image_labels(
            images=images,
            image_labels=image_labels,
            num_classes=num_classes,
        )

        pattern_targets = self._make_full_image_pattern_targets(
            images=images,
            image_labels=image_labels,
            num_classes=num_classes,
        )

        return self.fit_prototypes(
            support_images=images,
            pattern_targets=pattern_targets,
        )

    def _make_full_image_pattern_targets(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> torch.Tensor:
        s, _, h, w = images.shape
        device = images.device
        image_labels = image_labels.to(device=device, dtype=torch.long)

        pattern_targets = torch.zeros(
            s,
            int(num_classes),
            h,
            w,
            dtype=torch.float32,
            device=device,
        )
        pattern_targets[
            torch.arange(s, device=device),
            image_labels,
        ] = 1.0
        return pattern_targets

    def _validate_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [S,C,H,W], got {tuple(images.shape)}"
            )
        if image_labels.ndim != 1:
            raise ValueError(
                f"image_labels must have shape [S], got {tuple(image_labels.shape)}"
            )
        if images.shape[0] != image_labels.shape[0]:
            raise ValueError("images and image_labels must have the same batch size")

        if image_labels.numel() > 0:
            min_label = int(image_labels.min().item())
            max_label = int(image_labels.max().item())
            if min_label < 0 or max_label >= int(num_classes):
                raise ValueError(
                    f"image_labels must lie in [0, {int(num_classes) - 1}]"
                )


class CompartmentPrototypeDecoder(Encoder):
    """
    Compartment-first prototype decoder.

    Head A:
        predicts tissue compartments.

    Head B:
        uses a compartment-wise prototype bank.
        For each selected compartment c:
            - foreground class prototypes (k, c) are fitted if overlap is high enough
            - one background prototype (0, c) is also fitted

    Background logit is composed of:
        1) background prototypes inside selected compartments
        2) bias from unselected compartments via head A probabilities
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        num_compartments: int = 16,
        num_classes_b: int = 7,
        proto_dim: int = 256,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
        allowed_compartments: list[int] | None = None,
        compartment_threshold: float = 0.2,
        bg_unselected_scale: float = 1.0,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_compartments = int(num_compartments)
        self.num_classes_b = int(num_classes_b)
        self.center = bool(center)
        self.eps = float(eps)

        self.compartment_threshold = float(compartment_threshold)
        self.bg_unselected_scale = float(bg_unselected_scale)

        if allowed_compartments is None:
            allowed_compartments = list(range(self.num_compartments))
        self.allowed_compartments = [int(c) for c in allowed_compartments]
        if len(self.allowed_compartments) == 0:
            raise ValueError("allowed_compartments must not be empty")

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)
        self.proto_proj = ProtoProjLite(
            self.embed_dim,
            proto_dim,
            hidden_dim=proto_dim,
        )

        self.head_b_aux = (
            nn.Conv2d(self.embed_dim, self.num_classes_b, kernel_size=1)
            if use_aux_b_head
            else None
        )

        log_temp = torch.log(torch.tensor(float(temperature_init)))
        if learnable_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

    # --------------------------------------------------------
    # utilities
    # --------------------------------------------------------
    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))

    def _empty_ctx(
        self, device: torch.device, proto_dim: int
    ) -> dict[str, torch.Tensor]:
        return {
            "selected_compartments": torch.empty(0, dtype=torch.long, device=device),
            "compartment_offsets": torch.zeros(1, dtype=torch.long, device=device),
            "prototypes": torch.empty(0, proto_dim, device=device),
            "label_ids": torch.empty(0, dtype=torch.long, device=device),
            "compartment_ids": torch.empty(0, dtype=torch.long, device=device),
        }

    def forward(self, x: torch.Tensor, ctx: dict[str, torch.Tensor] | None = None):
        feat = self.forward_features_map(x)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if self.head_b_aux is not None:
            out["logits_b_aux"] = self.head_b_aux(feat)

        if ctx is None:
            return out

        b, _, h, w = feat.shape
        logits_b = feat.new_full((b, self.num_classes_b, h, w), -1e4)
        logits_b[:, 0, :, :] = 0.0

        if ctx["prototypes"].numel() == 0:
            out["logits_b"] = logits_b
            out["label_ids_b"] = torch.empty(0, dtype=torch.long, device=feat.device)
            out["pair_label_ids_b"] = ctx["label_ids"]
            out["compartment_ids_b"] = ctx["compartment_ids"]
            return out

        proto_feat = self.proto_proj(feat.detach())

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        pair_logits = torch.einsum("bdhw,pd->bphw", proto_feat, ctx["prototypes"])
        pair_logits = pair_logits * torch.exp(self.log_temp)

        probs_a_for_b = probs_a.detach()
        comp_probs = probs_a_for_b[:, ctx["compartment_ids"], :, :].clamp(0.1)
        pair_logits = pair_logits * comp_probs

        for p, lab in enumerate(ctx["label_ids"].tolist()):
            logits_b[:, int(lab), :, :] = (
                logits_b[:, int(lab), :, :] + pair_logits[:, p]
            )

        # bg bias from unselected compartments
        selected_mask = torch.zeros(
            self.num_compartments,
            dtype=torch.bool,
            device=feat.device,
        )
        if ctx["selected_compartments"].numel() > 0:
            selected_mask[ctx["selected_compartments"]] = True

        all_compartments = torch.arange(self.num_compartments, device=feat.device)
        unselected_compartments = all_compartments[~selected_mask]

        if unselected_compartments.numel() > 0:
            bg_bias = probs_a_for_b[:, unselected_compartments, :, :].sum(dim=1)
            logits_b[:, 0, :, :] = (
                logits_b[:, 0, :, :] + self.bg_unselected_scale * bg_bias
            )

        out["logits_b"] = logits_b
        out["label_ids_b"] = torch.unique(ctx["label_ids"], sorted=True)
        out["pair_label_ids_b"] = ctx["label_ids"]
        out["compartment_ids_b"] = ctx["compartment_ids"]
        return out

    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S, K, H, W]
    ) -> dict[str, torch.Tensor]:
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        proto_feat = self.proto_proj(feat)

        s, d, h, w = proto_feat.shape
        k_total = pattern_targets.shape[1]

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        patt_flat = pattern_targets.reshape(s, k_total, h * w)  # [S,K,N]
        comp_flat = probs_a.reshape(s, self.num_compartments, h * w)  # [S,C,N]

        allowed_comp_ids = torch.tensor(
            self.allowed_compartments,
            device=device,
            dtype=torch.long,
        )

        prototypes_list: list[torch.Tensor] = []
        label_ids_list: list[int] = []
        compartment_ids_list: list[int] = []
        selected_compartments: list[int] = []

        for c in allowed_comp_ids.tolist():
            comp_c = comp_flat[:, int(c)]  # [S,N]
            comp_mass = comp_c.sum()

            if comp_mass <= self.eps:
                continue

            found_any_for_this_compartment = False

            for k in range(k_total):  # includes bg class 0
                patt_k = patt_flat[:, int(k)]  # [S,N]

                overlap_k = (patt_k * comp_c).sum() / comp_mass.clamp_min(self.eps)
                if overlap_k <= self.compartment_threshold:
                    continue

                weights = patt_k * comp_c  # [S,N]
                weight_sum = weights.sum()
                if weight_sum <= self.eps:
                    continue

                # weighted mean feature for class k inside compartment c
                proto_sum = torch.einsum("sn,snd->d", weights, feat_flat)
                prototype = proto_sum / weight_sum.clamp_min(self.eps)
                prototype = F.normalize(prototype, dim=0)

                prototypes_list.append(prototype)
                label_ids_list.append(int(k))
                compartment_ids_list.append(int(c))
                found_any_for_this_compartment = True

            if found_any_for_this_compartment:
                selected_compartments.append(int(c))

        if len(prototypes_list) == 0:
            ctx = {
                "prototypes": torch.empty(0, d, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
                "compartment_ids": torch.empty(0, dtype=torch.long, device=device),
                "selected_compartments": torch.empty(
                    0, dtype=torch.long, device=device
                ),
            }
            if center is not None:
                ctx["center"] = center
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes_list, dim=0),  # [P,D]
            "label_ids": torch.tensor(
                label_ids_list, dtype=torch.long, device=device
            ),  # [P]
            "compartment_ids": torch.tensor(
                compartment_ids_list,
                dtype=torch.long,
                device=device,
            ),  # [P]
            "selected_compartments": torch.tensor(
                selected_compartments,
                dtype=torch.long,
                device=device,
            ),  # [Csel]
        }
        if center is not None:
            ctx["center"] = center
        return ctx

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
        device = self.pixel_mean.device
        images = images.to(device=device)
        image_labels = image_labels.to(device=device)

        self._validate_image_labels(
            images=images,
            image_labels=image_labels,
            num_classes=num_classes,
        )

        pattern_targets = self._make_full_image_pattern_targets(
            images=images,
            image_labels=image_labels,
            num_classes=num_classes,
        )

        return self.fit_prototypes(
            support_images=images,
            pattern_targets=pattern_targets,
        )

    def _make_full_image_pattern_targets(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> torch.Tensor:
        s, _, h, w = images.shape
        device = images.device
        image_labels = image_labels.to(device=device, dtype=torch.long)

        pattern_targets = torch.zeros(
            s,
            int(num_classes),
            h,
            w,
            dtype=torch.float32,
            device=device,
        )
        pattern_targets[
            torch.arange(s, device=device),
            image_labels,
        ] = 1.0
        return pattern_targets

    def _validate_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [S,C,H,W], got {tuple(images.shape)}"
            )
        if image_labels.ndim != 1:
            raise ValueError(
                f"image_labels must have shape [S], got {tuple(image_labels.shape)}"
            )
        if images.shape[0] != image_labels.shape[0]:
            raise ValueError("images and image_labels must have the same batch size")

        if image_labels.numel() > 0:
            min_label = int(image_labels.min().item())
            max_label = int(image_labels.max().item())
            if min_label < 0 or max_label >= int(num_classes):
                raise ValueError(
                    f"image_labels must lie in [0, {int(num_classes) - 1}]"
                )


class TumorOnlyPrototypeDecoderRefined(Encoder):
    """
    Prototype decoder restricted to a subset of compartments (e.g. tumor only).

    Key ideas:
    - prototypes built only inside selected compartments
    - background class (0) is ignored for prototype fitting
    - logits are NOT gated (masking handled in loss)
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        center: bool = True,
        num_compartments: int = 16,
        num_classes_b: int = 7,
        selected_compartments: list[int],
        label_ids_by_compartment: dict[int, list[int]],
        proto_dim: int = 256,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
        use_aux_b_head: bool = False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        if not selected_compartments:
            raise ValueError("selected_compartments must not be empty")

        self.selected_compartments = [int(c) for c in selected_compartments]
        self.label_ids_by_compartment = {
            int(k): [int(vv) for vv in v] for k, v in label_ids_by_compartment.items()
        }

        self.num_compartments = int(num_compartments)
        self.num_classes_b = int(num_classes_b)
        self.center = bool(center)
        self.eps = float(eps)

        # --- heads ---
        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, 1)
        self.proto_proj = ProtoProjLite(self.embed_dim, proto_dim, hidden_dim=proto_dim)
        if use_aux_b_head:
            self.head_b_aux = nn.Conv2d(self.embed_dim, self.num_classes_b, 1)
        else:
            self.head_b_aux = None

        # temperature
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        else:
            self.register_buffer(
                "log_temp",
                torch.log(torch.tensor(temperature_init)),
            )

    # --------------------------------------------------------
    # utilities
    # --------------------------------------------------------
    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x):
        return self._tokens_to_map(super().forward(x))

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(self, x, ctx=None):
        feat = self.forward_features_map(x)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if self.head_b_aux is not None:
            logits_b_aux = self.head_b_aux(feat)
            out["logits_b_aux"] = logits_b_aux

        if ctx is None:
            return out

        proto_feat = self.proto_proj(feat.detach())
        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        logits_b = torch.einsum("bdhw,kd->bkhw", proto_feat, ctx["prototypes"])
        logits_b = logits_b * torch.exp(self.log_temp)

        out["logits_b"] = logits_b
        out["label_ids_b"] = ctx["label_ids"]
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,  # [S,K,H,W]
    ):
        device = support_images.device
        support_images = support_images.to(device)
        pattern_targets = pattern_targets.to(device).float()

        feat = self.forward_features_map(support_images)
        logits_a = self.head_a(feat)
        probs_a = F.softmax(logits_a, dim=1)

        proto_feat = self.proto_proj(feat)

        S, D, H, W = proto_feat.shape
        K = pattern_targets.shape[1]

        # resize GT if needed
        if pattern_targets.shape[-2:] != (H, W):
            pattern_targets = F.interpolate(pattern_targets, (H, W), mode="nearest")

        # ----------------------------------------
        # tumor mask from head A
        # ----------------------------------------
        comp_idx = torch.tensor(
            self.selected_compartments,
            device=device,
            dtype=torch.long,
        )
        prob_tumor = probs_a[:, comp_idx].sum(dim=1, keepdim=True)  # [S,1,H,W]

        # ----------------------------------------
        # center + normalize
        # ----------------------------------------
        if self.center:
            center = proto_feat.permute(0, 2, 3, 1).reshape(-1, D).mean(0)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        else:
            center = None

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(S, H * W, D)
        target_flat = pattern_targets.reshape(S, K, H * W)
        tumor_flat = prob_tumor.reshape(S, 1, H * W)

        # ----------------------------------------
        # remove background (class 0)
        # ----------------------------------------
        valid_classes = torch.arange(K, device=device)
        valid_classes = valid_classes[valid_classes != 0]

        target_flat = target_flat[:, valid_classes]

        # ----------------------------------------
        # weighted prototype
        # ----------------------------------------
        weights = target_flat * tumor_flat  # [S,K',N]

        proto_sum = torch.einsum("skn,snd->kd", weights, feat_flat)
        weight_sum = weights.sum(dim=(0, 2))

        valid = weight_sum > self.eps
        if not valid.any():
            return {
                "prototypes": torch.empty(0, D, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
            }

        prototypes = proto_sum / weight_sum.clamp_min(self.eps).unsqueeze(-1)
        prototypes = F.normalize(prototypes, dim=-1)

        label_ids = valid_classes[valid]

        ctx = {
            "prototypes": prototypes[valid],
            "label_ids": label_ids,
        }
        if center is not None:
            ctx["center"] = center

        return ctx

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
        """
        Build prototypes from image-level labels by assuming each image belongs
        entirely to one pattern class, while compartment localization comes from:
          - compartment_targets if provided
          - otherwise predicted head A compartments inside fit_prototypes()
        """
        device = self.pixel_mean.device
        images = images.to(device=device)
        image_labels = image_labels.to(device=device)

        self._validate_image_labels(
            images,
            image_labels,
            num_classes=num_classes,
        )

        pattern_targets = self._make_full_image_pattern_targets(
            images,
            image_labels,
            num_classes=num_classes,
        )

        return self.fit_prototypes(
            support_images=images,
            pattern_targets=pattern_targets,
        )

    def _make_full_image_pattern_targets(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Build dense per-image one-hot support masks:
            [S, K, H, W]
        where the whole image is assigned to its image-level label.
        """
        s, _, h, w = images.shape
        device = images.device
        image_labels = image_labels.to(device=device, dtype=torch.long)

        pattern_targets = torch.zeros(
            s,
            int(num_classes),
            h,
            w,
            dtype=torch.float32,
            device=device,
        )
        pattern_targets[
            torch.arange(s, device=device),
            image_labels,
        ] = 1.0
        return pattern_targets

    def _validate_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [S, C, H, W], got {tuple(images.shape)}"
            )
        if image_labels.ndim != 1:
            raise ValueError(
                f"image_labels must have shape [S], got {tuple(image_labels.shape)}"
            )
        if images.shape[0] != image_labels.shape[0]:
            raise ValueError("images and image_labels must have the same batch size")
        if image_labels.numel() > 0:
            min_label = int(image_labels.min().item())
            max_label = int(image_labels.max().item())
            if min_label < 0 or max_label >= int(num_classes):
                raise ValueError(
                    f"image_labels must lie in [0, {int(num_classes) - 1}]"
                )
