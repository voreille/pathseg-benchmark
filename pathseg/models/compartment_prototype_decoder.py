from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder


class _PrototypeDecoderBase(Encoder):
    def __init__(
        self,
        *,
        encoder_id: str,
        img_size: tuple[int, int],
        ckpt_path: str,
        sub_norm: bool,
        discard_last_mlp: bool,
        discard_last_block: bool,
        center: bool,
        num_compartments: int,
        proto_dim: int,
        temperature_init: float,
        learnable_temp: bool,
        eps: float,
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
        self.proto_dim = int(proto_dim)
        self.center = bool(center)
        self.eps = float(eps)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)
        self.proto_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.proto_dim, kernel_size=1),
        )

        if learnable_temp:
            self.log_temp = nn.Parameter(
                torch.log(torch.tensor(float(temperature_init)))
            )
        else:
            self.register_buffer(
                "log_temp",
                torch.log(torch.tensor(float(temperature_init))),
            )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, N, C], got {tuple(x.shape)}")
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))

    def _normalize_proto_feat_with_ctx(
        self,
        proto_feat: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if ctx is not None and "center" in ctx:
            center = ctx["center"].to(device=proto_feat.device, dtype=proto_feat.dtype)
            proto_feat = proto_feat - center.view(1, -1, 1, 1)
        return F.normalize(proto_feat, dim=1)

    @staticmethod
    def _empty_ctx_unconditioned(
        device: torch.device,
        proto_dim: int,
        center_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx = {
            "prototypes": torch.empty(0, proto_dim, device=device, dtype=torch.float32),
            "label_ids": torch.empty(0, dtype=torch.long, device=device),
        }
        if center_vec is not None:
            ctx["center"] = center_vec
        return ctx

    @staticmethod
    def _empty_ctx_conditioned(
        device: torch.device,
        proto_dim: int,
        center_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx = {
            "prototypes": torch.empty(0, proto_dim, device=device, dtype=torch.float32),
            "label_ids": torch.empty(0, dtype=torch.long, device=device),
            "compartment_ids": torch.empty(0, dtype=torch.long, device=device),
        }
        if center_vec is not None:
            ctx["center"] = center_vec
        return ctx

    def _fit_features_fp32(
        self,
        support_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns fp32 tensors:
            feat_map  : [S, E, h, w]
            logits_a  : [S, Ca, h, w]
            proto_feat: [S, D, h, w]
        """
        device = support_images.device
        with torch.autocast(device_type=device.type, enabled=False):
            x = support_images.float()
            feat_map = self.forward_features_map(x)
            logits_a = self.head_a(feat_map)
            proto_feat = self.proto_proj(feat_map)
        return feat_map.float(), logits_a.float(), proto_feat.float()

    def _validate_support_tensors(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,
    ) -> None:
        if support_images.ndim != 4:
            raise ValueError(
                f"support_images must have shape [S, C, H, W], got {tuple(support_images.shape)}"
            )
        if pattern_targets.ndim != 4:
            raise ValueError(
                f"pattern_targets must have shape [S, K, H, W], got {tuple(pattern_targets.shape)}"
            )
        if support_images.shape[0] != pattern_targets.shape[0]:
            raise ValueError(
                "support_images and pattern_targets must have the same batch size"
            )

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


class CompartmentPrototypeDecoder(_PrototypeDecoderBase):
    """
    ViT encoder + dense compartment head + compartment-conditioned prototype head.

    fit_prototypes inputs:
      - support_images:   [S, C, H, W]
      - pattern_targets:  [S, K, H, W]
      - compartment_targets (optional): [S, H, W] or [S, 1, H, W]

    Returns:
        {
            "prototypes":      [M, D],
            "label_ids":       [M],
            "compartment_ids": [M],
            "center":          [D]   # optional
        }
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
        num_compartments: int,
        selected_compartments: list[int],
        proto_dim: int = 256,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
    ) -> None:
        if not selected_compartments:
            raise ValueError("selected_compartments must not be empty")

        selected_compartments = [int(c) for c in selected_compartments]
        if len(set(selected_compartments)) != len(selected_compartments):
            raise ValueError("selected_compartments must be unique")

        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            num_compartments=num_compartments,
            proto_dim=proto_dim,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
        )

        self.selected_compartments = selected_compartments
        self.register_buffer(
            "selected_compartments_tensor",
            torch.tensor(self.selected_compartments, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat_map = self.forward_features_map(x)
        logits_a = self.head_a(feat_map)
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat_map,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if ctx is None:
            return out

        proto_feat = self.proto_proj(feat_map)
        proto_feat = self._normalize_proto_feat_with_ctx(proto_feat, ctx)

        logits_b, label_ids_b = self.compute_pattern_logits(
            proto_feat=proto_feat,
            probs_a=probs_a,
            ctx=ctx,
        )

        out["proto_feat"] = proto_feat
        out["logits_b"] = logits_b
        out["label_ids_b"] = label_ids_b
        return out

    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,
        compartment_targets: torch.Tensor | None = None,
        *,
        compartment_ignore_index: int = 255,
    ) -> dict[str, torch.Tensor]:
        self._validate_support_tensors(support_images, pattern_targets)

        if compartment_targets is not None:
            if compartment_targets.ndim not in (3, 4):
                raise ValueError(
                    "compartment_targets must have shape [S,H,W] or [S,1,H,W]"
                )
            if compartment_targets.shape[0] != support_images.shape[0]:
                raise ValueError(
                    "support_images and compartment_targets must have the same batch size"
                )

        device = self.pixel_mean.device
        support_images = support_images.to(device=device)
        pattern_targets = pattern_targets.to(device=device, dtype=torch.float32)

        _, logits_a, proto_feat = self._fit_features_fp32(support_images)
        probs_a = F.softmax(logits_a, dim=1)

        s, d, h, w = proto_feat.shape
        num_labels = int(pattern_targets.shape[1])
        num_selected_comp = len(self.selected_compartments)

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        if compartment_targets is None:
            comp_weights = probs_a[:, self.selected_compartments_tensor, :, :].float()
        else:
            comp_weights = self._compartment_maps_to_weights(
                compartment_targets=compartment_targets.to(device=device),
                out_h=h,
                out_w=w,
                ignore_index=compartment_ignore_index,
            )

        center_vec = None
        if self.center:
            center_vec = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(dim=0)
            proto_feat = proto_feat - center_vec.view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        label_flat = pattern_targets.reshape(s, num_labels, h * w)  # [S,K,N]
        comp_flat = comp_weights.reshape(s, num_selected_comp, h * w)  # [S,I,N]

        weights = label_flat.unsqueeze(2) * comp_flat.unsqueeze(1)  # [S,K,I,N]

        proto_sum = torch.einsum("skin,snd->kid", weights, feat_flat)  # [K,I,D]
        weight_sum = weights.sum(dim=(0, 3))  # [K,I]

        valid = weight_sum > self.eps
        if not valid.any():
            return self._empty_ctx_conditioned(device, self.proto_dim, center_vec)

        prototypes = proto_sum / weight_sum.clamp_min(self.eps).unsqueeze(-1)
        prototypes = F.normalize(prototypes, dim=-1)

        label_ids, comp_idx = torch.nonzero(valid, as_tuple=True)

        ctx = {
            "prototypes": prototypes[label_ids, comp_idx].contiguous(),
            "label_ids": label_ids.to(dtype=torch.long, device=device),
            "compartment_ids": self.selected_compartments_tensor[comp_idx].to(
                device=device
            ),
        }
        if center_vec is not None:
            ctx["center"] = center_vec
        return ctx

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        num_classes: int,
        compartment_targets: torch.Tensor | None = None,
        compartment_ignore_index: int = 255,
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
            compartment_targets=compartment_targets,
            compartment_ignore_index=compartment_ignore_index,
        )

    def compute_pattern_logits(
        self,
        *,
        proto_feat: torch.Tensor,  # [B,D,H,W]
        probs_a: torch.Tensor,  # [B,Ca,H,W]
        ctx: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = ctx["prototypes"].to(
            device=proto_feat.device, dtype=proto_feat.dtype
        )
        label_ids = ctx["label_ids"].to(device=proto_feat.device)
        comp_ids = ctx["compartment_ids"].to(device=proto_feat.device)

        if prototypes.numel() == 0:
            b, _, h, w = proto_feat.shape
            return proto_feat.new_zeros(b, 0, h, w), label_ids.new_empty(0)

        sim = torch.einsum("bdhw,md->bmhw", proto_feat, prototypes)
        sim = sim * torch.exp(self.log_temp.to(dtype=proto_feat.dtype)).clamp_min(1e-3)

        comp_weights = probs_a[:, comp_ids, :, :]
        weighted = sim * comp_weights

        unique_label_ids = torch.unique(label_ids, sorted=True)
        logits_b = torch.cat(
            [
                weighted[:, label_ids == lid].sum(dim=1, keepdim=True)
                for lid in unique_label_ids
            ],
            dim=1,
        )

        used_compartments = torch.unique(comp_ids)
        gate = probs_a[:, used_compartments].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        logits_b = logits_b * gate

        return logits_b, unique_label_ids

    def _compartment_maps_to_weights(
        self,
        *,
        compartment_targets: torch.Tensor,
        out_h: int,
        out_w: int,
        ignore_index: int,
    ) -> torch.Tensor:
        if compartment_targets.ndim == 4:
            if compartment_targets.shape[1] != 1:
                raise ValueError(
                    "compartment_targets with 4 dims must have shape [S,1,H,W]"
                )
            compartment_targets = compartment_targets[:, 0]

        if compartment_targets.ndim != 3:
            raise ValueError("compartment_targets must have shape [S,H,W] or [S,1,H,W]")

        if compartment_targets.shape[-2:] != (out_h, out_w):
            compartment_targets = F.interpolate(
                compartment_targets.unsqueeze(1).float(),
                size=(out_h, out_w),
                mode="nearest",
            )[:, 0].long()
        else:
            compartment_targets = compartment_targets.long()

        s = compartment_targets.shape[0]
        i = len(self.selected_compartments)

        weights = torch.zeros(
            s,
            i,
            out_h,
            out_w,
            dtype=torch.float32,
            device=compartment_targets.device,
        )

        valid = compartment_targets != ignore_index
        for idx, comp_id in enumerate(self.selected_compartments):
            weights[:, idx] = ((compartment_targets == comp_id) & valid).float()

        denom = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return weights / denom


class PatternPrototypeDecoder(_PrototypeDecoderBase):
    """
    ViT encoder + dense compartment head + unconditioned prototype head.

    fit_prototypes inputs:
      - support_images:  [S, C, H, W]
      - pattern_targets: [S, K, H, W]

    Returns:
        {
            "prototypes": [L, D],
            "label_ids":  [L],
            "center":     [D]   # optional
        }
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
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            num_compartments=num_compartments,
            proto_dim=proto_dim,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
        )
        self.num_classes_b = int(num_classes_b)

    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat_map = self.forward_features_map(x)
        logits_a = self.head_a(feat_map)
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat_map,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if ctx is None:
            return out

        proto_feat = self.proto_proj(feat_map)
        proto_feat = self._normalize_proto_feat_with_ctx(proto_feat, ctx)

        logits_b, label_ids_b = self.compute_pattern_logits(
            proto_feat=proto_feat,
            ctx=ctx,
        )

        out["proto_feat"] = proto_feat
        out["logits_b"] = logits_b
        out["label_ids_b"] = label_ids_b
        return out

    @torch.no_grad()
    def fit_prototypes(
        self,
        support_images: torch.Tensor,
        pattern_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_support_tensors(support_images, pattern_targets)

        device = self.pixel_mean.device
        support_images = support_images.to(device=device)
        pattern_targets = pattern_targets.to(device=device, dtype=torch.float32)

        _, _, proto_feat = self._fit_features_fp32(support_images)

        s, d, h, w = proto_feat.shape
        num_labels = int(pattern_targets.shape[1])

        if self.num_classes_b != num_labels:
            raise ValueError(
                f"pattern_targets has {num_labels} channels but num_classes_b={self.num_classes_b}"
            )

        if pattern_targets.shape[-2:] != (h, w):
            pattern_targets = F.interpolate(
                pattern_targets,
                size=(h, w),
                mode="nearest",
            )

        center_vec = None
        if self.center:
            center_vec = proto_feat.permute(0, 2, 3, 1).reshape(-1, d).mean(dim=0)
            proto_feat = proto_feat - center_vec.view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        feat_flat = proto_feat.permute(0, 2, 3, 1).reshape(s, h * w, d)  # [S,N,D]
        target_flat = pattern_targets.reshape(s, num_labels, h * w)  # [S,K,N]

        proto_sum = torch.einsum("skn,snd->kd", target_flat, feat_flat)  # [K,D]
        weight_sum = target_flat.sum(dim=(0, 2))  # [K]

        valid = weight_sum > self.eps
        if not valid.any():
            return self._empty_ctx_unconditioned(device, self.proto_dim, center_vec)

        prototypes = proto_sum / weight_sum.clamp_min(self.eps).unsqueeze(-1)
        prototypes = F.normalize(prototypes, dim=-1)

        label_ids = torch.nonzero(valid, as_tuple=False).flatten().long()

        ctx = {
            "prototypes": prototypes[label_ids].contiguous(),
            "label_ids": label_ids.to(device=device),
        }
        if center_vec is not None:
            ctx["center"] = center_vec
        return ctx

    @torch.no_grad()
    def infer_spatial_support_from_head_a(
        self,
        images: torch.Tensor,
        *,
        selected_compartments: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Returns spatial support weights [S,1,h,w] derived from head A.

        If selected_compartments is provided:
            use the soft union of those compartments.

        Otherwise:
            use non-background support = 1 - p(bg),
            where bg is channel 0.
        """
        device = self.pixel_mean.device
        images = images.to(device=device)

        _, logits_a, _ = self._fit_features_fp32(images)
        probs_a = F.softmax(logits_a, dim=1)  # [S,Ca,h,w]

        if selected_compartments is not None:
            if len(selected_compartments) == 0:
                raise ValueError("selected_compartments must not be empty")
            comp_idx = torch.tensor(
                [int(c) for c in selected_compartments],
                dtype=torch.long,
                device=device,
            )
            return probs_a[:, comp_idx].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        return (1.0 - probs_a[:, 0:1]).clamp(0.0, 1.0)

    @torch.no_grad()
    def fit_prototypes_from_image_labels(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        *,
        selected_compartments: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Build unconditioned prototypes from image-level labels.

        Spatial support is restricted using head A:
          - union of selected_compartments if provided
          - otherwise non-background support from head A
        """
        device = self.pixel_mean.device
        images = images.to(device=device)
        image_labels = image_labels.to(device=device)

        self._validate_image_labels(
            images,
            image_labels,
            num_classes=self.num_classes_b,
        )

        full_pattern_targets = self._make_full_image_pattern_targets(
            images,
            image_labels,
            num_classes=self.num_classes_b,
        )  # [S,K,H,W]

        spatial_support = self.infer_spatial_support_from_head_a(
            images,
            selected_compartments=selected_compartments,
        )  # [S,1,h,w]

        if spatial_support.shape[-2:] != full_pattern_targets.shape[-2:]:
            spatial_support = F.interpolate(
                spatial_support,
                size=full_pattern_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pattern_targets = full_pattern_targets * spatial_support

        return self.fit_prototypes(
            support_images=images,
            pattern_targets=pattern_targets,
        )

    def compute_pattern_logits(
        self,
        *,
        proto_feat: torch.Tensor,
        ctx: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = ctx["prototypes"].to(
            device=proto_feat.device, dtype=proto_feat.dtype
        )
        label_ids = ctx["label_ids"].to(device=proto_feat.device)

        if prototypes.numel() == 0:
            b, _, h, w = proto_feat.shape
            return proto_feat.new_zeros(b, 0, h, w), label_ids.new_empty(0)

        logits_b = torch.einsum("bdhw,ld->blhw", proto_feat, prototypes)
        logits_b = logits_b * torch.exp(
            self.log_temp.to(dtype=proto_feat.dtype)
        ).clamp_min(1e-3)

        return logits_b, label_ids


class ResBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ProtoProj(nn.Module):
    def __init__(self, in_dim: int, proto_dim: int, depth: int = 2):
        super().__init__()
        self.spatial = nn.Sequential(*[ResBlock(in_dim) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim, proto_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.proj(x)
        return x


class PatternPrototypeDecoderRefined(PatternPrototypeDecoder):
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
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            num_compartments=num_compartments,
            num_classes_b=num_classes_b,
            proto_dim=proto_dim,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
        )
        self.proto_proj = ProtoProj(self.embed_dim, self.proto_dim, depth=2)


class CompartmentPrototypeDecoderRefined(CompartmentPrototypeDecoder):
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
        num_compartments: int,
        selected_compartments: list[int],
        proto_dim: int = 256,
        temperature_init: float = 20.0,
        learnable_temp: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            center=center,
            num_compartments=num_compartments,
            selected_compartments=selected_compartments,
            proto_dim=proto_dim,
            temperature_init=temperature_init,
            learnable_temp=learnable_temp,
            eps=eps,
        )
        self.proto_proj = ProtoProj(self.embed_dim, self.proto_dim, depth=2)
