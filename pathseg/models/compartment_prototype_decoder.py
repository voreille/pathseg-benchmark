# TODO: check the shape of the prototypes
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder


PatternTarget = Union[int, torch.Tensor]


class CompartmentPrototypeDecoder(Encoder):
    """
    ViT encoder + dense compartment head + compartment-conditioned prototype head.

    Pattern target API for fit_prototypes:
      - int:
          whole image supports one label
      - Tensor[K, H, W]:
          per-label support weights, where channel index = label id

    Optional compartment_targets API:
      - list[Tensor[H, W]]:
          dense compartment label maps
      - if None:
          use predicted compartments from head A

    Returned ctx format:
        {
            "prototypes":      Tensor[M, D],
            "label_ids":       Tensor[M],
            "compartment_ids": Tensor[M],
            "center":          Tensor[D],   # only if center=True
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

        self.num_compartments = int(num_compartments)
        self.selected_compartments = [int(c) for c in selected_compartments]
        self.proto_dim = int(proto_dim)
        self.center = bool(center)
        self.eps = float(eps)

        self.head_a = nn.Conv2d(self.embed_dim, self.num_compartments, kernel_size=1)
        self.proto_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.proto_dim, kernel_size=1),
        )

        self.register_buffer(
            "selected_compartments_tensor",
            torch.tensor(self.selected_compartments, dtype=torch.long),
            persistent=False,
        )
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature_init)))

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.grid_size
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, C], got {tuple(x.shape)}")
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))  # [B, E, H, W]

    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat_map = self.forward_features_map(x)  # [B,E,H,W]
        logits_a = self.head_a(feat_map)  # [B,Ca,H,W]
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat_map,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if ctx is None:
            return out

        proto_feat = self.proto_proj(feat_map)  # [B,D,H,W]
        if ctx is not None and "center" in ctx:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)
        proto_feat = F.normalize(proto_feat, dim=1)

        logits_b, label_ids_b = self.compute_pattern_logits(
            proto_feat=proto_feat,
            probs_a=probs_a,
            ctx=ctx,
        )
        out["logits_b"] = logits_b
        out["label_ids_b"] = label_ids_b
        out["proto_feat"] = proto_feat

        return out

    @torch.no_grad()
    def fit_prototypes(
        self,
        images: list[torch.Tensor],
        pattern_targets: list[PatternTarget],
        compartment_targets: list[torch.Tensor] | None = None,
        *,
        compartment_ignore_index: int = 255,
    ) -> dict[str, torch.Tensor]:
        if not images:
            raise ValueError("images must not be empty")
        if len(images) != len(pattern_targets):
            raise ValueError("images and pattern_targets must have same length")
        if compartment_targets is not None and len(images) != len(compartment_targets):
            raise ValueError("images and compartment_targets must have same length")

        device = self.pixel_mean.device

        support_feats: list[torch.Tensor] = []
        support_label_ids: list[torch.Tensor] = []
        support_label_weights: list[torch.Tensor] = []
        support_comp_weights: list[torch.Tensor] = []

        for i, image in enumerate(images):
            image = image.to(device).unsqueeze(0)  # [1,C,H,W]

            feat_map = self.forward_features_map(image)  # [1,E,h,w]
            logits_a = self.head_a(feat_map)  # [1,Ca,h,w]
            probs_a = F.softmax(logits_a, dim=1)  # [1,Ca,h,w]
            proto_feat = self.proto_proj(feat_map)[0]  # [D,h,w]

            h, w = proto_feat.shape[-2:]

            label_ids, label_weights = self._pattern_target_to_weights(
                pattern_targets[i],
                out_h=h,
                out_w=w,
                device=device,
            )  # label_ids:[L], label_weights:[L,h,w]

            if compartment_targets is None:
                comp_weights = probs_a[0, self.selected_compartments_tensor]  # [I,h,w]
            else:
                comp_weights = self._compartment_map_to_weights(
                    compartment_targets[i].to(device),
                    out_h=h,
                    out_w=w,
                    ignore_index=compartment_ignore_index,
                )  # [I,h,w]

            support_feats.append(proto_feat)
            support_label_ids.append(label_ids)
            support_label_weights.append(label_weights)
            support_comp_weights.append(comp_weights)

        center_vec = None
        if self.center:
            all_flat = torch.cat(
                [
                    feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
                    for feat in support_feats
                ],
                dim=0,
            )
            center_vec = all_flat.mean(dim=0)

        proto_vectors = []
        proto_label_ids = []
        proto_compartment_ids = []

        for feat, label_ids, label_weights, comp_weights in zip(
            support_feats,
            support_label_ids,
            support_label_weights,
            support_comp_weights,
        ):
            if center_vec is not None:
                feat = feat - center_vec.view(-1, 1, 1)
            feat = F.normalize(feat, dim=0)

            feat_flat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])  # [N,D]
            label_flat = label_weights.reshape(label_weights.shape[0], -1)  # [L,N]
            comp_flat = comp_weights.reshape(comp_weights.shape[0], -1)  # [I,N]

            for li, label_id in enumerate(label_ids.tolist()):
                lw = label_flat[li]  # [N]

                for ci, comp_id in enumerate(self.selected_compartments):
                    cw = comp_flat[ci]  # [N]
                    weights = lw * cw
                    weight_sum = weights.sum()

                    if weight_sum <= self.eps:
                        continue

                    proto = (weights.unsqueeze(1) * feat_flat).sum(dim=0) / weight_sum
                    proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)

                    proto_vectors.append(proto)
                    proto_label_ids.append(int(label_id))
                    proto_compartment_ids.append(int(comp_id))

        if proto_vectors:
            ctx = {
                "prototypes": torch.stack(proto_vectors, dim=0),
                "label_ids": torch.tensor(
                    proto_label_ids, dtype=torch.long, device=device
                ),
                "compartment_ids": torch.tensor(
                    proto_compartment_ids, dtype=torch.long, device=device
                ),
            }
        else:
            ctx = {
                "prototypes": torch.empty(0, self.proto_dim, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
                "compartment_ids": torch.empty(0, dtype=torch.long, device=device),
            }

        if center_vec is not None:
            ctx["center"] = center_vec

        return ctx

    def compute_pattern_logits(
        self,
        *,
        proto_feat: torch.Tensor,  # [B,D,H,W]
        probs_a: torch.Tensor,  # [B,Ca,H,W]
        ctx: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = ctx["prototypes"]  # [M,D]
        label_ids = ctx["label_ids"]  # [M]
        comp_ids = ctx["compartment_ids"]  # [M]

        if prototypes.numel() == 0:
            b, _, h, w = proto_feat.shape
            return proto_feat.new_zeros(b, 0, h, w), label_ids.new_empty(0)

        sim = torch.einsum("bdhw,md->bmhw", proto_feat, prototypes)
        sim = sim * torch.exp(self.log_temp).clamp_min(1e-3)

        comp_weights = probs_a[:, comp_ids, :, :]  # [B,M,H,W]
        weighted = sim * comp_weights

        used_compartments = torch.unique(comp_ids)
        gate = probs_a[:, used_compartments].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        unique_label_ids = torch.unique(label_ids, sorted=True)
        logits_b = torch.cat(
            [
                weighted[:, label_ids == lid].sum(dim=1, keepdim=True)
                for lid in unique_label_ids
            ],
            dim=1,
        )
        logits_b = logits_b * gate

        return logits_b, unique_label_ids

    def _pattern_target_to_weights(
        self,
        target: PatternTarget,
        *,
        out_h: int,
        out_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          label_ids: [L]
          label_weights: [L, H, W]

        Supported target formats:
          - int
          - Tensor[K, H, W]
        """
        if isinstance(target, int):
            label_ids = torch.tensor([int(target)], dtype=torch.long, device=device)
            label_weights = torch.ones(
                1, out_h, out_w, dtype=torch.float32, device=device
            )
            return label_ids, label_weights

        if not isinstance(target, torch.Tensor):
            raise TypeError("pattern target must be int or Tensor[K,H,W]")

        target = target.to(device=device, dtype=torch.float32)
        if target.ndim != 3:
            raise ValueError("pattern target tensor must have shape [K, H, W]")

        if target.shape[-2:] != (out_h, out_w):
            target = F.interpolate(
                target.unsqueeze(0),
                size=(out_h, out_w),
                mode="nearest",
            ).squeeze(0)

        label_mass = target.flatten(1).sum(dim=1)
        keep = label_mass > 0

        label_ids = torch.nonzero(keep, as_tuple=False).flatten().to(dtype=torch.long)
        label_weights = target[keep]

        return label_ids, label_weights

    def _compartment_map_to_weights(
        self,
        target: torch.Tensor,
        *,
        out_h: int,
        out_w: int,
        ignore_index: int,
    ) -> torch.Tensor:
        """
        target:
            dense compartment map [H, W]

        returns:
            weights [I, H, W] over selected compartments
        """
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]
        if target.ndim != 2:
            raise ValueError("compartment target must have shape [H,W] or [1,H,W]")

        if target.shape != (out_h, out_w):
            target = F.interpolate(
                target.unsqueeze(0).unsqueeze(0).float(),
                size=(out_h, out_w),
                mode="nearest",
            )[0, 0].long()

        weights = torch.zeros(
            len(self.selected_compartments),
            out_h,
            out_w,
            dtype=torch.float32,
            device=target.device,
        )

        valid = target != ignore_index
        for i, comp_id in enumerate(self.selected_compartments):
            weights[i] = ((target == comp_id) & valid).float()

        denom = weights.sum(dim=0, keepdim=True).clamp_min(self.eps)
        weights = weights / denom
        return weights


class PatternPrototypeDecoder(Encoder):
    """
    ViT encoder + dense compartment head + prototype head without compartment conditioning.

    Head A:
      - standard dense compartment segmentation

    Head B:
      - one prototype per pattern label
      - no compartment weighting, no gating

    fit_prototypes API
    ------------------
    pattern_targets must be a list of tensors [K, H, W], where:
      - K = number of labels (including bg if desired)
      - each channel is a support weight map for that label

    Returned ctx format
    -------------------
        {
            "prototypes": Tensor[L, D],
            "label_ids":  Tensor[L],
            "center":     Tensor[D],   # only if center=True
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
        )

        self.num_compartments = int(num_compartments)
        self.num_classes_b = int(num_classes_b)
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
            raise ValueError(f"x must be [B, N, C], got {tuple(x.shape)}")
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))  # [B, E, H, W]

    def forward(
        self,
        x: torch.Tensor,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feat_map = self.forward_features_map(x)  # [B,E,H,W]
        logits_a = self.head_a(feat_map)  # [B,Ca,H,W]
        probs_a = F.softmax(logits_a, dim=1)

        out = {
            "feat_map": feat_map,
            "logits_a": logits_a,
            "probs_a": probs_a,
        }

        if ctx is None:
            return out

        proto_feat = self.proto_proj(feat_map)  # [B,D,H,W]
        if "center" in ctx:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)
        proto_feat = F.normalize(proto_feat, dim=1)

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
        images: list[torch.Tensor],
        pattern_targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images:
            list of [C,H,W]
        pattern_targets:
            list of [K,H,W] support weight maps
        """
        if not images:
            raise ValueError("images must not be empty")
        if len(images) != len(pattern_targets):
            raise ValueError("images and pattern_targets must have same length")

        device = self.pixel_mean.device

        support_feats: list[torch.Tensor] = []
        support_label_ids: list[torch.Tensor] = []
        support_label_weights: list[torch.Tensor] = []

        for image, target in zip(images, pattern_targets):
            image = image.to(device).unsqueeze(0)  # [1,C,H,W]
            feat_map = self.forward_features_map(image)  # [1,E,h,w]
            proto_feat = self.proto_proj(feat_map)[0]  # [D,h,w]

            h, w = proto_feat.shape[-2:]
            label_ids, label_weights = self._pattern_target_to_weights(
                target,
                out_h=h,
                out_w=w,
                device=device,
            )  # label_ids:[L], label_weights:[L,h,w]

            support_feats.append(proto_feat)
            support_label_ids.append(label_ids)
            support_label_weights.append(label_weights)

        center_vec = None
        if self.center:
            all_flat = torch.cat(
                [
                    feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
                    for feat in support_feats
                ],
                dim=0,
            )
            center_vec = all_flat.mean(dim=0)

        proto_vectors = []
        proto_label_ids = []

        for feat, label_ids, label_weights in zip(
            support_feats,
            support_label_ids,
            support_label_weights,
        ):
            if center_vec is not None:
                feat = feat - center_vec.view(-1, 1, 1)
            feat = F.normalize(feat, dim=0)

            feat_flat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])  # [N,D]
            label_flat = label_weights.reshape(label_weights.shape[0], -1)  # [L,N]

            for li, label_id in enumerate(label_ids.tolist()):
                weights = label_flat[li]  # [N]
                weight_sum = weights.sum()

                if weight_sum <= self.eps:
                    continue

                proto = (weights.unsqueeze(1) * feat_flat).sum(dim=0) / weight_sum
                proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)

                proto_vectors.append(proto)
                proto_label_ids.append(int(label_id))

        if proto_vectors:
            ctx = {
                "prototypes": torch.stack(proto_vectors, dim=0),  # [L,D]
                "label_ids": torch.tensor(
                    proto_label_ids, dtype=torch.long, device=device
                ),
            }
        else:
            ctx = {
                "prototypes": torch.empty(0, self.proto_dim, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
            }

        if center_vec is not None:
            ctx["center"] = center_vec

        return ctx

    def compute_pattern_logits(
        self,
        *,
        proto_feat: torch.Tensor,  # [B,D,H,W]
        ctx: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = ctx["prototypes"]  # [L,D]
        label_ids = ctx["label_ids"]  # [L]

        if prototypes.numel() == 0:
            b, _, h, w = proto_feat.shape
            return proto_feat.new_zeros(b, 0, h, w), label_ids.new_empty(0)

        logits_b = torch.einsum("bdhw,ld->blhw", proto_feat, prototypes)
        logits_b = logits_b * torch.exp(self.log_temp).clamp_min(1e-3)

        return logits_b, label_ids

    def _pattern_target_to_weights(
        self,
        target: torch.Tensor,
        *,
        out_h: int,
        out_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        target:
            Tensor[K,H,W] support weight maps

        Returns
        -------
        label_ids:
            [L]
        label_weights:
            [L,H,W]
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError("pattern target must be Tensor[K,H,W]")

        target = target.to(device=device, dtype=torch.float32)
        if target.ndim != 3:
            raise ValueError("pattern target tensor must have shape [K,H,W]")

        if target.shape[-2:] != (out_h, out_w):
            target = F.interpolate(
                target.unsqueeze(0),
                size=(out_h, out_w),
                mode="nearest",
            ).squeeze(0)

        label_mass = target.flatten(1).sum(dim=1)
        keep = label_mass > 0

        label_ids = torch.nonzero(keep, as_tuple=False).flatten().to(dtype=torch.long)
        label_weights = target[keep]

        return label_ids, label_weights
