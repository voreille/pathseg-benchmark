from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder
from pathseg.models.refiner_layers import ProtoProjLite


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


class CompartmentPrototypeDecoder(Encoder):
    """
    Prototype decoder with one prototype per (pattern class, compartment).

    Forward returns:
      - logits_a: [B, num_compartments, H, W]
      - logits_b: [B, num_classes_b, H, W]   # global class space
      - label_ids_b: [K']                    # labels supported by current episode
      - pair_label_ids_b: [P]                # one per prototype pair
      - compartment_ids_b: [P]               # one per prototype pair

    Design:
      - class 0 is background for head B
      - prototypes are built only for foreground classes
      - background is not explicitly modeled by prototypes
      - unsupported foreground classes get a very low logit
      - background gets a neutral baseline logit of 0
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
        max_compartments_per_class: int | None = 2,
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
        self.max_compartments_per_class = max_compartments_per_class

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

    def _aggregate_pair_logits_to_local_class_logits(
        self,
        pair_logits: torch.Tensor,  # [B, P, H, W]
        pair_label_ids: torch.Tensor,  # [P]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_logits.ndim != 4:
            raise ValueError(
                f"pair_logits must have shape [B,P,H,W], got {tuple(pair_logits.shape)}"
            )
        if pair_label_ids.ndim != 1:
            raise ValueError(
                f"pair_label_ids must have shape [P], got {tuple(pair_label_ids.shape)}"
            )
        if pair_logits.shape[1] != pair_label_ids.numel():
            raise ValueError("pair_logits.shape[1] must match pair_label_ids.numel()")

        label_ids_b = torch.unique(pair_label_ids, sorted=True)
        b, _, h, w = pair_logits.shape
        logits_local = pair_logits.new_zeros((b, label_ids_b.numel(), h, w))

        for local_idx, global_label in enumerate(label_ids_b.tolist()):
            mask = pair_label_ids == int(global_label)
            logits_local[:, local_idx] = pair_logits[:, mask].sum(dim=1)

        return logits_local, label_ids_b

    def _expand_local_to_global_logits(
        self,
        logits_local: torch.Tensor,  # [B, K', H, W]
        label_ids_b: torch.Tensor,  # [K']
    ) -> torch.Tensor:
        """
        Expand episodic local logits to global class space.

        Convention:
          - background channel 0 is always available with neutral logit 0
          - unsupported foreground classes get a very low logit
        """
        b, _, h, w = logits_local.shape
        logits_global = logits_local.new_full(
            (b, self.num_classes_b, h, w),
            -1e4,
        )
        logits_global[:, 0, :, :] = 0.0

        if label_ids_b.numel() > 0:
            logits_global[:, label_ids_b, :, :] = logits_local

        return logits_global

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

        pair_label_ids = ctx["label_ids"]
        pair_compartment_ids = ctx["compartment_ids"]

        # empty episode -> only background available
        if ctx["prototypes"].numel() == 0:
            b, _, h, w = feat.shape
            logits_b = feat.new_full((b, self.num_classes_b, h, w), -1e4)
            logits_b[:, 0, :, :] = 0.0

            out["logits_b"] = logits_b
            out["label_ids_b"] = torch.empty(0, dtype=torch.long, device=feat.device)
            out["pair_label_ids_b"] = pair_label_ids
            out["compartment_ids_b"] = pair_compartment_ids
            return out

        # prototype branch should not backprop through encoder or head A
        # proto_feat = self.proto_proj(feat.detach())
        proto_feat = self.proto_proj(feat)

        if ctx.get("center") is not None:
            proto_feat = proto_feat - ctx["center"].view(1, -1, 1, 1)

        proto_feat = F.normalize(proto_feat, dim=1)

        # similarity to each retained (class, compartment) prototype
        # [B, P, H, W]
        pair_logits = torch.einsum("bdhw,pd->bphw", proto_feat, ctx["prototypes"])
        pair_logits = pair_logits * torch.exp(self.log_temp)

        # gate each pair by detached head A compartment probability
        probs_a_for_b = probs_a.detach()
        comp_probs = probs_a_for_b[:, pair_compartment_ids, :, :]
        pair_logits = pair_logits * comp_probs

        # aggregate pair logits -> episodic class logits
        logits_local, label_ids_b = self._aggregate_pair_logits_to_local_class_logits(
            pair_logits=pair_logits,
            pair_label_ids=pair_label_ids,
        )

        # episodic -> global class space
        logits_b_global = self._expand_local_to_global_logits(
            logits_local=logits_local,
            label_ids_b=label_ids_b,
        )

        out["pair_logits_b"] = pair_logits
        out["logits_b"] = logits_b_global
        out["label_ids_b"] = label_ids_b
        out["pair_label_ids_b"] = pair_label_ids
        out["compartment_ids_b"] = pair_compartment_ids
        return out

    # --------------------------------------------------------
    # prototype fitting
    # --------------------------------------------------------
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

        prototypes: list[torch.Tensor] = []
        label_ids: list[int] = []
        compartment_ids: list[int] = []

        # skip background class 0
        for k in range(1, k_total):
            patt_k = patt_flat[:, k]  # [S,N]
            patt_mass = patt_k.sum()
            if patt_mass <= self.eps:
                continue

            # how much class-k overlaps each allowed compartment
            overlap = (patt_k.unsqueeze(1) * comp_flat[:, allowed_comp_ids]).sum(
                dim=(0, 2)
            )
            overlap = overlap / patt_mass.clamp_min(self.eps)

            valid_local = torch.nonzero(
                overlap >= self.compartment_threshold,
                as_tuple=False,
            ).flatten()

            # fallback: keep best compartment
            if valid_local.numel() == 0:
                valid_local = torch.tensor(
                    [int(torch.argmax(overlap).item())],
                    device=device,
                    dtype=torch.long,
                )

            if (
                self.max_compartments_per_class is not None
                and valid_local.numel() > int(self.max_compartments_per_class)
            ):
                vals = overlap[valid_local]
                top_idx = torch.topk(
                    vals,
                    k=int(self.max_compartments_per_class),
                    largest=True,
                ).indices
                valid_local = valid_local[top_idx]

            valid_comp_ids = allowed_comp_ids[valid_local]

            for c in valid_comp_ids.tolist():
                weights = patt_k * comp_flat[:, int(c)]  # [S,N]
                weight_sum = weights.sum()
                if weight_sum <= self.eps:
                    continue

                proto_sum = torch.einsum("sn,snd->d", weights, feat_flat)
                proto = proto_sum / weight_sum.clamp_min(self.eps)
                proto = F.normalize(proto, dim=0)

                prototypes.append(proto)
                label_ids.append(int(k))
                compartment_ids.append(int(c))

        if len(prototypes) == 0:
            ctx = {
                "prototypes": torch.empty(0, d, device=device),
                "label_ids": torch.empty(0, dtype=torch.long, device=device),
                "compartment_ids": torch.empty(0, dtype=torch.long, device=device),
            }
            if center is not None:
                ctx["center"] = center
            return ctx

        ctx = {
            "prototypes": torch.stack(prototypes, dim=0),  # [P,D]
            "label_ids": torch.tensor(label_ids, dtype=torch.long, device=device),
            "compartment_ids": torch.tensor(
                compartment_ids,
                dtype=torch.long,
                device=device,
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
