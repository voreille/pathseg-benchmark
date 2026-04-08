# TODO: for export use static output or tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.encoder import Encoder


class ConditionalLinearDecoder(Encoder):
    """
        Shared ViT encoder -> two linear heads:
          - Head A (compartments): per-token classification
          - Head B (patterns): per-token classification conditioned on Head A output
    k
        Assumes Encoder.forward(x) returns image tokens only (no prefix tokens),
        shape: [B, N, D], where N = grid_h * grid_w.
    """

    def __init__(
        self,
        encoder_id: str,
        num_classes_a: int,  # Head A classes (e.g., tumor/stroma/bg/...)
        num_classes_b: int,  # Head B classes (e.g., 7 patterns incl bg, or 6 patterns)
        img_size: tuple[int, int],
        cond_dim: int = 16,  # K: conditioning feature channels derived from A
        cond_from: str = "logits",  # "probs" or "logits", apparently with stop grad better logits
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        assert cond_from in {"probs", "logits"}, "cond_from must be 'probs' or 'logits'"
        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b
        self.cond_dim = cond_dim
        self.cond_from = cond_from

        # Head A: token -> compartment logits
        self.head_a = nn.Linear(self.embed_dim, num_classes_a)

        # Adapter: A-output (CA) -> cond_dim (K)
        self.a_adapter = nn.Sequential(
            nn.Linear(num_classes_a, cond_dim),
            nn.GELU(),
        )

        # Head B: token features (D + K) -> pattern logits
        self.head_b = nn.Linear(self.embed_dim + cond_dim, num_classes_b)

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> [B, C, H, W]
        """
        # grid_size comes from Encoder (e.g., (H, W) in patch-grid space)
        gh, gw = self.grid_size
        x = x.transpose(1, 2)  # [B, C, N]
        return x.reshape(x.shape[0], -1, gh, gw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
          - logits_a: [B, CA, H, W]
          - logits_b: [B, CB, H, W]
        """
        # Encoder returns image tokens only: [B, N, D]
        tok = super().forward(x)

        # ---- Head A ----
        a_tok_logits = self.head_a(tok)  # [B, N, CA]

        # Conditioning signal from A
        # TODO: a a_feat_in = alpha * p.detach() + (1 - alpha) * p   # alpha in [0,1]
        if self.cond_from == "probs":
            a_feat_in = F.softmax(a_tok_logits, dim=-1).detach()  # [B, N, CA]
        else:
            a_feat_in = a_tok_logits.detach()  # [B, N, CA]

        a_tok_feat = self.a_adapter(a_feat_in)  # [B, N, K]

        # ---- Head B (conditioned) ----
        b_in = torch.cat([tok, a_tok_feat], dim=-1)  # [B, N, D+K]
        b_tok_logits = self.head_b(b_in)  # [B, N, CB]

        return {
            "logits_a": self._tokens_to_map(a_tok_logits),
            "logits_b": self._tokens_to_map(b_tok_logits),
        }


class TwoHeadsLinearDecoder(Encoder):
    """
        Shared ViT encoder -> two linear heads:
          - Head A (compartments): per-token classification
          - Head B (patterns): per-token classification conditioned on Head A output
    k
        Assumes Encoder.forward(x) returns image tokens only (no prefix tokens),
        shape: [B, N, D], where N = grid_h * grid_w.
    """

    def __init__(
        self,
        encoder_id: str,
        num_classes_a: int,  # Head A classes (e.g., tumor/stroma/bg/...)
        num_classes_b: int,  # Head B classes (e.g., 7 patterns incl bg, or 6 patterns)
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b

        # Head A: token -> compartment logits
        self.head_a = nn.Linear(self.embed_dim, num_classes_a)

        # Head B: token features (D + K) -> pattern logits
        self.head_b = nn.Linear(self.embed_dim, num_classes_b)

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> [B, C, H, W]
        """
        # grid_size comes from Encoder (e.g., (H, W) in patch-grid space)
        gh, gw = self.grid_size
        x = x.transpose(1, 2)  # [B, C, N]
        return x.reshape(x.shape[0], -1, gh, gw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
          - logits_a: [B, CA, H, W]
          - logits_b: [B, CB, H, W]
        """
        # Encoder returns image tokens only: [B, N, D]
        tok = super().forward(x)

        a_tok_logits = self.head_a(tok)  # [B, N, CA]
        b_tok_logits = self.head_b(tok)  # [B, N, CB]

        return {
            "logits_a": self._tokens_to_map(a_tok_logits),
            "logits_b": self._tokens_to_map(b_tok_logits),
        }


class ConvRefineBlockGN(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        num_groups: int = 8,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        activation: str = "gelu",
    ):
        super().__init__()

        if dim % num_groups != 0:
            raise ValueError("dim must be divisible by num_groups")

        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)

        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)

        return residual + x


class LogitFeatureRefiner(nn.Module):
    """
    Residual refiner:
        refined_logits = base_logits + delta(features, base_logits)
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        refine_dim: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        num_groups: int = 8,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        activation: str = "gelu",
    ):
        super().__init__()

        in_dim = feat_dim + num_classes

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_dim, refine_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=refine_dim),
            nn.GELU() if activation == "gelu" else nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ConvRefineBlockGN(
                    dim=refine_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    num_groups=num_groups,
                    dropout=dropout,
                    layer_scale_init_value=layer_scale_init_value,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_proj = nn.Conv2d(refine_dim, num_classes, kernel_size=1, bias=True)

        # Start near identity: initially delta ~= 0
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([feat, base_logits], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        delta = self.out_proj(x)
        return base_logits + delta


class TwoHeadsLinearARefinedBDecoder(Encoder):
    """
    Two-head decoder:
      - Head A: simple 1x1 conv classifier
      - Head B: base 1x1 conv classifier + residual logit/feature refiner

    Returns:
      - logits_a:      [B, CA, H, W]
      - logits_b_base: [B, CB, H, W]
      - logits_b:      [B, CB, H, W]
    """

    def __init__(
        self,
        encoder_id: str,
        num_classes_a: int,
        num_classes_b: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        refine_dim: int = 128,
        refine_num_blocks: int = 3,
        refine_kernel_size: int = 5,
        refine_mlp_ratio: float = 2.0,
        refine_num_groups: int = 8,
        refine_dropout: float = 0.0,
        refine_layer_scale_init_value: float = 1e-6,
        refine_activation: str = "gelu",
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
        )

        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b

        # Head A: token-map -> compartment logits
        self.head_a = nn.Conv2d(self.embed_dim, self.num_classes_a, kernel_size=1)

        # Head B base: token-map -> pattern logits
        self.head_b = nn.Conv2d(self.embed_dim, self.num_classes_b, kernel_size=1)

        # Head B residual refiner
        self.refiner_b = LogitFeatureRefiner(
            feat_dim=self.embed_dim,
            num_classes=self.num_classes_b,
            refine_dim=refine_dim,
            num_blocks=refine_num_blocks,
            kernel_size=refine_kernel_size,
            mlp_ratio=refine_mlp_ratio,
            num_groups=refine_num_groups,
            dropout=refine_dropout,
            layer_scale_init_value=refine_layer_scale_init_value,
            activation=refine_activation,
        )

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_map(super().forward(x))

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> [B, C, H, W]
        """
        gh, gw = self.grid_size
        x = x.transpose(1, 2)  # [B, C, N]
        return x.reshape(x.shape[0], -1, gh, gw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
          - logits_a:      [B, CA, H, W]
          - logits_b_base: [B, CB, H, W]
          - logits_b:      [B, CB, H, W]
        """
        feat = self.forward_features_map(x)  # [B, D, H, W]

        logits_a = self.head_a(feat)
        logits_b_base = self.head_b(feat)
        logits_b = self.refiner_b(feat, logits_b_base)

        return {
            "logits_a": logits_a,
            "logits_b_base": logits_b_base,
            "logits_b": logits_b,
        }
