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


class ConvRefiner(nn.Module):
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
