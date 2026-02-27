import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMLPPredictionHead,
    Mask2FormerSinePositionEmbedding,
)

from pathseg.models.decoder_block import DecoderBlock
from pathseg.models.encoder import Encoder


class ConceptQuerySemanticDecoder(Encoder):
    """
    Semantic segmentation using a bank of learnable concept queries (MaskFormer-style).

    At each decoder layer:
      - concept_mask_logits: (B, K, H, W)
      - concept_class_logits: (B, K, C)

    Compose semantic per-pixel probabilities:
        P(c|x,y) = sum_k softmax(concept_class_logits)[b,k,c] * sigmoid(concept_mask_logits)[b,k,x,y]
    and return semantic_logits = log(P) suitable for pixel-wise CrossEntropy.

    Returns a dict with stable keys:
      - "semantic_logits": (B, C, H, W) final semantic logits
      - "concept_mask_logits": (B, K, H, W) final concept mask logits
      - "concept_class_logits": (B, K, C) final concept class logits
      - "semantic_logits_per_layer": list[(B, C, H, W)] (optional diagnostics/deep sup)
      - "concept_mask_logits_per_layer": list[(B, K, H, W)]
      - "concept_class_logits_per_layer": list[(B, K, C)]
    """

    def __init__(
        self,
        img_size,
        num_classes: int,  # semantic classes, e.g. 2 for {other, target}
        encoder_id,
        sub_norm: bool = False,
        num_concepts: int = 64,  # K concept queries
        num_attn_heads: int = 8,
        num_blocks: int = 6,
        concept_dim: int = 256,
        ckpt_path: str = "",
        return_per_layer: bool = True,  # keep per-layer outputs for analysis (or deep sup later)
    ):
        super().__init__(
            img_size=img_size,
            encoder_id=encoder_id,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
        )

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for semantic segmentation.")

        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.num_attn_heads = num_attn_heads
        self.return_per_layer = return_per_layer

        # project encoder token dim -> concept_dim
        self.token_proj = nn.Linear(self.embed_dim, concept_dim)

        # 2D sine pos embedding for spatial tokens (keys)
        self.key_pos_emb = Mask2FormerSinePositionEmbedding(
            num_pos_feats=concept_dim // 2, normalize=True
        )

        # concept queries + query-slot identity embeddings
        self.concept_queries = nn.Embedding(num_concepts, concept_dim)
        self.query_pos_emb = nn.Embedding(num_concepts, concept_dim)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(concept_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        self.query_norm = nn.LayerNorm(concept_dim)

        # query -> mask projection
        self.query_to_mask = Mask2FormerMLPPredictionHead(
            concept_dim, concept_dim, concept_dim
        )

        # query -> semantic class logits (NO +1 here; pure semantic classes)
        self.query_to_class = nn.Linear(concept_dim, num_classes)

    @staticmethod
    def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.log(x.clamp(min=eps, max=1.0 - eps))

    def _compute_concept_mask_logits(
        self,
        query_tokens: torch.Tensor,  # (K, B, C)
        feature_map: torch.Tensor,  # (B, C, H, W)
    ) -> torch.Tensor:
        """
        Returns:
            concept_mask_logits: (B, K, H, W)
        """
        # (K,B,C) -> (B,K,C)
        concept_feats = self.query_to_mask(self.query_norm(query_tokens)).permute(
            1, 0, 2
        )

        B, K, C = concept_feats.shape
        Bx, Cx, H, W = feature_map.shape
        if B != Bx or C != Cx:
            raise RuntimeError(
                f"shape mismatch: concept_feats={(B, K, C)} feature_map={(Bx, Cx, H, W)}"
            )

        # (B,C,HW)
        feat_flat = feature_map.reshape(B, C, H * W)

        # (B,K,C) @ (B,C,HW) -> (B,K,HW)
        masks_flat = torch.bmm(concept_feats, feat_flat)

        return masks_flat.view(B, K, H, W)

    def _compute_concept_class_logits(self, query_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            concept_class_logits: (B, K, num_classes)
        """
        return self.query_to_class(self.query_norm(query_tokens)).transpose(0, 1)

    def compose_semantic_logits(
        self,
        concept_mask_logits: torch.Tensor,  # (B, K, H, W)
        concept_class_logits: torch.Tensor,  # (B, K, C)
    ) -> torch.Tensor:
        """
        Returns:
            semantic_logits: (B, C, H, W)
        """
        mask_probs = torch.sigmoid(concept_mask_logits)  # (B,K,H,W)
        class_probs = F.softmax(concept_class_logits, dim=-1)  # (B,K,C)

        semantic_probs = torch.einsum("bkc,bkhw->bchw", class_probs, mask_probs)
        return self._safe_log(semantic_probs)

    def forward(self, x: torch.Tensor) -> dict:
        # encoder tokens: (B, N, Denc)
        tokens = super().forward(x)
        tokens = self.token_proj(tokens)  # (B, N, concept_dim)

        B, N, C = tokens.shape

        # concept queries: (K,B,C)
        query_tokens = self.concept_queries.weight[:, None, :].expand(-1, B, -1)

        # values for cross-attn: (N,B,C)
        value_tokens = tokens.transpose(0, 1)

        # 2D feature map for mask projection: (B,C,H,W)
        feature_map = tokens.transpose(1, 2).reshape(B, C, *self.grid_size)

        # keys for cross-attn: add 2D sine pos emb, keep token layout (N,B,C)
        key_tokens = value_tokens + self.key_pos_emb(feature_map).flatten(2).permute(
            2, 0, 1
        )

        # query position embeddings: (K,B,C)
        query_pos = self.query_pos_emb.weight[:, None, :].expand(-1, B, -1)

        # optional per-layer outputs
        semantic_logits_per_layer = []
        concept_mask_logits_per_layer = []
        concept_class_logits_per_layer = []

        for block in self.decoder_blocks:
            concept_mask_logits = self._compute_concept_mask_logits(
                query_tokens, feature_map
            )
            concept_class_logits = self._compute_concept_class_logits(query_tokens)
            semantic_logits = self.compose_semantic_logits(
                concept_mask_logits, concept_class_logits
            )

            if self.return_per_layer:
                semantic_logits_per_layer.append(semantic_logits)
                concept_mask_logits_per_layer.append(concept_mask_logits)
                concept_class_logits_per_layer.append(concept_class_logits)

            # refine queries
            query_tokens = block(query_tokens, key_tokens, value_tokens, query_pos)

        # final outputs after last refinement
        concept_mask_logits = self._compute_concept_mask_logits(
            query_tokens, feature_map
        )
        concept_class_logits = self._compute_concept_class_logits(query_tokens)
        semantic_logits = self.compose_semantic_logits(
            concept_mask_logits, concept_class_logits
        )

        out = {
            "semantic_logits": semantic_logits,  # (B,C,H,W)
            "concept_mask_logits": concept_mask_logits,  # (B,K,H,W)
            "concept_class_logits": concept_class_logits,  # (B,K,C)
        }

        if self.return_per_layer:
            out["semantic_logits_per_layer"] = semantic_logits_per_layer
            out["concept_mask_logits_per_layer"] = concept_mask_logits_per_layer
            out["concept_class_logits_per_layer"] = concept_class_logits_per_layer

        return out


class PrimitiveBankSemanticDecoder(Encoder):
    """
    Option 1: Pixel-driven mixture of global primitive prototypes + dataset-specific mapping.

    - Encoder is your ViT Encoder superclass that returns tokens: (B, N, Denc)
    - We project tokens -> per-pixel embeddings (B, D, H', W') using the same grid_size logic.
    - A global bank of K primitive prototypes P_k in embedding space (K, D).
    - Per-pixel assignment weights over primitives:
        a[b,k,h,w] = softmax_k( <e[b,:,h,w], P_k> / tau )
    - Dataset-specific class logits:
        logits[b,c,h,w] = sum_k W_d[c,k] * a[b,k,h,w] + b_d[c]

    Outputs:
      - "semantic_logits": (B, C_d, H', W') for the chosen dataset head
      - "primitive_logits": (B, K, H', W') similarity scores before softmax (useful for XAI)
      - "primitive_probs": (B, K, H', W') assignment weights a
      - "pixel_embeddings": (B, D, H', W')
    """

    def __init__(
        self,
        img_size,
        encoder_id,
        num_classes: int,
        ckpt_path: str = "",
        sub_norm: bool = False,
        primitive_dim: int = 256,  # D
        num_primitives: int = 128,  # K
        temperature: float = 1.0,  # tau
        normalize_embeddings: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            encoder_id=encoder_id,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
        )

        self.num_primitives = num_primitives
        self.primitive_dim = primitive_dim
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)

        # project encoder token dim -> primitive_dim (pixel embedding dim)
        self.token_proj = nn.Linear(self.embed_dim, primitive_dim)

        # global primitive prototypes P: (K, D)
        self.primitive_prototypes = nn.Parameter(
            torch.empty(num_primitives, primitive_dim)
        )
        nn.init.trunc_normal_(self.primitive_prototypes, std=0.02)

        # dataset-specific primitive->class mappings
        # W_d: (C_d, K), b_d: (C_d,)

        self.head = nn.Linear(num_primitives, num_classes, bias=True)

        # optional: encourage sparse class usage by initializing small weights
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D)
        returns feature_map: (B, D, H', W') with H'*W' == N (as in your existing grid_size logic)
        """
        B, N, D = tokens.shape
        H, W = self.grid_size  # inherited from Encoder
        if H * W != N:
            raise RuntimeError(
                f"Token count N={N} does not match grid_size {self.grid_size} (H*W={H * W})."
            )
        # (B, N, D) -> (B, D, H, W)
        return tokens.transpose(1, 2).reshape(B, D, H, W)

    def _compute_primitive_logits(self, pixel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        pixel_embeddings: (B, D, H, W)
        returns primitive_logits: (B, K, H, W) where logits are dot(e, Pk)/tau
        """
        B, D, H, W = pixel_embeddings.shape
        P = self.primitive_prototypes  # (K, D)

        e = pixel_embeddings
        if self.normalize_embeddings:
            e = F.normalize(e, dim=1)
            Pn = F.normalize(P, dim=1)
        else:
            Pn = P

        # Compute dot products: (B, D, H, W) with (K, D)
        # Use einsum: bdhw,kd -> bkhw
        primitive_logits = torch.einsum("bdhw,kd->bkhw", e, Pn) / max(
            self.temperature, 1e-6
        )
        return primitive_logits

    def forward(self, x: torch.Tensor) -> dict:
        """
        dataset: key from datasets dict passed at init, selects which class head to use.
        """

        # ViT encoder tokens: (B, N, Denc)
        tokens = super().forward(x)

        # Project to primitive embedding dim: (B, N, D)
        tokens = self.token_proj(tokens)

        # Per-pixel embedding map: (B, D, H', W')
        pixel_embeddings = self._tokens_to_feature_map(tokens)

        # Primitive similarity logits: (B, K, H', W')
        primitive_logits = self._compute_primitive_logits(pixel_embeddings)

        # Primitive assignment weights: (B, K, H', W')
        primitive_probs = F.softmax(primitive_logits, dim=1)

        # Dataset-specific class logits from primitive mixture:
        # For each pixel, we have a K-dim vector primitive_probs[:, :, h, w]
        # Apply linear head over K: head takes (B, K, H, W) -> (B, C, H, W)
        head = self.head
        # reshape to apply nn.Linear over K:
        B, K, H, W = primitive_logits.shape
        logits_flat = primitive_logits.permute(0, 2, 3, 1).reshape(
            B * H * W, K
        )  # (BHW, K)
        logits_flat = head(logits_flat)  # (BHW, C)
        C = logits_flat.shape[-1]
        semantic_logits = logits_flat.view(B, H, W, C).permute(
            0, 3, 1, 2
        )  # (B, C, H, W)

        return {
            "semantic_logits": semantic_logits,  # (B, C_d, H', W')
            "primitive_logits": primitive_logits,  # (B, K, H', W')
            "primitive_probs": primitive_probs,  # (B, K, H', W')
            "pixel_embeddings": pixel_embeddings,  # (B, D, H', W')
        }
