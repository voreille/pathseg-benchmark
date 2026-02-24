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


class ConceptQuerySemanticDecoderChatGPT(Encoder):
    """
    Semantic segmentation via a bank of learnable "concept queries".

    Always returns a dict with stable keys:
      - "semantic_logits": (B, C, H, W) final semantic logits (for CE)
      - "concept_mask_logits": (B, K, H, W) final concept mask logits
      - "concept_class_logits": (B, K, C) final per-concept class logits

    Optionally (if enabled) also returns:
      - "semantic_logits_per_layer": list[(B, C, H, W)]
      - "concept_mask_logits_per_layer": list[(B, K, H, W)]
      - "concept_class_logits_per_layer": list[(B, K, C)]

    Composition (MaskFormer-style):
        P(class=c | x,y) = sum_k softmax(class_logits)[k,c] * sigmoid(mask_logits)[k,x,y]
    then semantic_logits = log(P) for CE training.
    """

    def __init__(
        self,
        img_size,
        num_classes: int,
        encoder_id,
        sub_norm: bool = False,
        num_concepts: int = 64,
        num_attn_heads: int = 8,
        num_blocks: int = 6,
        concept_dim: int = 256,
        ckpt_path: str = "",
        class_temperature: float = 1.0,
        return_per_layer: bool = False,  # deep supervision / diagnostics
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
        self.class_temperature = float(class_temperature)
        self.return_per_layer = bool(return_per_layer)

        # project encoder token dim -> concept_dim
        self.proj = nn.Linear(self.embed_dim, concept_dim)

        # 2D sine pos-emb for spatial tokens (keys)
        self.k_pos = Mask2FormerSinePositionEmbedding(
            num_pos_feats=concept_dim // 2,
            normalize=True,
        )

        # concept queries + query-slot identity embeddings
        self.q = nn.Embedding(num_concepts, concept_dim)
        self.q_pos = nn.Embedding(num_concepts, concept_dim)

        self.decoder = nn.ModuleList(
            [DecoderBlock(concept_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        self.q_norm = nn.LayerNorm(concept_dim)

        self.q_to_mask = Mask2FormerMLPPredictionHead(
            concept_dim, concept_dim, concept_dim
        )
        self.q_to_class = nn.Linear(concept_dim, num_classes)

    @staticmethod
    def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.log(x.clamp(min=eps, max=1.0 - eps))

    def enable_deep_supervision(self):
        self.return_per_layer = True

    def _concept_mask_logits(
        self, q_tokens: torch.Tensor, x_map: torch.Tensor
    ) -> torch.Tensor:
        """
        q_tokens: (K, B, C)
        x_map:    (B, C, H, W)
        returns:  (B, K, H, W)
        """
        q_feats = self.q_to_mask(self.q_norm(q_tokens)).permute(1, 0, 2)  # (B,K,C)

        B, K, C = q_feats.shape
        Bx, Cx, H, W = x_map.shape
        if B != Bx or C != Cx:
            raise RuntimeError(
                f"shape mismatch: q_feats={(B, K, C)} vs x_map={(Bx, Cx, H, W)}"
            )

        x_flat = x_map.reshape(B, C, H * W)  # (B,C,HW)
        masks_flat = torch.bmm(q_feats, x_flat)  # (B,K,HW)
        return masks_flat.view(B, K, H, W)

    def _concept_class_logits(self, q_tokens: torch.Tensor) -> torch.Tensor:
        """
        q_tokens: (K,B,C)
        returns:  (B,K,num_classes)
        """
        logits = self.q_to_class(self.q_norm(q_tokens)).transpose(0, 1)  # (B,K,C)
        if self.class_temperature != 1.0:
            logits = logits / self.class_temperature
        return logits

    def _semantic_logits(
        self, concept_mask_logits: torch.Tensor, concept_class_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        concept_mask_logits:  (B,K,H,W)
        concept_class_logits: (B,K,C)
        returns:              (B,C,H,W) semantic logits for CE
        """
        mask_probs = torch.sigmoid(concept_mask_logits)  # (B,K,H,W)
        class_probs = F.softmax(concept_class_logits, dim=-1)  # (B,K,C)
        semantic_probs = torch.einsum("bkc,bkhw->bchw", class_probs, mask_probs)
        return self._safe_log(semantic_probs)

    def forward(self, x: torch.Tensor) -> dict:
        tokens = super().forward(x)  # (B,N,embed_dim)
        tokens = self.proj(tokens)  # (B,N,concept_dim)

        B, N, C = tokens.shape

        # (K,B,C)
        q = self.q.weight[:, None, :].expand(-1, B, -1)

        # (N,B,C)
        v = tokens.transpose(0, 1)

        # (B,C,H,W)
        x_map = tokens.transpose(1, 2).reshape(B, C, *self.grid_size)

        # (N,B,C)
        k = v + self.k_pos(x_map).flatten(2).permute(2, 0, 1)

        # (K,B,C)
        q_pos = self.q_pos.weight[:, None, :].expand(-1, B, -1)

        # optional per-layer collectors
        sem_layers = []
        m_layers = []
        c_layers = []

        for block in self.decoder:
            m = self._concept_mask_logits(q, x_map)  # (B,K,H,W)
            c_logits = self._concept_class_logits(q)  # (B,K,C)
            sem = self._semantic_logits(m, c_logits)  # (B,C,H,W)

            if self.return_per_layer:
                sem_layers.append(sem)
                m_layers.append(m)
                c_layers.append(c_logits)

            q = block(q, k, v, q_pos)

        # final outputs
        concept_mask_logits = self._concept_mask_logits(q, x_map)
        concept_class_logits = self._concept_class_logits(q)
        semantic_logits = self._semantic_logits(
            concept_mask_logits, concept_class_logits
        )

        out = {
            "semantic_logits": semantic_logits,  # (B,C,H,W)
            "concept_mask_logits": concept_mask_logits,  # (B,K,H,W)
            "concept_class_logits": concept_class_logits,  # (B,K,C)
        }

        if self.return_per_layer:
            out["semantic_logits_per_layer"] = sem_layers
            out["concept_mask_logits_per_layer"] = m_layers
            out["concept_class_logits_per_layer"] = c_layers

        return out
