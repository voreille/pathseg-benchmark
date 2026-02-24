import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMLPPredictionHead,
    Mask2FormerSinePositionEmbedding,
)

from pathseg.models.decoder_block import DecoderBlock
from pathseg.models.encoder import Encoder


class Mask2formerDecoder(Encoder):
    def __init__(
        self,
        img_size,
        num_classes,
        encoder_id,
        sub_norm=False,
        num_queries=100,
        num_attn_heads=8,
        num_blocks=9,
        embed_dim=256,
        ckpt_path="",
    ):
        super().__init__(
            img_size=img_size,
            encoder_id=encoder_id,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
        )

        self.num_attn_heads = num_attn_heads

        self.proj = nn.Linear(self.embed_dim, embed_dim)

        self.k_embed_pos = Mask2FormerSinePositionEmbedding(
            num_pos_feats=embed_dim // 2, normalize=True
        )

        self.q = nn.Embedding(num_queries, embed_dim)

        self.transformer_decoder = nn.ModuleList(
            [DecoderBlock(embed_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        self.q_pos_embed = nn.Embedding(num_queries, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)

        self.q_mlp = Mask2FormerMLPPredictionHead(embed_dim, embed_dim, embed_dim)

        self.q_class = nn.Linear(embed_dim, num_classes + 1)

    def _compute_mask_logits(
        self, q_intermediate: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes mask logits using batched matmul instead of einsum.

        Args:
            q_intermediate: (Q, B, C)
            x: (B, C, H, W)

        Returns:
            mask_logits: (B, Q, H, W)
        """
        # (Q, B, C) -> (Q, B, C)
        q_feats = self.q_mlp(q_intermediate)

        # (Q, B, C) -> (B, Q, C)
        q_feats = q_feats.permute(1, 0, 2)

        B, Q, C = q_feats.shape
        Bx, Cx, H, W = x.shape

        # sanity check
        assert B == Bx and C == Cx, "shape mismatch in compute_mask_logits"

        # flatten spatial dims: (B, C, HW)
        x_flat = x.reshape(B, C, H * W)

        # batched matmul: (B, Q, C) @ (B, C, HW) -> (B, Q, HW)
        masks_flat = torch.bmm(q_feats, x_flat)

        # reshape back to (B, Q, H, W)
        return masks_flat.view(B, Q, H, W)

    def _predict(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ):
        q_intermediate = self.q_norm(q)

        class_logits = self.q_class(q_intermediate).transpose(0, 1)

        mask_logits = self._compute_mask_logits(q_intermediate, x)

        attn_mask = (mask_logits < 0).bool().flatten(-2)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        return attn_mask, mask_logits, class_logits

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = self.proj(x)

        q = self.q.weight
        q = q[:, None, :].expand(-1, x.shape[0], -1)  # (Q, B, C) view, no copy

        v = x.transpose(0, 1)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)

        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight[:, None, :].expand(-1, x.shape[0], -1)

        mask_logits_per_layer, class_logits_per_layer = [], []

        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits = self._predict(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)

            q: torch.Tensor = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits = self._predict(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )


class ModifiedMask2formerDecoder(Encoder):
    def __init__(
        self,
        img_size,
        num_classes,
        encoder_id,
        sub_norm=False,
        num_queries=100,
        num_attn_heads=8,
        num_blocks=9,
        embed_dim=256,
        ckpt_path="",
    ):
        super().__init__(
            img_size=img_size,
            encoder_id=encoder_id,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
        )

        self.num_attn_heads = num_attn_heads

        self.proj = nn.Linear(self.embed_dim, embed_dim)

        self.k_embed_pos = Mask2FormerSinePositionEmbedding(
            num_pos_feats=embed_dim // 2, normalize=True
        )

        self.q = nn.Embedding(num_queries, embed_dim)

        self.transformer_decoder = nn.ModuleList(
            [DecoderBlock(embed_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        self.q_pos_embed = nn.Embedding(num_queries, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)

        self.q_mlp = Mask2FormerMLPPredictionHead(embed_dim, embed_dim, embed_dim)

        self.q_class = nn.Linear(embed_dim, num_classes + 1)

    def _predict(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ):
        q_intermediate = self.q_norm(q)

        class_logits = self.q_class(q_intermediate).transpose(0, 1)

        # MODFIED to output it
        mask_embeddings = self.q_mlp(q_intermediate)

        mask_logits = torch.einsum("qbc, bchw -> bqhw", mask_embeddings, x)

        attn_mask = (mask_logits < 0).bool().flatten(-2)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        return attn_mask, mask_logits, class_logits, mask_embeddings

    def forward_dict(self, x: torch.Tensor):
        x = super().forward(x)
        x = self.proj(x)

        q = self.q.weight
        q = q[:, None, :].repeat(1, x.shape[0], 1)

        v = x.transpose(0, 1)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)

        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight
        q_pos_embeds = q_pos_embeds[:, None, :].repeat(1, x.shape[0], 1)

        mask_logits_per_layer, class_logits_per_layer = [], []
        mask_embeddings_per_layer = []

        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits, mask_embeddings = self._predict(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)
            mask_embeddings_per_layer.append(mask_embeddings)

            q: torch.Tensor = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits, mask_embeddings = self._predict(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        mask_embeddings_per_layer.append(mask_embeddings)

        return {
            "mask_logits_per_layer": mask_logits_per_layer,
            "class_logits_per_layer": class_logits_per_layer,
            "mask_embeddings_per_layer": mask_embeddings_per_layer,
            "per_pixel_embeddings": x,
        }

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = self.proj(x)

        q = self.q.weight
        q = q[:, None, :].repeat(1, x.shape[0], 1)

        v = x.transpose(0, 1)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)

        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight
        q_pos_embeds = q_pos_embeds[:, None, :].repeat(1, x.shape[0], 1)

        mask_logits_per_layer, class_logits_per_layer = [], []

        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits, _ = self._predict(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)

            q: torch.Tensor = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits, _ = self._predict(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )


class ConceptQuerySemanticDecoder(Encoder):
    """
    Semantic segmentation via a bank of learnable "concept queries".

    Core idea:
      - Learn K concept queries, each produces a concept mask (B, K, H, W).
      - Each concept query also produces a semantic class distribution over C classes.
      - Compose per-pixel semantic probabilities as:
            P(class=c | x,y) = sum_k P(c | concept_k) * P(concept_k active at x,y)
        then train with pixel-wise CrossEntropy.

    This keeps concept masks available for XAI:
      - visualize concept masks
      - analyze which concepts contribute to each class and where.

    Notes:
      - This is NOT instance segmentation (no Hungarian matching).
      - Works with ignore_index through CE loss (you said you already have that mechanism).
    """

    def __init__(
        self,
        img_size,
        num_classes: int,         # semantic classes, e.g. 2 for {other, target}
        encoder_id,
        sub_norm: bool = False,
        num_concepts: int = 64,   # K concept queries
        num_attn_heads: int = 8,
        num_blocks: int = 6,
        concept_dim: int = 256,
        ckpt_path: str = "",
        # optional: make concepts "sharper" / more discrete
        class_temperature: float = 1.0,
        deep_supervision: bool = False,  # whether to return intermediate layer outputs for deep supervision
    ):
        super().__init__(
            img_size=img_size,
            encoder_id=encoder_id,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
        )

        if num_classes < 2:
            raise ValueError("num_classes should be >= 2 for semantic segmentation (e.g., {bg/other, target}).")

        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.num_attn_heads = num_attn_heads
        self.class_temperature = class_temperature
        self.deep_supervision = deep_supervision

        # project encoder token dim -> concept_dim
        self.proj = nn.Linear(self.embed_dim, concept_dim)

        # 2D sine pos-emb for spatial tokens (keys)
        self.k_pos = Mask2FormerSinePositionEmbedding(
            num_pos_feats=concept_dim // 2, normalize=True
        )

        # concept queries + their positional embeddings (query-slot identity)
        self.q = nn.Embedding(num_concepts, concept_dim)
        self.q_pos = nn.Embedding(num_concepts, concept_dim)

        self.decoder = nn.ModuleList(
            [DecoderBlock(concept_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        # normalize concept tokens (standard)
        self.q_norm = nn.LayerNorm(concept_dim)

        # concept-to-mask projection head (per concept)
        self.q_to_mask = Mask2FormerMLPPredictionHead(concept_dim, concept_dim, concept_dim)

        # concept-to-class logits (per concept)
        self.q_to_class = nn.Linear(concept_dim, num_classes)

    @staticmethod
    def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.log(x.clamp(min=eps, max=1.0 - eps))

    def _concept_masks_from_qx(self, q_tokens: torch.Tensor, x_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: (K, B, C)
            x_map:    (B, C, H, W)

        Returns:
            concept_mask_logits: (B, K, H, W)
        """
        # (K,B,C) -> (K,B,C)
        q_feats = self.q_to_mask(self.q_norm(q_tokens))
        # (K,B,C) -> (B,K,C)
        q_feats = q_feats.permute(1, 0, 2)

        B, K, C = q_feats.shape
        Bx, Cx, H, W = x_map.shape
        assert (B == Bx) and (C == Cx), "shape mismatch in _concept_masks_from_qx"

        x_flat = x_map.reshape(B, C, H * W)                 # (B,C,HW)
        masks_flat = torch.bmm(q_feats, x_flat)             # (B,K,HW)
        return masks_flat.view(B, K, H, W)                  # (B,K,H,W)

    def _concept_class_logits(self, q_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: (K,B,C)

        Returns:
            class_logits: (B,K,num_classes)
        """
        qn = self.q_norm(q_tokens)                          # (K,B,C)
        logits = self.q_to_class(qn).transpose(0, 1)        # (B,K,C)
        if self.class_temperature != 1.0:
            logits = logits / self.class_temperature
        return logits

    def compose_semantic_logits(
        self,
        concept_mask_logits: torch.Tensor,  # (B,K,H,W)
        concept_class_logits: torch.Tensor, # (B,K,C)
    ) -> torch.Tensor:
        """
        Compose per-pixel semantic logits (B,C,H,W) from:
          - concept activation maps (sigmoid of mask logits)
          - per-concept class distribution (softmax of class logits)

        P(class=c|x,y) = sum_k softmax(class_logits)[k,c] * sigmoid(mask_logits)[k,x,y]
        semantic_logits = log(P)
        """
        mask_probs = torch.sigmoid(concept_mask_logits)                     # (B,K,H,W)
        class_probs = F.softmax(concept_class_logits, dim=-1)               # (B,K,C)

        # (B,K,C) and (B,K,H,W) -> (B,C,H,W)
        semantic_probs = torch.einsum("bkc,bkhw->bchw", class_probs, mask_probs)

        return self._safe_log(semantic_probs)                               # (B,C,H,W)

    def forward(
        self,
        x: torch.Tensor,
        return_concepts: bool = True,
    ):
        """
        Returns:
          If deep_supervision:
            dict with lists per layer:
              - semantic_logits_per_layer: list[(B,C,H,W)]
              - concept_mask_logits_per_layer: list[(B,K,H,W)]  (optional)
              - concept_class_logits_per_layer: list[(B,K,C)]   (optional)
          Else:
            dict with only final tensors.
        """
        # Encoder tokens: expected (B, N, embed_dim)
        tokens = super().forward(x)
        tokens = self.proj(tokens)                        # (B, N, concept_dim)

        B, N, C = tokens.shape

        # concept queries: (K,B,C)
        q = self.q.weight[:, None, :].expand(-1, B, -1)

        # values: (N,B,C)
        v = tokens.transpose(0, 1)

        # reshape tokens into 2D grid feature map for mask projection & pos-emb
        # tokens: (B,N,C) -> (B,C,H,W) via grid_size from Encoder
        x_map = tokens.transpose(1, 2).reshape(B, C, *self.grid_size)       # (B,C,H,W)

        # keys: add 2D sine pos embedding in token space (N,B,C)
        k = v + self.k_pos(x_map).flatten(2).permute(2, 0, 1)              # (N,B,C)

        q_pos = self.q_pos.weight[:, None, :].expand(-1, B, -1)            # (K,B,C)

        sem_logits_layers = []
        concept_masks_layers = []
        concept_class_layers = []

        # decode
        for block in self.decoder:
            concept_mask_logits = self._concept_masks_from_qx(q, x_map)     # (B,K,H,W)
            concept_class_logits = self._concept_class_logits(q)            # (B,K,C)
            sem_logits = self.compose_semantic_logits(concept_mask_logits, concept_class_logits)

            sem_logits_layers.append(sem_logits)
            if return_concepts:
                concept_masks_layers.append(concept_mask_logits)
                concept_class_layers.append(concept_class_logits)

            # Optional: you can keep your attention mask trick; for semantic it’s not required.
            # If you want it, you can derive it from concept_mask_logits similarly.
            q = block(q, k, v, q_pos)

        # final prediction after last block
        concept_mask_logits = self._concept_masks_from_qx(q, x_map)
        concept_class_logits = self._concept_class_logits(q)
        sem_logits = self.compose_semantic_logits(concept_mask_logits, concept_class_logits)

        # if self.deep_supervision:
        #     sem_logits_layers.append(sem_logits)
        #     if return_concepts:
        #         concept_masks_layers.append(concept_mask_logits)
        #         concept_class_layers.append(concept_class_logits)

        #     out = {"semantic_logits_per_layer": sem_logits_layers}
        #     if return_concepts:
        #         out["concept_mask_logits_per_layer"] = concept_masks_layers
        #         out["concept_class_logits_per_layer"] = concept_class_layers
        #     return out

        # non-deep-supervised: return only final
        # out = {"semantic_logits": sem_logits}
        # if return_concepts:
        #     out["concept_mask_logits"] = concept_mask_logits
        #     out["concept_class_logits"] = concept_class_logits
        return sem_logits