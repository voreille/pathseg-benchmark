import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg.models.histo_encoder import Encoder


class AttnPool(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1),
        )

    def forward(
        self, H: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # H: (B, S, H)
        logits = self.score(H).squeeze(-1)  # (B, S)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        a = torch.softmax(logits, dim=1)  # (B, S)
        return (H * a.unsqueeze(-1)).sum(dim=1)  # (B, H)

class MetaLinearHeadBinary(nn.Module):
    def __init__(self, embed_dim: int, h_dim: int = 256, use_gate: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gate = use_gate

        self.phi = nn.Sequential(
            nn.Linear(embed_dim + 1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.pool = AttnPool(h_dim)

        out_dim = embed_dim + 1 + (embed_dim if use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

        self.w0 = nn.Parameter(torch.zeros(embed_dim))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, Xs: torch.Tensor, ys: torch.Tensor, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if ys.dim() == 2:
            ys = ys.unsqueeze(-1)  # (B,S,1)

        H = self.phi(torch.cat([Xs, ys], dim=-1))  # (B,S,H)
        r = self.pool(H, mask=mask)                # (B,H)
        params = self.psi(r)                       # (B, out_dim)

        if self.use_gate:
            dw = params[:, : self.embed_dim]
            db = params[:, self.embed_dim : self.embed_dim + 1]
            gate_logits = params[:, self.embed_dim + 1 :]
            gate = torch.sigmoid(gate_logits)
            w = self.w0.unsqueeze(0) + gate * dw
        else:
            w = self.w0.unsqueeze(0) + params[:, : self.embed_dim]
            db = params[:, self.embed_dim : self.embed_dim + 1]

        b = self.b0.unsqueeze(0) + db
        return w, b

    def predict_query(self, Xq: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if Xq.dim() == 2:
            Xq = Xq.unsqueeze(0)  # (1,Q,E)

        logits = torch.einsum("bqe,be->bq", Xq, w) + b.squeeze(-1).unsqueeze(1)
        return logits


class MetaLinearHeadMC(nn.Module):
    """
    Multiclass meta linear head for a single episode.

    Inputs:
      Xs: (S, E)
      ys: (S, C) one-hot or soft labels (rows sum to 1 is typical, but not required)

    Outputs:
      W: (C, E)
      b: (C,)
    """

    def __init__(
        self, embed_dim: int, h_dim: int = 256, use_gate: bool = True, eps: float = 1e-6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gate = use_gate
        self.eps = eps

        # Encode support features (optionally you can also concatenate ys, but here we do class-conditioning via pooling)
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # Map class representation -> class weights/bias (and optional gate)
        out_dim = embed_dim + 1 + (embed_dim if use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

        # Optional global anchor shared across classes
        self.W0 = nn.Parameter(torch.zeros(embed_dim))  # acts like a prior direction
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(
        self, Xs: torch.Tensor, ys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Xs: (S, E)
        ys: (S, C)
        returns:
          W: (C, E)
          b: (C,)
        """
        assert Xs.dim() == 2, "Xs must be (S,E)"
        assert ys.dim() == 2, "ys must be (S,C)"
        S, E = Xs.shape
        S2, C = ys.shape
        assert S == S2, "Xs and ys must share S"
        assert E == self.embed_dim, "embed_dim mismatch"

        H = self.phi(Xs)  # (S, h_dim)

        # Weighted mean pooling per class:
        # r_c = sum_i y_{i,c} * H_i / (sum_i y_{i,c} + eps)
        weights = ys  # (S, C)
        denom = weights.sum(dim=0, keepdim=True).clamp_min(self.eps)  # (1, C)
        r = (H.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=0) / denom.transpose(
            0, 1
        )  # (C, h_dim)

        params = self.psi(r)  # (C, out_dim)

        if self.use_gate:
            dW = params[:, :E]  # (C,E)
            db = params[:, E : E + 1].squeeze(-1)  # (C,)
            gate_logits = params[:, E + 1 :]  # (C,E)
            gate = torch.sigmoid(gate_logits)  # (C,E)
            W = self.W0.unsqueeze(0) + gate * dW
        else:
            W = self.W0.unsqueeze(0) + params[:, :E]
            db = params[:, E : E + 1].squeeze(-1)

        b = self.b0.squeeze(0) + db  # (C,)
        return W, b

    @staticmethod
    def predict_query(
        Xq: torch.Tensor, W: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Xq: (Q,E)
        W:  (C,E)
        b:  (C,)
        returns logits: (Q,C)
        """
        return Xq @ W.t() + b.unsqueeze(0)


class MetaLinearHeadML(nn.Module):
    """
    Multi-label meta linear head (C independent sigmoids).

    Inputs:
      Xs: (S, E)
      ys: (S, C)   (soft or hard multi-label targets in [0,1])
    Outputs:
      W: (C, E)
      b: (C,)
    """

    def __init__(
        self, embed_dim: int, h_dim: int = 256, use_gate: bool = True, eps: float = 1e-6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gate = use_gate
        self.eps = eps

        self.phi = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # we feed [r_pos, r_neg] (2*h_dim) -> head params
        out_dim = embed_dim + 1 + (embed_dim if use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

        # optional global anchor
        self.W0 = nn.Parameter(torch.zeros(embed_dim))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(
        self, Xs: torch.Tensor, ys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert Xs.dim() == 2, "Xs must be (S,E)"
        assert ys.dim() == 2, "ys must be (S,C)"
        S, E = Xs.shape
        S2, C = ys.shape
        assert S == S2 and E == self.embed_dim

        H = self.phi(Xs)  # (S, h_dim)

        w_pos = ys  # (S,C)
        w_neg = 1.0 - ys  # (S,C)

        denom_pos = w_pos.sum(dim=0).clamp_min(self.eps)  # (C,)
        denom_neg = w_neg.sum(dim=0).clamp_min(self.eps)  # (C,)

        r_pos = (H.unsqueeze(1) * w_pos.unsqueeze(-1)).sum(dim=0) / denom_pos.unsqueeze(
            -1
        )  # (C,h)
        r_neg = (H.unsqueeze(1) * w_neg.unsqueeze(-1)).sum(dim=0) / denom_neg.unsqueeze(
            -1
        )  # (C,h)

        r = torch.cat([r_pos, r_neg], dim=-1)  # (C, 2h)

        params = self.psi(r)  # (C, out_dim)

        if self.use_gate:
            dW = params[:, :E]
            db = params[:, E : E + 1].squeeze(-1)  # (C,)
            gate = torch.sigmoid(params[:, E + 1 :])  # (C,E)
            W = self.W0.unsqueeze(0) + gate * dW
        else:
            W = self.W0.unsqueeze(0) + params[:, :E]
            db = params[:, E : E + 1].squeeze(-1)

        b = self.b0.squeeze(0) + db
        return W, b

    @staticmethod
    def predict_query(
        Xq: torch.Tensor, W: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        # Xq: (Q,E), W: (C,E), b: (C,) -> logits: (Q,C)
        return Xq @ W.t() + b.unsqueeze(0)


class ClassCrossAttnPool(nn.Module):
    """
    Cross-attention pooling:
      queries Q: (C, H)
      keys/values Hs: (S, H)
    returns:
      R: (C, H)
    """

    def __init__(self, h_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        queries: torch.Tensor,  # (C, H)
        support: torch.Tensor,  # (S, H)
        mask_s: torch.Tensor | None = None,  # (S,) True for valid
    ) -> torch.Tensor:
        Q = queries.unsqueeze(0)  # (1, C, H)
        K = support.unsqueeze(0)  # (1, S, H)
        V = support.unsqueeze(0)  # (1, S, H)

        key_padding_mask = None
        if mask_s is not None:
            # nn.MultiheadAttention expects True for positions to IGNORE
            key_padding_mask = ~mask_s.unsqueeze(0)  # (1, S)

        out, _ = self.mha(
            Q, K, V, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out.squeeze(0)  # (C, H)


class MetaLinearHeadMC_Attn(nn.Module):
    """
    Multiclass meta linear head with attention pooling (single episode).

    Inputs:
      Xs: (S, E)
      ys: (S, C)  one-hot or soft labels in [0,1]
      mask_s: (S,) optional boolean mask (True valid)

    Outputs:
      W: (C, E)
      b: (C,)
    """
    def __init__(
        self,
        embed_dim: int,
        h_dim: int = 256,
        num_heads: int = 4,
        use_gate: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.use_gate = use_gate
        self.eps = eps

        self.phi = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.attn_pool = ClassCrossAttnPool(h_dim=h_dim, num_heads=num_heads)

        out_dim = embed_dim + 1 + (embed_dim if use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

        # optional global anchor / prior
        self.W0 = nn.Parameter(torch.zeros(embed_dim))
        self.b0 = nn.Parameter(torch.zeros(1))

    def _masked_weighted_mean(self, Hs: torch.Tensor, ys: torch.Tensor, mask_s: torch.Tensor | None):
        """
        Hs: (S,H), ys: (S,C) -> Q0: (C,H)
        """
        if mask_s is not None:
            m = mask_s.float().unsqueeze(-1)           # (S,1)
            Hs = Hs * m                                # zero out padded supports
            ys = ys * mask_s.float().unsqueeze(-1)     # zero out padded labels

        denom = ys.sum(dim=0).clamp_min(self.eps)      # (C,)
        Q0 = (Hs.unsqueeze(1) * ys.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)  # (C,H)
        return Q0

    def forward(self, Xs: torch.Tensor, ys: torch.Tensor, mask_s: torch.Tensor | None = None):
        assert Xs.dim() == 2 and ys.dim() == 2
        S, E = Xs.shape
        S2, C = ys.shape
        assert S == S2 and E == self.embed_dim

        Hs = self.phi(Xs)  # (S,H)

        # initial class queries (prototype-like), then refine via cross-attention
        Q0 = self._masked_weighted_mean(Hs, ys, mask_s)            # (C,H)
        Rc = self.attn_pool(queries=Q0, support=Hs, mask_s=mask_s) # (C,H)

        params = self.psi(Rc)  # (C, out_dim)

        if self.use_gate:
            dW = params[:, :E]                     # (C,E)
            db = params[:, E:E+1].squeeze(-1)      # (C,)
            gate = torch.sigmoid(params[:, E+1:])  # (C,E)
            W = self.W0.unsqueeze(0) + gate * dW
        else:
            W = self.W0.unsqueeze(0) + params[:, :E]
            db = params[:, E:E+1].squeeze(-1)

        b = self.b0.squeeze(0) + db
        return W, b

    @staticmethod
    def predict_query(Xq: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Xq: (Q,E), W: (C,E), b:(C,) -> logits: (Q,C)
        return Xq @ W.t() + b.unsqueeze(0)

class MetaLinearHeadML_Attn(nn.Module):
    """
    Multi-label meta linear head with attention pooling (single episode).

    Inputs:
      Xs: (S, E)
      ys: (S, C) soft/hard multilabel targets in [0,1]
    Outputs:
      W: (C, E)
      b: (C,)
    """
    def __init__(
        self,
        embed_dim: int,
        h_dim: int = 256,
        num_heads: int = 4,
        use_gate: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.use_gate = use_gate
        self.eps = eps

        self.phi = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.attn_pool = ClassCrossAttnPool(h_dim=h_dim, num_heads=num_heads)

        out_dim = embed_dim + 1 + (embed_dim if use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

        self.W0 = nn.Parameter(torch.zeros(embed_dim))
        self.b0 = nn.Parameter(torch.zeros(1))

    def _masked_weighted_mean(self, Hs: torch.Tensor, w: torch.Tensor, mask_s: torch.Tensor | None):
        """
        Hs: (S,H), w: (S,C) -> Q0: (C,H)
        """
        if mask_s is not None:
            m = mask_s.float().unsqueeze(-1)
            Hs = Hs * m
            w = w * mask_s.float().unsqueeze(-1)

        denom = w.sum(dim=0).clamp_min(self.eps)  # (C,)
        Q0 = (Hs.unsqueeze(1) * w.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)  # (C,H)
        return Q0

    def forward(self, Xs: torch.Tensor, ys: torch.Tensor, mask_s: torch.Tensor | None = None):
        assert Xs.dim() == 2 and ys.dim() == 2
        S, E = Xs.shape
        S2, C = ys.shape
        assert S == S2 and E == self.embed_dim

        Hs = self.phi(Xs)  # (S,H)

        w_pos = ys
        w_neg = 1.0 - ys

        Qpos0 = self._masked_weighted_mean(Hs, w_pos, mask_s)  # (C,H)
        Qneg0 = self._masked_weighted_mean(Hs, w_neg, mask_s)  # (C,H)

        Rpos = self.attn_pool(Qpos0, Hs, mask_s=mask_s)  # (C,H)
        Rneg = self.attn_pool(Qneg0, Hs, mask_s=mask_s)  # (C,H)

        Rc = torch.cat([Rpos, Rneg], dim=-1)  # (C,2H)
        params = self.psi(Rc)                 # (C,out_dim)

        if self.use_gate:
            dW = params[:, :E]
            db = params[:, E:E+1].squeeze(-1)
            gate = torch.sigmoid(params[:, E+1:])
            W = self.W0.unsqueeze(0) + gate * dW
        else:
            W = self.W0.unsqueeze(0) + params[:, :E]
            db = params[:, E:E+1].squeeze(-1)

        b = self.b0.squeeze(0) + db
        return W, b

    @staticmethod
    def predict_query(Xq: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return Xq @ W.t() + b.unsqueeze(0)

