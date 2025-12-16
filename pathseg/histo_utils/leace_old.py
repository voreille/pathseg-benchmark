from typing import Iterable, Tuple

import torch


def leace_projector(
    X: torch.Tensor,
    Y: torch.Tensor,
    concept_rank: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a LEACE-style linear projector that removes all *linear* information
    about Y from X, while minimally distorting X.

    Args:
        X: [N, D] tensor of features (e.g., ViT patch tokens).
        Y: [N, K] tensor of concept labels (e.g., (x,y) positions or stain params).
        concept_rank: number of concept dimensions to erase (r).
            - If None: uses r = min(K, D) or the numerical rank of the cross-cov.
        eps: small jitter for numerical stability in eigen decomposition.

    Returns:
        P: [D, D] projection matrix.
           For centered features Xc = X - X.mean(0), X_clean = Xc @ P.T.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X must be [N, D] and Y must be [N, K].")

    N, D = X.shape
    Ny, K = Y.shape
    if Ny != N:
        raise ValueError(
            f"X and Y must have the same number of rows; got {N} and {Ny}."
        )

    # Work in float64 for numerical stability
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)

    # 1) Center X and Y
    X_mean = X.mean(dim=0, keepdim=True)  # [1, D]
    Y_mean = Y.mean(dim=0, keepdim=True)  # [1, K]
    Xc = X - X_mean  # [N, D]
    Yc = Y - Y_mean  # [N, K]

    # 2) Covariances
    #    Sigma_xx: [D, D], Sigma_xy: [D, K]
    denom = max(N - 1, 1)
    Sigma_xx = (Xc.T @ Xc) / denom  # [D, D]
    Sigma_xy = (Xc.T @ Yc) / denom  # [D, K]

    # 3) Whiten X: find W such that W @ Sigma_xx @ W^T = I
    #    Use eigen-decomposition for symmetric positive semi-definite matrix
    eigvals, eigvecs = torch.linalg.eigh(Sigma_xx)  # eigvecs: [D, D]
    # Clamp eigenvalues for stability
    eigvals_clamped = torch.clamp(eigvals, min=eps)

    D_inv_sqrt = torch.diag(eigvals_clamped.rsqrt())  # [D, D]
    D_sqrt = torch.diag(eigvals_clamped.sqrt())  # [D, D]

    # whitening transform: Xw = Xc @ W^T
    W = eigvecs @ D_inv_sqrt @ eigvecs.T  # [D, D]
    Winv = eigvecs @ D_sqrt @ eigvecs.T  # [D, D]

    Xw = Xc @ W.T  # [N, D]

    # 4) Cross-covariance in whitened space
    #    C = Cov(Xw, Y)
    C = (Xw.T @ Yc) / denom  # [D, K]

    # 5) SVD of C: C = U S V^T
    #    Columns of U span the concept subspace (in whitened space)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)  # U: [D, min(D,K)]

    # Decide how many concept dims to erase
    max_rank = min(D, K, U.shape[1])
    if concept_rank is None or concept_rank <= 0 or concept_rank > max_rank:
        r = max_rank
    else:
        r = concept_rank

    U_r = U[:, :r]  # [D, r]

    # 6) Projector in whitened space: P_w = I - U_r U_r^T
    I = torch.eye(D, dtype=torch.float64, device=X.device)
    P_w = I - U_r @ U_r.T  # [D, D]

    # 7) Map projector back to original space:
    #    P = Sigma_xx^{1/2} P_w Sigma_xx^{-1/2}
    P = Winv @ P_w @ W  # [D, D]

    # Return as float32 to save memory, typically enough for downstream use
    return P.to(torch.float32)


def leace_projector_streaming(
    batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    concept_rank: int | None = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming LEACE-style projector:
    Compute a linear projector P that removes all *linear* information about Y from X,
    using only mini-batches (no need to keep all X,Y in memory).

    Args:
        batch_iter: iterable yielding (X_batch, Y_batch) tuples:
            - X_batch: [B, D] float tensor of features
            - Y_batch: [B, K] float tensor of concept labels
        concept_rank: number of concept dimensions to erase (r).
            - If None or <=0: uses r = min(D, K, rank(C)).
        eps: small jitter for numerical stability.

    Returns:
        P:       [D, D] float32 projection matrix.
                 For centered Xc, X_clean = Xc @ P.T
        mean_x:  [D] float32 mean of X over all samples (used for centering).
    """
    # 1) First pass: accumulate sums and second moments
    n_total = 0
    sum_x = None  # [D]
    sum_y = None  # [K]
    sum_xx = None  # [D, D]
    sum_xy = None  # [D, K]

    for Xb, Yb in batch_iter:
        if Xb.ndim != 2 or Yb.ndim != 2:
            raise ValueError("Each X_batch must be [B, D] and Y_batch [B, K].")

        Xb = Xb.to(torch.float64)
        Yb = Yb.to(torch.float64)

        B, D = Xb.shape
        _, K = Yb.shape

        if sum_x is None:
            sum_x = torch.zeros(D, dtype=torch.float64, device=Xb.device)
            sum_y = torch.zeros(K, dtype=torch.float64, device=Xb.device)
            sum_xx = torch.zeros(D, D, dtype=torch.float64, device=Xb.device)
            sum_xy = torch.zeros(D, K, dtype=torch.float64, device=Xb.device)

        n_total += B
        sum_x += Xb.sum(dim=0)  # [D]
        sum_y += Yb.sum(dim=0)  # [K]
        sum_xx += Xb.T @ Xb  # [D, D]
        sum_xy += Xb.T @ Yb  # [D, K]

    if n_total == 0:
        raise RuntimeError("leace_projector_streaming: no samples seen in batch_iter.")

    # 2) Means
    n = float(n_total)
    mean_x = sum_x / n  # [D]
    mean_y = sum_y / n  # [K]

    # 3) Covariances using: Σ (x - μ)(x - μ)^T = Σ xx^T - n μ μ^T
    denom = max(n_total - 1, 1)

    mean_x_col = mean_x.unsqueeze(1)  # [D,1]
    mean_y_row = mean_y.unsqueeze(0)  # [1,K]

    Sigma_xx = (sum_xx - n * (mean_x_col @ mean_x_col.T)) / denom  # [D,D]
    Sigma_xy = (sum_xy - n * (mean_x_col @ mean_y_row)) / denom  # [D,K]

    # 4) Whiten X: Sigma_xx^{-1/2}
    eigvals, eigvecs = torch.linalg.eigh(Sigma_xx)  # eigvecs: [D,D]
    eigvals_clamped = torch.clamp(eigvals, min=eps)

    D_inv_sqrt = torch.diag(eigvals_clamped.rsqrt())  # [D,D]
    D_sqrt = torch.diag(eigvals_clamped.sqrt())  # [D,D]

    W = eigvecs @ D_inv_sqrt @ eigvecs.T  # Sigma_xx^{-1/2}, [D,D]
    Winv = eigvecs @ D_sqrt @ eigvecs.T  # Sigma_xx^{+1/2}, [D,D]

    # 5) Cross-covariance in whitened space: C = Cov(Xw, Y)
    #    C = W Sigma_xy
    C = W @ Sigma_xy  # [D,K]

    # 6) SVD of C: concept subspace in whitened space
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)  # U: [D, min(D,K)]

    max_rank = min(U.shape[1], Sigma_xy.shape[0], Sigma_xy.shape[1])
    if concept_rank is None or concept_rank <= 0 or concept_rank > max_rank:
        r = max_rank
    else:
        r = concept_rank

    U_r = U[:, :r]  # [D,r]

    # 7) Projector in whitened space and map back:
    I = torch.eye(U_r.shape[0], dtype=torch.float64, device=U_r.device)
    P_w = I - U_r @ U_r.T  # [D,D]

    P = Winv @ P_w @ W  # [D,D] in original space

    return P.to(torch.float32), mean_x.to(torch.float32)
