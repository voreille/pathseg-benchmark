from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

FitBatch: TypeAlias = torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]


class FullPCA(nn.Module):
    """
    Full PCA model fitted from streamed embedding batches.

    Stores:
        mean:                       [D]
        components:                 [D, D]
        explained_variance:         [D]
        explained_variance_ratio:   [D]

    Components are ordered from highest to lowest explained variance.
    """

    def __init__(
        self,
        feature_dim: int,
        normalize_embeddings: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.feature_dim = feature_dim
        self.normalize_embeddings = normalize_embeddings
        self.eps = eps

        self.register_buffer(
            "mean",
            torch.zeros(feature_dim),
        )
        self.register_buffer(
            "components",
            torch.eye(feature_dim),
        )
        self.register_buffer(
            "explained_variance",
            torch.zeros(feature_dim),
        )
        self.register_buffer(
            "explained_variance_ratio",
            torch.zeros(feature_dim),
        )
        self.register_buffer(
            "_is_fitted",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "_n_fit_tokens",
            torch.tensor(0, dtype=torch.long),
        )

    @property
    def is_fitted(self) -> bool:
        return bool(self._is_fitted.item())

    @property
    def n_fit_tokens(self) -> int:
        return int(self._n_fit_tokens.item())

    def preprocess(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {tokens.shape[-1]}."
            )

        tokens = tokens.float()

        if self.normalize_embeddings:
            tokens = F.normalize(
                tokens,
                p=2,
                dim=-1,
                eps=self.eps,
            )

        return tokens

    def _prepare_fit_batch(
        self,
        item: FitBatch,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if isinstance(item, tuple):
            tokens, mask = item
        else:
            tokens = item
            mask = None

        if tokens.ndim not in (2, 3):
            raise ValueError(
                "Tokens must have shape [M, D] or [B, N, D], "
                f"got {tuple(tokens.shape)}."
            )

        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {tokens.shape[-1]}."
            )

        tokens = tokens.detach()

        if mask is not None:
            expected_shape = tokens.shape[:-1]

            if tuple(mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Expected mask shape {tuple(expected_shape)}, "
                    f"got {tuple(mask.shape)}."
                )

            tokens = tokens[mask.detach().bool()]
        else:
            tokens = tokens.reshape(-1, self.feature_dim)

        if tokens.numel() == 0:
            return None

        tokens = tokens.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        if self.normalize_embeddings:
            tokens = F.normalize(
                tokens,
                p=2,
                dim=-1,
                eps=self.eps,
            )

        if not torch.isfinite(tokens).all():
            raise ValueError("Non-finite values found in PCA fitting tokens.")

        return tokens

    @torch.no_grad()
    def fit(
        self,
        token_batches: Iterable[FitBatch],
        *,
        fit_device: str | torch.device = "cpu",
        fit_dtype: torch.dtype = torch.float64,
    ) -> FullPCA:
        fit_device = torch.device(fit_device)

        count = 0
        running_mean = None
        running_m2 = None

        for item in token_batches:
            x = self._prepare_fit_batch(
                item,
                device=fit_device,
                dtype=fit_dtype,
            )

            if x is None:
                continue

            batch_count = x.shape[0]
            batch_mean = x.mean(dim=0)

            centered = x - batch_mean
            batch_m2 = centered.T @ centered

            if running_mean is None:
                count = batch_count
                running_mean = batch_mean
                running_m2 = batch_m2
                continue

            total_count = count + batch_count
            delta = batch_mean - running_mean

            running_m2 = (
                running_m2
                + batch_m2
                + torch.outer(delta, delta) * (count * batch_count / total_count)
            )

            running_mean = running_mean + delta * (batch_count / total_count)

            count = total_count

        if running_mean is None or running_m2 is None:
            raise RuntimeError("No valid tokens were supplied for PCA fitting.")

        if count <= self.feature_dim:
            raise ValueError(
                "A full PCA requires more observations than feature "
                f"dimensions. Received {count} tokens for "
                f"{self.feature_dim} dimensions."
            )

        covariance = running_m2 / (count - 1)

        # Eigenvalues/eigenvectors are returned in ascending order.
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]

        # One PCA component per row.
        components = eigenvectors.T.contiguous()

        total_variance = eigenvalues.sum()

        if total_variance > 0:
            variance_ratio = eigenvalues / total_variance
        else:
            variance_ratio = torch.zeros_like(eigenvalues)

        storage_device = self.mean.device

        self.mean.copy_(
            running_mean.to(
                device=storage_device,
                dtype=torch.float32,
            )
        )
        self.components.copy_(
            components.to(
                device=storage_device,
                dtype=torch.float32,
            )
        )
        self.explained_variance.copy_(
            eigenvalues.to(
                device=storage_device,
                dtype=torch.float32,
            )
        )
        self.explained_variance_ratio.copy_(
            variance_ratio.to(
                device=storage_device,
                dtype=torch.float32,
            )
        )

        self._n_fit_tokens.fill_(count)
        self._is_fitted.fill_(True)

        return self

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Transform embeddings into the complete PCA coordinate system.

        Input:
            [..., D]

        Output:
            [..., D]
        """
        if not self.is_fitted:
            raise RuntimeError("PCA has not been fitted.")

        tokens = self.preprocess(tokens)
        centered = tokens - self.mean

        return centered @ self.components.T

    def cumulative_explained_variance_ratio(
        self,
    ) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA has not been fitted.")

        return torch.cumsum(
            self.explained_variance_ratio,
            dim=0,
        )

class PCAResidualScorer(nn.Module):
    """
    Score tokens using a selected interval of PCA coordinates.

    The score uses components:

        [n_major_components : end_component]

    Without whitening:
        Euclidean norm in the selected PCA subspace.

    With whitening:
        Regularized Mahalanobis norm in the selected PCA subspace.

    Parameters
    ----------
    pca:
        A fitted FullPCA instance.

    n_major_components:
        Number of leading, high-variance PCA components to exclude.

        0 uses the complete PCA space.

    whiten:
        Whether to standardize each selected coordinate by its fitted
        standard deviation.

    end_component:
        Exclusive index of the last component used. If None, the cutoff
        is inferred from `min_eigenvalue_ratio`, or all components are used.

    min_eigenvalue_ratio:
        Optionally exclude components whose eigenvalue is below:

            largest_eigenvalue * min_eigenvalue_ratio

        Because eigenvalues are sorted in descending order, this determines
        an exclusive end index.

    relative_ridge:
        Ridge added to eigenvalues during whitening, relative to the largest
        eigenvalue.

    normalize_by_dimension:
        Divide the norm by sqrt(number of selected components). This does
        not change ranking for a fixed configuration, but helps compare
        score scales across different component ranges.
    """

    def __init__(
        self,
        pca: "FullPCA",
        n_major_components: int = 0,
        *,
        whiten: bool = False,
        end_component: Optional[int] = None,
        min_eigenvalue_ratio: Optional[float] = None,
        relative_ridge: float = 1e-4,
        normalize_by_dimension: bool = False,
    ) -> None:
        super().__init__()

        if not pca.is_fitted:
            raise ValueError("PCA must be fitted before creating the scorer.")

        if end_component is not None and min_eigenvalue_ratio is not None:
            raise ValueError(
                "Specify either end_component or min_eigenvalue_ratio, not both."
            )

        if relative_ridge < 0:
            raise ValueError("relative_ridge must be non-negative.")

        if min_eigenvalue_ratio is not None and min_eigenvalue_ratio < 0:
            raise ValueError("min_eigenvalue_ratio must be non-negative.")

        self.pca = pca
        self.whiten = whiten
        self.relative_ridge = relative_ridge
        self.normalize_by_dimension = normalize_by_dimension

        self.n_major_components = n_major_components
        self.explicit_end_component = end_component
        self.min_eigenvalue_ratio = min_eigenvalue_ratio

        # Validate the resulting range immediately.
        self._component_range()

    def _component_range(self) -> tuple[int, int]:
        start = self.n_major_components
        feature_dim = self.pca.feature_dim

        if not 0 <= start < feature_dim:
            raise ValueError(
                f"n_major_components must be in [0, {feature_dim - 1}], got {start}."
            )

        if self.explicit_end_component is not None:
            end = self.explicit_end_component

        elif self.min_eigenvalue_ratio is not None:
            eigenvalues = self.pca.explained_variance
            threshold = eigenvalues[0] * self.min_eigenvalue_ratio

            # Number of eigenvalues that remain above the threshold.
            end = int((eigenvalues >= threshold).sum().item())

        else:
            end = feature_dim

        if not start < end <= feature_dim:
            raise ValueError(
                f"The selected PCA interval is empty or invalid: [{start}, {end})."
            )

        return start, end

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [..., embedding_dim]

        Returns:
            One scalar score per token: [...]
        """
        start, end = self._component_range()

        coordinates = self.pca.transform(tokens)
        selected = coordinates[..., start:end]

        if self.whiten:
            variances = self.pca.explained_variance[start:end]

            ridge = self.relative_ridge * self.pca.explained_variance[0]

            selected = selected / torch.sqrt(variances + ridge)

        scores = torch.linalg.vector_norm(
            selected,
            ord=2,
            dim=-1,
        )

        if self.normalize_by_dimension:
            scores = scores / (end - start) ** 0.5

        return scores

    def extra_repr(self) -> str:
        start, end = self._component_range()

        return (
            f"components=[{start}, {end}), "
            f"whiten={self.whiten}, "
            f"relative_ridge={self.relative_ridge}, "
            f"normalize_by_dimension={self.normalize_by_dimension}"
        )
