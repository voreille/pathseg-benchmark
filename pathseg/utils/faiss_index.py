from dataclasses import dataclass

import faiss
import numpy as np
import torch


@dataclass
class PatchTokenIndex:
    index: faiss.Index
    image_ids: np.ndarray
    patch_ids: np.ndarray
    num_patches: int
    dimension: int

    def search(
        self,
        query_tokens: torch.Tensor,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        queries = prepare_tokens(query_tokens)

        similarities, token_ids = self.index.search(queries, k)

        valid = token_ids >= 0

        matched_image_ids = np.full_like(token_ids, -1)
        matched_patch_ids = np.full_like(token_ids, -1)

        matched_image_ids[valid] = self.image_ids[token_ids[valid]]
        matched_patch_ids[valid] = self.patch_ids[token_ids[valid]]

        return (
            similarities,
            token_ids,
            matched_image_ids,
            matched_patch_ids,
        )


def prepare_tokens(tokens: torch.Tensor) -> np.ndarray:
    if tokens.ndim == 3:
        tokens = tokens.flatten(0, 1)

    if tokens.ndim != 2:
        raise ValueError(f"Expected [T, D] or [M, P, D], got {tuple(tokens.shape)}.")

    vectors = tokens.detach().to(dtype=torch.float32).cpu().contiguous().numpy()

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    return vectors


def build_patch_token_index(
    training_tokens: torch.Tensor,
    *,
    use_ivf: bool = True,
    nlist: int | None = None,
    nprobe: int = 16,
    seed: int = 42,
) -> PatchTokenIndex:
    if training_tokens.ndim != 3:
        raise ValueError(
            "Expected training tokens with shape [images, patches, dimension]."
        )

    num_images, num_patches, dimension = training_tokens.shape
    vectors = prepare_tokens(training_tokens)
    total_tokens = len(vectors)

    image_ids = np.repeat(
        np.arange(num_images, dtype=np.int64),
        num_patches,
    )
    patch_ids = np.tile(
        np.arange(num_patches, dtype=np.int64),
        num_images,
    )

    if not use_ivf:
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

    else:
        if nlist is None:
            nlist = min(4096, max(64, int(np.sqrt(total_tokens))))

        if nlist >= total_tokens:
            raise ValueError(
                f"nlist={nlist} must be smaller than "
                f"the number of tokens={total_tokens}."
            )

        quantizer = faiss.IndexFlatIP(dimension)

        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        rng = np.random.default_rng(seed)

        # Aim for at least 50 examples per centroid.
        num_training_vectors = min(
            total_tokens,
            max(100_000, 50 * nlist),
        )

        selected = rng.choice(
            total_tokens,
            size=num_training_vectors,
            replace=False,
        )

        clustering_vectors = np.ascontiguousarray(
            vectors[selected],
            dtype=np.float32,
        )

        index.train(clustering_vectors)

        if not index.is_trained:
            raise RuntimeError("Faiss IVF training failed.")

        index.add(vectors)
        index.nprobe = min(nprobe, nlist)

    return PatchTokenIndex(
        index=index,
        image_ids=image_ids,
        patch_ids=patch_ids,
        num_patches=num_patches,
        dimension=dimension,
    )
