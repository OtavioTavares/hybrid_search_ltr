import numpy as np

from product_search import config


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def rrf(ranked_lists, k: int = config.N_CANDIDATES, rrf_k: int = config.RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion: usa só posições, então não precisa calibrar escalas de score."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked):
            fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + rank + 1)
    return np.array(sorted(fused, key=fused.get, reverse=True)[:k], dtype=int)
