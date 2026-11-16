from functools import lru_cache

import numpy as np

from product_search import config


class Dense:
    def __init__(self, model_name: str = config.DENSE_MODEL, embeddings: np.ndarray | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=config.DENSE_REVISION, device="cpu")
        self.embeddings = embeddings
        self.encode = lru_cache(maxsize=2048)(self._encode)

    def fit(self, texts, batch_size: int = 256):
        self.embeddings = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return self

    def _encode(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]

    def scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.encode(query)
