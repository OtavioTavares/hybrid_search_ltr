import pickle
import time
import unicodedata
from dataclasses import dataclass

import numpy as np

from product_search import config
from product_search.data.loading import load_products
from product_search.ranking import features
from product_search.retrieval.bm25 import BM25
from product_search.retrieval.dense import Dense
from product_search.retrieval.fusion import rrf, top_k
from product_search.text import tokenize


def clean_query(query) -> str:
    if not isinstance(query, str):
        raise TypeError("a busca deve ser texto")
    query = unicodedata.normalize("NFKC", query)
    query = " ".join("".join(c if c.isprintable() else " " for c in query).split())
    if not query:
        raise ValueError("a busca está vazia")
    return query[: config.MAX_QUERY_CHARS]


def check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= config.N_CANDIDATES:
        raise ValueError(f"k deve ser um inteiro entre 1 e {config.N_CANDIDATES}")
    return k


@dataclass
class Retrieved:
    candidates: np.ndarray
    bm25: np.ndarray
    cosine: np.ndarray


class SearchEngine:
    def __init__(self, products, bm25: BM25, dense: Dense, ranker=None):
        self.products, self.bm25, self.dense, self.ranker = products, bm25, dense, ranker
        self.table = features.product_table(products)
        self.method = getattr(ranker, "retriever", "hybrid")

    @classmethod
    def load(cls, with_ranker: bool = True):
        with open(config.ARTIFACTS / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        dense = Dense(embeddings=np.load(config.ARTIFACTS / "embeddings.npy"))
        ranker = None
        if with_ranker and (config.ARTIFACTS / "ranker.pkl").exists():
            with open(config.ARTIFACTS / "ranker.pkl", "rb") as f:
                ranker = pickle.load(f)
        engine = cls(load_products(), bm25, dense, ranker)
        engine.search("warm up")
        return engine

    def retrieve(self, query: str, method: str | None = None, k: int = config.N_CANDIDATES) -> Retrieved:
        method = method or self.method
        bm25, cosine = self.bm25.scores(query), self.dense.scores(query)
        lists = {"bm25": top_k(bm25, k), "dense": top_k(cosine, k)}
        cand = rrf(lists.values(), k) if method == "hybrid" else lists[method]
        return Retrieved(cand, bm25, cosine)

    def features(self, query: str, r: Retrieved):
        return features.build(query, r.candidates, self.table, r.bm25, r.cosine)

    def rank(self, query: str, r: Retrieved) -> np.ndarray:
        if self.ranker is None:
            return r.candidates
        scores = self.ranker.predict(self.features(query, r))
        return r.candidates[np.argsort(-scores, kind="stable")]

    def search(self, query, k: int = 10) -> dict:
        query, k = clean_query(query), check_k(k)
        if not tokenize(query):
            return {"query": query, "results": [], "latency_ms": {}}
        t0 = time.perf_counter()
        r = self.retrieve(query)
        t1 = time.perf_counter()
        ranked = self.rank(query, r)[:k]
        t2 = time.perf_counter()
        cols = ["product_id", "product_name", "product_class"]
        return {
            "query": query,
            "results": self.products.iloc[ranked][cols].to_dict("records"),
            "latency_ms": {"retrieval": round((t1 - t0) * 1000, 1), "ranking": round((t2 - t1) * 1000, 1)},
        }
