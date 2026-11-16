"""Indexação (BM25 + embeddings) -> artifacts/, custo -> reports/retrieval/index_cost.json.  Uso: uv run build-index"""

import pickle
import time

import numpy as np

from product_search import config
from product_search.data.loading import load_products
from product_search.evaluation.runs import save_json
from product_search.retrieval.bm25 import BM25
from product_search.retrieval.dense import Dense


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    products = load_products()

    t0 = time.perf_counter()
    bm25 = BM25().fit(products.lexical_text)
    t_bm25 = time.perf_counter() - t0
    with open(config.ARTIFACTS / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    t0 = time.perf_counter()
    embeddings = Dense().fit(products.dense_text).embeddings.astype(np.float32)
    t_dense = time.perf_counter() - t0
    np.save(config.ARTIFACTS / "embeddings.npy", embeddings)

    cost = {
        name: {
            "seconds": round(seconds, 1),
            "ms_per_product": round(1000 * seconds / len(products), 2),
            "mb": round((config.ARTIFACTS / file).stat().st_size / 1e6, 1),
        }
        for name, seconds, file in [("bm25", t_bm25, "bm25.pkl"), ("embeddings", t_dense, "embeddings.npy")]
    }
    save_json("retrieval/index_cost.json", {"n_products": len(products)} | cost)


if __name__ == "__main__":
    main()
