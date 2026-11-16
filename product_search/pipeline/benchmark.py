"""Latência p50/p95 de retrieval e ranking, separadas, nas queries de teste -> reports/latency.json.

Uso: uv run benchmark
"""

import time

import numpy as np

from product_search.engine import SearchEngine
from product_search.evaluation.runs import load_dataset, save_json


def percentiles(seconds) -> dict:
    return {f"p{p}_ms": round(1000 * float(np.percentile(seconds, p)), 1) for p in (50, 95)}


def main():
    queries = load_dataset().queries["test"]["query"].tolist()
    engine = SearchEngine.load()
    engine.dense.encode.cache_clear()
    t_retrieval, t_ranking = [], []
    for q in queries:
        t0 = time.perf_counter()
        r = engine.retrieve(q)
        t1 = time.perf_counter()
        engine.rank(q, r)
        t_retrieval.append(t1 - t0)
        t_ranking.append(time.perf_counter() - t1)
    save_json(
        "latency.json", {"n_queries": len(queries), "retrieval": percentiles(t_retrieval), "ranking": percentiles(t_ranking)}
    )


if __name__ == "__main__":
    main()
