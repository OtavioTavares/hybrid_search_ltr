"""Avaliação no teste, uma única vez, com o sistema já escolhido -> reports/test.json.

Uso: uv run eval-test [--force]
"""

import argparse
import platform

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from product_search import config
from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, mean, ranked_run, retrieval_run, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    if (config.REPORTS / "test.json").exists() and not parser.parse_args().force:
        raise SystemExit("o teste já foi avaliado (reports/test.json); use --force só se souber o que está fazendo")

    data = load_dataset()
    engine = SearchEngine.load()
    method, queries = engine.method, data.queries["test"]
    frame = build_frame(engine, queries, data.qrels, method)
    runs = {
        "bm25": retrieval_run(engine, queries, "bm25"),
        method: ranked_run(frame),
        f"{method} + LTR": ranked_run(frame, engine.ranker.predict(frame)),
    }
    per_query = {name: evaluate(run, data.qrels) for name, run in runs.items()}
    base, final = per_query[method], per_query[f"{method} + LTR"]

    save_json(
        "test.json",
        {
            "n_queries": len(queries),
            "test": {name: mean(df) for name, df in per_query.items()},
            "bootstrap": {
                f"{method} vs bm25, recall@100": paired_bootstrap(per_query["bm25"]["recall@100"], base["recall@100"]),
                f"LTR vs {method}, ndcg@10": paired_bootstrap(base["ndcg@10"], final["ndcg@10"]),
                f"LTR vs {method}, condensed_ndcg@10": paired_bootstrap(base["condensed_ndcg@10"], final["condensed_ndcg@10"]),
            },
            "environment": {
                "python": platform.python_version(),
                "lightgbm": lightgbm.__version__,
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "seed": config.SEED,
            },
        },
    )


if __name__ == "__main__":
    main()
