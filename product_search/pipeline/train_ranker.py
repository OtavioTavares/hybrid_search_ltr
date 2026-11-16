"""Treina o LambdaMART, faz a ablação por grupo de features e escolhe na validação.

Regra: fica o modelo base, a menos que outra variante ganhe dele em nDCG@10 com IC 95% acima de zero.

Saídas: artifacts/ranker.pkl, reports/ranking/metrics.json, reports/ranking/errors_val.csv
Uso: uv run train-ranker   (depois de eval-retrieval)
"""

import pickle

import pandas as pd

from product_search import config
from product_search.engine import SearchEngine
from product_search.evaluation.errors import error_analysis
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, load_json, mean, ranked_run, save_csv, save_json
from product_search.ranking.features import FEATURES, GROUPS
from product_search.ranking.model import Ranker

BASE = "base"
VARIANTS = {BASE: FEATURES} | {f"base sem {g}": [f for f in FEATURES if f not in feats] for g, feats in GROUPS.items()}


def main():
    data = load_dataset()
    retriever = load_json("retrieval/metrics.json")["selected"]
    engine = SearchEngine.load(with_ranker=False)
    train, val = (build_frame(engine, data.queries[s], data.qrels, retriever) for s in ["train", "val"])

    results = {f"ordem do {retriever}": mean(evaluate(ranked_run(val), data.qrels))}
    models, per_query = {}, {}
    for name, features in VARIANTS.items():
        models[name] = Ranker(features, retriever).fit(train, val)
        per_query[name] = evaluate(ranked_run(val, models[name].predict(val)), data.qrels)
        results[name] = mean(per_query[name])

    vs_base = {n: paired_bootstrap(per_query[BASE]["ndcg@10"], per_query[n]["ndcg@10"]) for n in VARIANTS if n != BASE}
    better = [n for n, b in vs_base.items() if b["ci95"][0] > 0]
    selected = max(better, key=lambda n: results[n]["ndcg@10"]) if better else BASE
    ranker = models[selected]
    with open(config.ARTIFACTS / "ranker.pkl", "wb") as f:
        pickle.dump(ranker, f)

    queries = pd.concat(data.queries.values())
    errors, summary = error_analysis(val, ranked_run(val, ranker.predict(val)), data.qrels, queries, engine.products)
    save_csv("ranking/errors_val.csv", errors)
    save_json(
        "ranking/metrics.json",
        {
            "retriever": retriever,
            "selected": selected,
            "best_iteration": int(ranker.model.best_iteration_ or 0),
            "val": results,
            "train": mean(evaluate(ranked_run(train, ranker.predict(train)), data.qrels)),
            "ndcg@10_vs_base": vs_base,
            "error_analysis": summary,
            "importance": ranker.importance(),
        },
    )


if __name__ == "__main__":
    main()
