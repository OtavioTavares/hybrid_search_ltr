"""Tentativa v2: features novas somadas à v1, na validação, com o mesmo treino do train-ranker.

Regra: uma variante só substitui a v1 se ganhar dela em nDCG@10 com IC 95% acima de zero.

Saída: reports/analysis/04_v2_features.json
Uso: uv run python analysis/04_v2_features.py
"""

from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, mean, ranked_run, save_json
from product_search.ranking.experimental import V2
from product_search.ranking.features import FEATURES, GROUPS
from product_search.ranking.model import Ranker

# Mesma ordem da execução original: com colsample_bytree < 1, a ordem das colunas muda o sorteio do LightGBM.
ALL = (
    GROUPS["retrieval"]
    + V2["normalized"]
    + GROUPS["text"]
    + V2["class"]
    + GROUPS["attributes"]
    + V2["attributes novos"]
    + GROUPS["numeric"]
)
VARIANTS = {"v1": FEATURES, "v2 (v1 + novas)": ALL}
VARIANTS |= {f"v2 sem {g}": [f for f in ALL if f not in feats] for g, feats in V2.items()}


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    train, val = (build_frame(engine, data.queries[s], data.qrels, "hybrid", with_experimental=True) for s in ["train", "val"])

    models, per_query = {}, {}
    for name, features in VARIANTS.items():
        models[name] = Ranker(features).fit(train, val)
        per_query[name] = evaluate(ranked_run(val, models[name].predict(val)), data.qrels)

    base = per_query["v1"]["ndcg@10"]
    save_json(
        "analysis/04_v2_features.json",
        {
            name: mean(df)
            | {
                "best_iteration": int(models[name].model.best_iteration_ or 0),
                "ndcg@10_vs_v1": paired_bootstrap(base, df["ndcg@10"]),
            }
            for name, df in per_query.items()
        },
    )


if __name__ == "__main__":
    main()
