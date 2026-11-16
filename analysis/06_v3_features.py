"""Correções das features da v1, avaliadas por validação cruzada.

Regra, definida antes de rodar: uma variante só é adotada se ganhar da v1 em nDCG@10 com IC 95% acima de
zero, na validação cruzada em 5 partes por query sobre treino + validação (383 buscas). O teste não entra.

Saída: reports/analysis/06_v3_features.json
Uso: uv run python analysis/06_v3_features.py
"""

import pandas as pd

from product_search.engine import SearchEngine
from product_search.evaluation.cv import cross_val_scores
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, mean, ranked_run, save_json
from product_search.ranking.experimental import V3
from product_search.ranking.features import FEATURES


def replace(features: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(f, f) for f in features]


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    queries = pd.concat([data.queries["train"], data.queries["val"]])
    frame = build_frame(engine, queries, data.qrels, "hybrid", with_experimental=True)

    all_fixes = {k: v for mapping in V3.values() for k, v in mapping.items()}
    variants = {"v1": FEATURES, "v1 + palavras desconhecidas (oov_share)": FEATURES + ["oov_share"]}
    variants |= {f"v1 com {name}": replace(FEATURES, mapping) for name, mapping in V3.items()}
    variants["v3 (todas as correções + oov_share)"] = replace(FEATURES, all_fixes) + ["oov_share"]

    per_query = {
        name: evaluate(ranked_run(frame, cross_val_scores(frame, feats)), data.qrels) for name, feats in variants.items()
    }
    base = per_query["v1"]
    save_json(
        "analysis/06_v3_features.json",
        {
            "n_queries": len(base),
            "variants": {
                name: mean(df)
                | {
                    "ndcg@10_vs_v1": paired_bootstrap(base["ndcg@10"], df["ndcg@10"]),
                    "condensed_vs_v1": paired_bootstrap(base["condensed_ndcg@10"], df["condensed_ndcg@10"]),
                }
                for name, df in per_query.items()
            },
        },
    )


if __name__ == "__main__":
    main()
