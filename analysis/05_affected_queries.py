"""As features de número (ideia 5) e de atributo (ideia 6) ajudam nas buscas que pedem número ou atributo?

Validação cruzada em 5 partes por query sobre treino + validação (o teste não entra), com número fixo de
árvores para não ajustar nada nas partes de avaliação.

Saída: reports/analysis/05_affected_queries.json
Uso: uv run python analysis/05_affected_queries.py
"""

import pandas as pd

from product_search.engine import SearchEngine
from product_search.evaluation.cv import cross_val_scores
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, ranked_run, save_json
from product_search.ranking import experimental
from product_search.ranking.experimental import ATTRIBUTE_KINDS
from product_search.ranking.features import FEATURES
from product_search.text import tokenize

KIND_MATCH = [f"{k}_match" for k in ATTRIBUTE_KINDS]
VARIANTS = {
    "v1": FEATURES,
    "v1 + número com tolerância (5)": FEATURES + ["number_close"],
    "v1 + cor/material/estilo (6)": FEATURES + KIND_MATCH,
    "v1 + 5 + 6": FEATURES + ["number_close"] + KIND_MATCH,
}


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    queries = pd.concat([data.queries["train"], data.queries["val"]])
    frame = build_frame(engine, queries, data.qrels, "hybrid", with_experimental=True)

    vocab = experimental.product_table(engine.products)["vocab"]
    text = queries.set_index("query_id")["query"]
    subsets = {
        "todas": text.index,
        "com número": text.index[text.str.contains(r"\d")],
        "pedem cor/material/estilo": text.index[text.map(lambda q: any(set(tokenize(q)) & vocab[k] for k in ATTRIBUTE_KINDS))],
    }
    subsets["sem número nem atributo"] = text.index.difference(subsets["com número"].union(subsets["pedem cor/material/estilo"]))

    ndcg = {
        name: evaluate(ranked_run(frame, cross_val_scores(frame, feats)), data.qrels)["ndcg@10"]
        for name, feats in VARIANTS.items()
    }
    result = {}
    for subset, ids in subsets.items():
        base = ndcg["v1"].loc[ids]
        result[subset] = {"n_queries": len(ids), "v1": round(base.mean(), 4)}
        for name in list(VARIANTS)[1:]:
            result[subset][name] = {
                "ndcg@10": round(ndcg[name].loc[ids].mean(), 4),
                "vs_v1": paired_bootstrap(base, ndcg[name].loc[ids]),
            }
    save_json("analysis/05_affected_queries.json", result)


if __name__ == "__main__":
    main()
