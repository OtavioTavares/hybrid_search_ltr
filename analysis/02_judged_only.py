"""Treinar só com os candidatos julgados, em vez de tratar os não julgados como irrelevantes.

Mesmo treino do train-ranker, na validação. Saída: reports/analysis/02_judged_only.json
Uso: uv run python analysis/02_judged_only.py
"""

from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, mean, ranked_run, save_json
from product_search.ranking.features import FEATURES
from product_search.ranking.model import Ranker


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    train, val = (build_frame(engine, data.queries[s], data.qrels, "hybrid") for s in ["train", "val"])

    models = {
        "todos os candidatos (final)": Ranker(FEATURES).fit(train, val),
        "só julgados": Ranker(FEATURES).fit(train[train.judged], val[val.judged]),
    }
    per_query = {name: evaluate(ranked_run(val, m.predict(val)), data.qrels) for name, m in models.items()}
    final, judged = per_query.values()
    save_json(
        "analysis/02_judged_only.json",
        {name: mean(df) for name, df in per_query.items()}
        | {
            "ndcg@10_vs_final": paired_bootstrap(final["ndcg@10"], judged["ndcg@10"]),
            "condensed_vs_final": paired_bootstrap(final["condensed_ndcg@10"], judged["condensed_ndcg@10"]),
        },
    )


if __name__ == "__main__":
    main()
