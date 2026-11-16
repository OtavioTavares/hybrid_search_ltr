"""Alternativas ao ranker final, na validação: peso do Exact no treino e outros objetivos.

Saída: reports/analysis/07_ranking_alternatives.json
Uso: uv run python analysis/07_ranking_alternatives.py   (depois de train-ranker)
"""

import lightgbm as lgb

from product_search import config
from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, ranked_run, save_json
from product_search.ranking.features import FEATURES

PARAMS = {k: v for k, v in config.LGBM.items() if k != "objective"}


def groups(frame):
    return frame.groupby("query_id", sort=False).size().to_numpy()


def fit_predict(kind, train, val, **extra):
    """Treina na base de features da v1 e devolve os scores da validação."""
    X, y, Xv, yv = train[FEATURES], train.label, val[FEATURES], val.label
    stop = [lgb.early_stopping(config.EARLY_STOPPING, verbose=False)]
    if kind == "regression":
        model = lgb.LGBMRegressor(objective="regression", **PARAMS)
        model.fit(X, y, eval_X=(Xv,), eval_y=(yv,), callbacks=stop)
        return model.predict(Xv)
    if kind == "binary":
        model = lgb.LGBMClassifier(objective="binary", **PARAMS)
        model.fit(X, y.eq(2).astype(int), eval_X=(Xv,), eval_y=(yv.eq(2).astype(int),), callbacks=stop)
        return model.predict_proba(Xv)[:, 1]
    model = lgb.LGBMRanker(objective=kind, **PARAMS, **extra)
    model.fit(X, y, group=groups(train), eval_X=(Xv,), eval_y=(yv,), eval_group=[groups(val)], eval_at=[10], callbacks=stop)
    return model.predict(Xv)


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    train, val = (build_frame(engine, data.queries[s], data.qrels, "hybrid") for s in ["train", "val"])

    runs = {
        "LambdaMART, Exact = 3 (final)": ("lambdarank", {}),
        **{f"LambdaMART, Exact = {g}": ("lambdarank", {"label_gain": [0, 1, g]}) for g in [2, 5, 10, 30]},
        "regressão (nota 0/1/2)": ("regression", {}),
        "classificação (é Exact?)": ("binary", {}),
        "XE-NDCG": ("rank_xendcg", {}),
    }
    per_query = {
        name: evaluate(ranked_run(val, fit_predict(kind, train, val, **extra)), data.qrels)
        for name, (kind, extra) in runs.items()
    }
    final = per_query["LambdaMART, Exact = 3 (final)"]["ndcg@10"]
    save_json(
        "analysis/07_ranking_alternatives.json",
        {
            name: {"ndcg@10": round(df["ndcg@10"].mean(), 4), "vs_final": paired_bootstrap(final, df["ndcg@10"])}
            for name, df in per_query.items()
        },
    )


if __name__ == "__main__":
    main()
