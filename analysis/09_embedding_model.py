"""Modelo de embedding: bge-small-en-v1.5, com texto curto e longo, contra o MiniLM atual.

Retrieval: denso e híbrido no treino e na validação. Ranking: LTR com as 17 features da v1 sobre os candidatos de
cada versão, CV em 5 partes x 3 seeds sobre treino + validação (383 buscas), com early stopping em 20% das buscas
de treino de cada fold; e a validação simples, como no train-ranker. O teste não entra.

Os embeddings do bge levam ~4 min (curto) e ~30 min (longo) em CPU e ficam em cache em artifacts/.
Saída: reports/analysis/09_embedding_model.json
Uso: uv run python analysis/09_embedding_model.py
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from product_search import config
from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate, paired_bootstrap
from product_search.evaluation.runs import build_frame, load_dataset, ranked_run, retrieval_run, save_json
from product_search.ranking.features import FEATURES

BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "
SEEDS = [42, 7, 123]
CURRENT = "MiniLM, texto curto (atual)"


class BgeDense:
    def __init__(self, model, embeddings):
        self.model, self.embeddings, self.cache = model, embeddings, {}

    def scores(self, query):
        if query not in self.cache:
            self.cache[query] = self.model.encode([BGE_QUERY + query], normalize_embeddings=True)[0]
        return self.embeddings @ self.cache[query]


def embed(model, texts, name):
    path = config.ARTIFACTS / f"embeddings_{name}.npy"
    if not path.exists():
        np.save(path, model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True))
    return np.load(path)


def groups(frame):
    return frame.groupby("query_id", sort=False).size().to_numpy()


def fit_predict(fit, stop, ev, seed):
    model = lgb.LGBMRanker(**config.LGBM | {"random_state": seed})
    model.fit(
        fit[FEATURES], fit.label, group=groups(fit),
        eval_set=[(stop[FEATURES], stop.label)], eval_group=[groups(stop)], eval_at=[10],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING, verbose=False)],
    )  # fmt: skip
    return model.predict(ev[FEATURES])


def cross_validated(frame, qrels):
    runs = []
    for seed in SEEDS:
        scores = np.zeros(len(frame))
        for fit_idx, eval_idx in GroupKFold(5).split(frame, groups=frame.query_id):
            fit = frame.iloc[fit_idx]
            a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(fit, groups=fit.query_id))
            scores[eval_idx] = fit_predict(fit.iloc[a], fit.iloc[b], frame.iloc[eval_idx], seed)
        runs.append(evaluate(ranked_run(frame, scores), qrels))
    return sum(runs) / len(runs)


def on_validation(frame, ids, qrels):
    train, val = frame[frame.query_id.isin(ids["train"])], frame[frame.query_id.isin(ids["val"])]
    return sum(evaluate(ranked_run(val, fit_predict(train, val, val, s)), qrels) for s in SEEDS) / len(SEEDS)


def encode_ms(model, queries, prefix=""):
    times = []
    for q in queries:
        t0 = time.perf_counter()
        model.encode([prefix + q], normalize_embeddings=True)
        times.append(1000 * (time.perf_counter() - t0))
    return {"p50_ms": round(np.median(times), 1), "p95_ms": round(np.percentile(times, 95), 1)}


def main():
    data = load_dataset()
    current = SearchEngine.load(with_ranker=False)
    p = current.products
    model = SentenceTransformer(BGE, device="cpu")
    long_text = p.dense_text + ". " + p.product_description + ". " + p.attributes
    texts = {"bge-small, texto curto": p.dense_text, "bge-small, texto longo": long_text}
    engines = {CURRENT: current}
    for (name, text), file in zip(texts.items(), ["bge_short", "bge_long"], strict=True):
        engines[name] = SearchEngine(p, current.bm25, BgeDense(model, embed(model, text, file)), None)

    sample = data.queries["val"]["query"].tolist()[:50]
    latency = {"MiniLM": encode_ms(current.dense.model, sample), "bge-small": encode_ms(model, sample, BGE_QUERY)}

    retrieval = {}
    for split in ["train", "val"]:
        per_query = {
            (name, method): evaluate(retrieval_run(engine, data.queries[split], method), data.qrels)
            for name, engine in engines.items()
            for method in ["dense", "hybrid"]
        }
        retrieval[split] = {
            f"{name} / {method}": {metric: round(df[metric].mean(), 4) for metric in ["recall@100", "ndcg@10"]}
            | {
                f"{metric}_vs_atual": paired_bootstrap(per_query[(CURRENT, method)][metric], df[metric])
                for metric in ["recall@100", "ndcg@10"]
            }
            for (name, method), df in per_query.items()
        }

    queries = pd.concat([data.queries["train"], data.queries["val"]])
    ids = {s: set(data.queries[s].query_id) for s in ["train", "val"]}
    cv, val = {}, {}
    for name, engine in engines.items():
        frame = build_frame(engine, queries, data.qrels, "hybrid")
        cv[name], val[name] = cross_validated(frame, data.qrels), on_validation(frame, ids, data.qrels)
    ranking = {
        name: {
            "cv_ndcg@10": round(cv[name]["ndcg@10"].mean(), 4),
            "cv_vs_atual": paired_bootstrap(cv[CURRENT]["ndcg@10"], cv[name]["ndcg@10"]),
            "cv_condensed_vs_atual": paired_bootstrap(cv[CURRENT]["condensed_ndcg@10"], cv[name]["condensed_ndcg@10"]),
            "val_ndcg@10": round(val[name]["ndcg@10"].mean(), 4),
            "val_vs_atual": paired_bootstrap(val[CURRENT]["ndcg@10"], val[name]["ndcg@10"]),
        }
        for name in engines
    }
    save_json("analysis/09_embedding_model.json", {"encode_latency": latency, "retrieval": retrieval, "ltr": ranking})


if __name__ == "__main__":
    main()
