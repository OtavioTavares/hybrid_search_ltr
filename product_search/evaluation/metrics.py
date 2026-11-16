import numpy as np
import pandas as pd

from product_search import config


def _dcg(gains) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg(ranked, labels: dict, k: int = 10) -> float:
    """O ideal usa todos os julgados da query, então o que o retrieval perdeu também pesa."""
    ideal = _dcg(sorted((config.GAIN[label] for label in labels.values()), reverse=True)[:k])
    gains = [config.GAIN.get(labels.get(p, "Irrelevant"), 0) for p in ranked[:k]]
    return _dcg(gains) / ideal if ideal > 0 else 0.0


def condensed_ndcg(ranked, labels: dict, k: int = 10) -> float:
    """nDCG ignorando produtos sem julgamento: mostra se um ganho vem só de trocar não julgados por julgados."""
    return ndcg([p for p in ranked if p in labels], labels, k)


def recall(ranked, labels: dict, k: int = 100) -> float:
    relevant = {p for p, label in labels.items() if label == "Exact"}
    return len(relevant.intersection(ranked[:k])) / len(relevant) if relevant else np.nan


def mrr(ranked, labels: dict, k: int = 10) -> float:
    return next((1 / (i + 1) for i, p in enumerate(ranked[:k]) if labels.get(p) == "Exact"), 0.0)


def judged(ranked, labels: dict, k: int = 10) -> float:
    return sum(p in labels for p in ranked[:k]) / k


def evaluate(run: dict, qrels: dict) -> pd.DataFrame:
    """run = {query_id: [product_ids ordenados]} -> uma linha por query."""
    rows = {
        qid: {
            "ndcg@10": ndcg(ranked, qrels[qid]),
            "condensed_ndcg@10": condensed_ndcg(ranked, qrels[qid]),
            "mrr@10": mrr(ranked, qrels[qid]),
            "recall@100": recall(ranked, qrels[qid]),
            "judged@10": judged(ranked, qrels[qid]),
        }
        for qid, ranked in run.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def paired_bootstrap(a, b, n: int = 2000, seed: int = config.SEED) -> dict:
    """Diferença média b - a por query, com IC 95%. Queries com métrica indefinida ficam de fora."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    diff = diff[~np.isnan(diff)]
    means = np.random.default_rng(seed).choice(diff, (n, len(diff))).mean(axis=1)
    return {"diff": round(diff.mean(), 4), "ci95": [round(np.percentile(means, 2.5), 4), round(np.percentile(means, 97.5), 4)]}
