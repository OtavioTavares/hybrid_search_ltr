"""Compara BM25, denso e híbrido e escolhe pelo Recall@100 na validação -> reports/retrieval/metrics.json.

Uso: uv run eval-retrieval
"""

from product_search.engine import SearchEngine
from product_search.evaluation.metrics import evaluate
from product_search.evaluation.runs import load_dataset, mean, retrieval_run, save_json


def main():
    data = load_dataset()
    engine = SearchEngine.load(with_ranker=False)
    metrics = {
        split: {m: mean(evaluate(retrieval_run(engine, data.queries[split], m), data.qrels)) for m in ["bm25", "dense", "hybrid"]}
        for split in ["train", "val"]
    }
    selected = max(metrics["val"], key=lambda m: metrics["val"][m]["recall@100"])
    save_json("retrieval/metrics.json", {"selected": selected} | metrics)


if __name__ == "__main__":
    main()
