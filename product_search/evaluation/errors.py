import numpy as np
import pandas as pd

from product_search.evaluation.metrics import evaluate, ndcg


def error_analysis(frame: pd.DataFrame, run: dict, qrels: dict, queries: pd.DataFrame, products: pd.DataFrame):
    """Oráculo = melhor ordem possível dos candidatos. 1 - oráculo é perda do retrieval; oráculo - sistema, do ranking."""
    product_class = products.set_index("product_id").product_class
    text = queries.set_index("query_id")["query"]
    rows = []
    for qid, ranked in run.items():
        best = frame[frame.query_id == qid].sort_values("label", ascending=False).product_id.tolist()
        rows.append(
            {
                "query_id": qid,
                "query": text[qid],
                "n_exact": sum(label == "Exact" for label in qrels[qid].values()),
                "oracle_ndcg@10": ndcg(best, qrels[qid]),
                "top3_classes": " | ".join(product_class[ranked[:3]].astype(str)),
            }
        )
    df = pd.DataFrame(rows).set_index("query_id").join(evaluate(run, qrels)).round(3).sort_values("ndcg@10")
    df["failure"] = np.select([df.n_exact == 0, df["recall@100"] < 0.3], ["no_exact_label", "retrieval"], "ranking")

    loss_retrieval = (1 - df["oracle_ndcg@10"]).sum()
    loss_ranking = (df["oracle_ndcg@10"] - df["ndcg@10"]).sum()
    total = loss_retrieval + loss_ranking
    summary = {
        "oracle_ndcg@10": round(df["oracle_ndcg@10"].mean(), 4),
        "loss_share_retrieval": round(loss_retrieval / total, 3),
        "loss_share_ranking": round(loss_ranking / total, 3),
        "worst_queries_failure": df[df["ndcg@10"] < 0.5].failure.value_counts().to_dict(),
    }
    return df, summary
