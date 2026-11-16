import pandas as pd
from sklearn.model_selection import train_test_split

from product_search import config


def split_queries(labels: pd.DataFrame, query_ids, seed: int = config.SEED) -> pd.DataFrame:
    """Split por query, estratificado pela fração de Exact (quartis), para os sets terem dificuldade parecida."""
    exact_share = labels.assign(exact=labels.label.eq("Exact")).groupby("query_id").exact.mean()
    exact_share = exact_share.reindex(query_ids)
    strata = pd.qcut(exact_share.rank(method="first"), 4, labels=False)

    holdout = config.SPLIT["val"] + config.SPLIT["test"]
    train, rest = train_test_split(exact_share.index, test_size=holdout, stratify=strata, random_state=seed)
    test_frac = config.SPLIT["test"] / holdout
    val, test = train_test_split(rest, test_size=test_frac, stratify=strata[rest], random_state=seed)

    parts = [pd.DataFrame({"query_id": ids, "split": name}) for name, ids in [("train", train), ("val", val), ("test", test)]]
    return pd.concat(parts).sort_values("query_id").reset_index(drop=True)
