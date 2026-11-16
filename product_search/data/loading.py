import pandas as pd

from product_search import config

TEXT_COLS = ["product_name", "product_class", "category_hierarchy", "product_description", "product_features"]


def attribute_values(raw: str) -> str:
    """'color : gold|material : wood|width:30' -> 'gold wood 30' (sem os yes/no)."""
    values = (pair.split(":", 1)[-1].strip() for pair in raw.split("|") if pair)
    return " ".join(v for v in values if v not in {"yes", "no"})


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / name, sep="\t")


def load_products() -> pd.DataFrame:
    df = _read("product.csv").rename(columns={"category hierarchy": "category_hierarchy"})
    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df["attributes"] = df.product_features.map(attribute_values)
    df["lexical_text"] = df[["product_name", "product_class", "category_hierarchy", "product_description"]].agg(" ".join, axis=1)
    df["dense_text"] = df.product_name + ". " + df.product_class + ". " + df.category_hierarchy
    return df.reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    """Remove pares (query, produto) repetidos e descarta os que têm rótulos conflitantes."""
    labels = _read("label.csv")[["query_id", "product_id", "label"]]
    n_labels = labels.groupby(["query_id", "product_id"]).label.transform("nunique")
    return labels[n_labels == 1].drop_duplicates(["query_id", "product_id"]).reset_index(drop=True)


def load_queries(labels: pd.DataFrame) -> pd.DataFrame:
    """Só queries com ao menos um Exact ou Partial: sem isso o nDCG é indefinido."""
    useful = labels.loc[labels.label != "Irrelevant", "query_id"].unique()
    queries = _read("query.csv")
    return queries[queries.query_id.isin(useful)].reset_index(drop=True)


def to_qrels(labels: pd.DataFrame) -> dict[int, dict[int, str]]:
    qrels: dict[int, dict[int, str]] = {}
    for qid, pid, label in labels.itertuples(index=False):
        qrels.setdefault(qid, {})[pid] = label
    return qrels
