import re

import numpy as np
import pandas as pd

from product_search.text import tokenize

GROUPS = {
    "retrieval": ["bm25", "cosine", "bm25_rank", "dense_rank", "retrieval_rank"],
    "text": ["name_overlap", "class_overlap", "category_overlap", "query_in_name", "query_len"],
    "attributes": ["attr_overlap", "number_match"],
    "numeric": ["log_rating_count", "average_rating", "log_review_count", "n_attributes", "log_desc_len"],
}
FEATURES = sum(GROUPS.values(), [])
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _token_sets(column: pd.Series) -> list[set[str]]:
    return [set(tokenize(t)) for t in column]


def product_table(products: pd.DataFrame) -> dict:
    """Pré-calcula o que não depende da query."""
    return {
        "name": _token_sets(products.product_name),
        "class": _token_sets(products.product_class),
        "category": _token_sets(products.category_hierarchy),
        "attributes": _token_sets(products.attributes),
        "name_lower": products.product_name.str.lower().tolist(),
        "numbers": [set(NUMBER.findall(f"{n} {a}")) for n, a in zip(products.product_name, products.attributes, strict=True)],
        "numeric": pd.DataFrame(
            {
                "log_rating_count": np.log1p(products.rating_count),
                "average_rating": products.average_rating,  # NaN = sem avaliação; o LightGBM trata nativamente
                "log_review_count": np.log1p(products.review_count),
                "n_attributes": products.product_features.str.count(r"\|") + products.product_features.ne("").astype(int),
                "log_desc_len": np.log1p(products.product_description.str.len()),
            }
        ).to_numpy(),
    }


def rank_within(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-values, kind="stable"), kind="stable")


def overlap(q: set[str], token_sets: list[set[str]], candidates: np.ndarray) -> list[float]:
    """Fração das palavras da busca presentes no campo do produto."""
    return [len(q & token_sets[i]) / max(len(q), 1) for i in candidates]


def _retrieval(candidates, bm25, cosine) -> dict:
    s_bm25, s_cos = bm25[candidates], cosine[candidates]
    return {
        "bm25": s_bm25,
        "cosine": s_cos,
        "bm25_rank": rank_within(s_bm25),
        "dense_rank": rank_within(s_cos),
        "retrieval_rank": np.arange(len(candidates)),
    }


def _text(query, q, candidates, table) -> dict:
    q_lower = query.lower()
    return {
        "name_overlap": overlap(q, table["name"], candidates),
        "class_overlap": overlap(q, table["class"], candidates),
        "category_overlap": overlap(q, table["category"], candidates),
        "query_in_name": [float(q_lower in table["name_lower"][i]) for i in candidates],
        "query_len": np.full(len(candidates), len(q)),
    }


def _attributes(query, q, candidates, table) -> dict:
    numbers = set(NUMBER.findall(query))
    return {
        "attr_overlap": overlap(q, table["attributes"], candidates),
        "number_match": [len(numbers & table["numbers"][i]) / len(numbers) if numbers else np.nan for i in candidates],
    }


def _numeric(candidates, table) -> dict:
    values = table["numeric"][candidates]
    return {name: values[:, j] for j, name in enumerate(GROUPS["numeric"])}


def build(query: str, candidates: np.ndarray, table: dict, bm25: np.ndarray, cosine: np.ndarray) -> pd.DataFrame:
    """Uma linha por candidato. `bm25`/`cosine` são os scores do catálogo inteiro, já calculados no retrieval."""
    q = set(tokenize(query))
    return pd.DataFrame(
        _retrieval(candidates, bm25, cosine)
        | _text(query, q, candidates, table)
        | _attributes(query, q, candidates, table)
        | _numeric(candidates, table)
    )
