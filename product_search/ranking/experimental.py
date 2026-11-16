"""Features testadas e não adotadas (v2 e v3). Usadas só pelos estudos em analysis/."""

from collections import Counter

import numpy as np
import pandas as pd

from product_search.ranking import features
from product_search.ranking.features import NUMBER, overlap
from product_search.text import tokenize

V2 = {
    "normalized": ["bm25_ratio", "cosine_gap"],
    "class": ["class_share"],
    "attributes novos": ["number_close", "color_match", "material_match", "style_match"],
}
V3 = {
    "query_in_name por palavra inteira": {"query_in_name": "query_in_name_tokens"},
    "overlaps pesados por IDF": {
        "name_overlap": "name_overlap_idf",
        "class_overlap": "class_overlap_idf",
        "category_overlap": "category_overlap_idf",
    },
    "nota média bayesiana": {"average_rating": "rating_bayes"},
    "atributos só descritivos": {"attr_overlap": "attr_overlap_desc"},
}
ATTRIBUTE_KINDS = ["color", "material", "style"]
_ADMIN_KEYS = (
    "warranty",
    "countryoforigin",
    "compliant",
    "certif",
    "proposition",
    "assembly",
    "regulation",
    "cwp",
    "gcc",
    "tsca",
    "carb",
    "canada",
    "installation",
    "productcare",
    "supplier",
    "uplr",
    "restriction",
    "flameretardant",
)
_IGNORED = {"and", "with", "the", "for", "other", "yes", "no", "not", "applicable"}
RATING_PRIOR_COUNT = 10
MIN_PRODUCTS_IN_VOCAB = 50


def _attribute_tokens(raw: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Tokens dos valores de cada tipo de atributo ('legcolor', 'basecolor'... entram em 'color').

    Devolve também os valores de uma palavra só, que formam o vocabulário do tipo: 'black' é cor,
    mas 'chair' (de 'chair cushion', um material) não é.
    """
    tokens = {kind: set() for kind in ATTRIBUTE_KINDS}
    single = {kind: set() for kind in ATTRIBUTE_KINDS}
    for pair in raw.split("|"):
        key, _, value = pair.partition(":")
        value_tokens = [t for t in tokenize(value) if t not in _IGNORED and not t.isdigit()]
        for kind in ATTRIBUTE_KINDS:
            if kind in key:
                tokens[kind].update(value_tokens)
                if len(value_tokens) == 1 and len(tokenize(value)) == 1:
                    single[kind].add(value_tokens[0])
    return tokens, single


def _vocab(singles: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    counts = {kind: Counter(t for s in singles for t in s[kind]) for kind in ATTRIBUTE_KINDS}
    return {kind: {t for t, n in c.items() if n >= MIN_PRODUCTS_IN_VOCAB} for kind, c in counts.items()}


def _descriptive_attributes(raw: str) -> set[str]:
    """Tokens dos atributos que descrevem o produto, sem garantia, origem, certificações etc."""
    return {
        t
        for pair in raw.split("|")
        if not any(a in pair.partition(":")[0] for a in _ADMIN_KEYS)
        for t in tokenize(pair.partition(":")[2])
    }


def _idf(token_sets: list[set[str]]) -> dict[str, float]:
    df = Counter(t for s in token_sets for t in s)
    n = len(token_sets)
    return {t: float(np.log((1 + n) / (1 + c))) for t, c in df.items()}


def _bayes_rating(products: pd.DataFrame) -> pd.Series:
    """Média bayesiana: com poucas avaliações, a nota é puxada para a média do catálogo."""
    prior = products.average_rating.mean()
    n = products.rating_count
    return (RATING_PRIOR_COUNT * prior + n * products.average_rating) / (RATING_PRIOR_COUNT + n)


def product_table(products: pd.DataFrame) -> dict:
    table = features.product_table(products)
    attributes, singles = zip(*(_attribute_tokens(raw) for raw in products.product_features), strict=True)
    description = [set(tokenize(t)) for t in products.product_description]
    fields = zip(table["name"], table["class"], table["category"], description, strict=True)
    return table | {
        "name_tokens": [tokenize(t) for t in products.product_name],
        "classes": products.product_class.str.split("|").map(lambda cs: {c.strip() for c in cs if c.strip()}).tolist(),
        "kinds": {kind: [a[kind] for a in attributes] for kind in ATTRIBUTE_KINDS},
        "vocab": _vocab(singles),
        "idf": _idf([n | k | c | d for n, k, c, d in fields]),
        "attributes_desc": [_descriptive_attributes(raw) for raw in products.product_features],
        "rating_bayes": _bayes_rating(products).to_numpy(),
    }


def _numbers_close(query_numbers: list[str], product_numbers: set[str]) -> float:
    """Fração dos números da busca com algum número do produto a até 10% (mín. 0,5) de distância: 24 casa com 24.75."""
    if not query_numbers:
        return np.nan
    if not product_numbers:
        return 0.0
    values = np.array([float(n) for n in product_numbers])
    return float(np.mean([np.abs(values - float(n)).min() <= max(0.5, 0.1 * float(n)) for n in query_numbers]))


def _contains_sequence(tokens: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n > 0 and any(tokens[i : i + n] == sub for i in range(len(tokens) - n + 1))


def _class_share(candidates: np.ndarray, classes: list[set], depth: int = 20) -> list[float]:
    """Fração dos `depth` primeiros candidatos que são da mesma classe do produto (consenso do retrieval)."""
    counts = Counter(c for i in candidates[:depth] for c in classes[i])
    n = min(depth, len(candidates))
    return [max((counts[c] for c in classes[i]), default=0) / n for i in candidates]


def _v2(query, q, candidates, table, bm25, cosine) -> dict:
    s_bm25, s_cos = bm25[candidates], cosine[candidates]
    q_numbers = sorted(set(NUMBER.findall(query)))

    def kind_match(kind):
        asked = q & table["vocab"][kind]
        if not asked:
            return np.full(len(candidates), np.nan)
        return [len(asked & table["kinds"][kind][i]) / len(asked) for i in candidates]

    return {
        "bm25_ratio": s_bm25 / s_bm25.max() if s_bm25.max() > 0 else np.zeros(len(candidates)),
        "cosine_gap": s_cos - s_cos.mean(),
        "class_share": _class_share(candidates, table["classes"]),
        "number_close": [_numbers_close(q_numbers, table["numbers"][i]) for i in candidates],
        **{f"{kind}_match": kind_match(kind) for kind in ATTRIBUTE_KINDS},
    }


def _v3(query, q, candidates, table) -> dict:
    q_list, idf = tokenize(query), table["idf"]
    max_idf = max(idf.values(), default=1.0)
    weight = {t: idf.get(t, max_idf) for t in q}
    total = sum(weight.values()) or 1.0

    def overlap_idf(field):
        return [sum(weight[t] for t in q & table[field][i]) / total for i in candidates]

    return {
        "query_in_name_tokens": [float(_contains_sequence(table["name_tokens"][i], q_list)) for i in candidates],
        "name_overlap_idf": overlap_idf("name"),
        "class_overlap_idf": overlap_idf("class"),
        "category_overlap_idf": overlap_idf("category"),
        "rating_bayes": table["rating_bayes"][candidates],
        "oov_share": np.full(len(candidates), sum(t not in idf for t in q) / max(len(q), 1)),
        "attr_overlap_desc": overlap(q, table["attributes_desc"], candidates),
    }


def build(query: str, candidates: np.ndarray, table: dict, bm25: np.ndarray, cosine: np.ndarray) -> pd.DataFrame:
    """Features da v1 + as experimentais, com a tabela de `product_table` deste módulo."""
    q = set(tokenize(query))
    extra = _v2(query, q, candidates, table, bm25, cosine) | _v3(query, q, candidates, table)
    return features.build(query, candidates, table, bm25, cosine).assign(**extra)
