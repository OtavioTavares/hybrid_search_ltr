import pandas as pd

from product_search.data.split import split_queries


def make_labels(n_queries=100):
    rows = [(q, p, "Exact" if p < q % 5 else "Partial") for q in range(n_queries) for p in range(5)]
    return pd.DataFrame(rows, columns=["query_id", "product_id", "label"])


def test_split_is_disjoint_complete_and_sized():
    s = split_queries(make_labels(), range(100))
    assert s.query_id.is_unique and set(s.query_id) == set(range(100))
    assert s.split.value_counts().to_dict() == {"train": 60, "val": 20, "test": 20}


def test_split_is_stratified_by_exact_share():
    s = split_queries(make_labels(), range(100)).set_index("query_id").split
    by_share = sorted(range(100), key=lambda q: (q % 5, q))  # fração de Exact = (q % 5) / 5
    for quartile in [by_share[:25], by_share[25:50], by_share[50:75], by_share[75:]]:
        counts = s[quartile].value_counts().to_dict()
        assert counts == {"train": 15, "val": 5, "test": 5}


def test_split_is_reproducible():
    a, b = split_queries(make_labels(), range(100)), split_queries(make_labels(), range(100))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(split_queries(make_labels(), range(100), seed=1))
