import pandas as pd
import pytest

from product_search import config
from product_search.data.loading import load_labels, load_queries, to_qrels


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    labels = [
        (0, 0, 1, "Exact"),
        (1, 0, 1, "Exact"),  # repetido
        (2, 0, 2, "Partial"),
        (3, 0, 2, "Irrelevant"),  # conflita com o anterior
        (4, 1, 3, "Irrelevant"),  # busca 1 só tem Irrelevant
        (5, 2, 4, "Partial"),
    ]
    queries = [(0, "gold chair", "Chairs"), (1, "blue sofa", "Sofas"), (2, "wood table", "Tables")]
    pd.DataFrame(labels, columns=["id", "query_id", "product_id", "label"]).to_csv(tmp_path / "label.csv", sep="\t", index=False)
    pd.DataFrame(queries, columns=["query_id", "query", "query_class"]).to_csv(tmp_path / "query.csv", sep="\t", index=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_labels_drop_duplicates_and_conflicts(data_dir):
    labels = load_labels()
    pairs = set(zip(labels.query_id, labels.product_id, labels.label, strict=True))
    assert pairs == {(0, 1, "Exact"), (1, 3, "Irrelevant"), (2, 4, "Partial")}


def test_queries_without_exact_or_partial_are_dropped(data_dir):
    assert load_queries(load_labels()).query_id.tolist() == [0, 2]


def test_qrels_by_query(data_dir):
    assert to_qrels(load_labels()) == {0: {1: "Exact"}, 1: {3: "Irrelevant"}, 2: {4: "Partial"}}
