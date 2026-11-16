import numpy as np
import pytest

from product_search.data.loading import attribute_values
from product_search.engine import SearchEngine, check_k, clean_query
from product_search.retrieval.bm25 import BM25
from product_search.retrieval.fusion import rrf, top_k
from product_search.text import tokenize

DOCS = ["blue velvet sofa", "wooden dining table", "outdoor dining chairs set", "velvet accent chair"]


def test_tokenize_lowercases_and_strips_plural():
    assert tokenize("Dining CHAIRS, glass!") == ["dining", "chair", "glass"]


def test_attribute_values_drop_keys_and_yes_no():
    assert attribute_values("color : gold|waterproof : yes|width:30") == "gold 30"


def test_bm25_finds_the_right_document():
    bm25 = BM25().fit(DOCS)
    assert top_k(bm25.scores("dining table"), 1)[0] == 1
    assert top_k(bm25.scores("velvet chair"), 1)[0] == 3
    assert bm25.scores("zzz").sum() == 0


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [2, 1, 4]], k=2).tolist() in ([1, 2], [2, 1])
    assert rrf([[1, 2], [3, 2]], k=1).tolist() == [2]


@pytest.mark.parametrize("bad, error", [("", ValueError), ("  \n\t ", ValueError), (None, TypeError), (42, TypeError)])
def test_invalid_queries(bad, error):
    with pytest.raises(error):
        clean_query(bad)


@pytest.mark.parametrize("bad", [0, -1, 101, 2.5, "5", True])
def test_invalid_k(bad):
    with pytest.raises(ValueError):
        check_k(bad)


def test_query_is_normalized_and_truncated():
    assert clean_query("  sofa\n\tbed ") == "sofa bed"
    assert len(clean_query("a" * 1000)) == 200


class FakeDense:
    def scores(self, query):
        return np.zeros(3, dtype=np.float32)


class OverlapRanker:
    features = ["name_overlap"]

    def predict(self, frame):
        return frame.name_overlap.to_numpy()


@pytest.fixture
def engine(products):
    products = products.assign(product_id=[10, 20, 30])
    return SearchEngine(products, BM25().fit(products.product_name), FakeDense(), OverlapRanker())


@pytest.mark.parametrize("bad, k", [("", 5), ("   ", 5), (None, 5), (42, 5), ("sofa", 0), ("sofa", -1), ("sofa", 101)])
def test_search_rejects_invalid_input(engine, bad, k):
    with pytest.raises((ValueError, TypeError)):
        engine.search(bad, k)


@pytest.mark.parametrize("query", ["!!!@@@", "😀"])
def test_search_without_words_returns_nothing(engine, query):
    assert engine.search(query)["results"] == []


def test_markup_is_searched_as_plain_text(engine):
    assert len(engine.search("<script>alert(1)</script> sofa")["results"]) == 3


def test_search_ranks_and_reports_latency(engine):
    out = engine.search("BLUE\u200b sofa 24" + " x" * 500, k=2)
    assert len(out["query"]) == 200
    assert [r["product_id"] for r in out["results"]][0] == 30 and len(out["results"]) == 2
    assert set(out["latency_ms"]) == {"retrieval", "ranking"}
