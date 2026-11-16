import math

import pytest

from product_search.evaluation.metrics import condensed_ndcg, judged, mrr, ndcg, paired_bootstrap, recall

LABELS = {1: "Exact", 2: "Partial", 3: "Irrelevant"}


def test_ndcg_perfect_and_worst():
    assert ndcg([1, 2, 3], LABELS) == pytest.approx(1.0)
    assert ndcg([3, 9], LABELS) == 0.0


def test_ndcg_by_hand():
    # ganhos 2,1; ordem invertida: (1 + 2/log2(3)) / (2 + 1/log2(3))
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg([2, 1], LABELS) == pytest.approx(expected)


def test_unjudged_counts_as_irrelevant_but_condensed_ignores_it():
    assert ndcg([9, 1, 2], LABELS) < 1.0
    assert condensed_ndcg([9, 1, 2], LABELS) == pytest.approx(1.0)


def test_recall_mrr_judged():
    assert recall([9, 1], LABELS, k=2) == 1.0
    assert math.isnan(recall([1], {1: "Partial"}))
    assert mrr([3, 2, 1], LABELS) == pytest.approx(1 / 3)
    assert judged([1, 9, 2, 8], LABELS, k=4) == 0.5


def test_bootstrap_detects_consistent_gain():
    out = paired_bootstrap([0.1] * 50, [0.3] * 50)
    assert out["diff"] == pytest.approx(0.2)
    assert out["ci95"][0] > 0
