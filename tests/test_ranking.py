import numpy as np
import pandas as pd

from product_search.ranking.features import FEATURES, build, product_table
from product_search.ranking.model import Ranker


def test_features_v1_and_expected_values(products):
    f = build("blue sofa 24", np.array([2, 0]), product_table(products), np.array([0.0, 1.0, 5.0]), np.array([0.1, 0.2, 0.9]))
    assert f.columns.tolist() == FEATURES
    assert f.name_overlap.tolist() == [2 / 3, 0.0]
    assert f.bm25.tolist() == [5.0, 0.0] and f.bm25_rank.tolist() == [0, 1]
    assert f.number_match.tolist() == [0.0, 0.0]  # 24 != 24.75: a v1 compara o número como texto
    assert np.isnan(f.average_rating).sum() == 0 and f.n_attributes.tolist() == [2, 2]


def test_number_match_is_missing_when_query_has_no_number(products):
    assert build("blue sofa", np.array([2]), product_table(products), np.zeros(3), np.zeros(3)).number_match.isna().all()


def test_ranker_learns_a_simple_signal():
    rng = np.random.default_rng(0)
    n_q, n_c = 40, 20
    x = rng.random(n_q * n_c)
    frame = pd.DataFrame(
        {
            "query_id": np.repeat(range(n_q), n_c),
            "x": x,
            "noise": rng.random(n_q * n_c),
            "label": (x > 0.7).astype(int) + (x > 0.9),
        }
    )
    train, val = frame[frame.query_id < 30], frame[frame.query_id >= 30]
    scores = Ranker(["x", "noise"]).fit(train, val).predict(val)
    assert np.corrcoef(scores, val.x)[0, 1] > 0.8
