import numpy as np

from product_search.ranking import experimental
from product_search.ranking.experimental import build, product_table


def test_v2_features(products):
    cand = np.array([2, 0])
    f = build("blue sofa 24", cand, product_table(products), np.array([0.0, 1.0, 5.0]), np.array([0.1, 0.2, 0.9]))
    assert f.bm25_ratio.tolist() == [1.0, 0.0] and np.allclose(f.cosine_gap, [0.4, -0.4])
    assert f.class_share.tolist() == [0.5, 0.5]
    assert f.number_close.tolist() == [1.0, 0.0]  # 24 está a menos de 10% de 24.75


def test_attribute_kinds_look_at_the_right_keys(products, monkeypatch):
    monkeypatch.setattr(experimental, "MIN_PRODUCTS_IN_VOCAB", 1)
    f = build("gold velvet chair", np.array([0, 2]), product_table(products), np.zeros(3), np.zeros(3))
    assert f.color_match.tolist() == [1.0, 0.0]  # 'legcolor' conta como cor
    assert f.material_match.tolist() == [1.0, 0.0]
    assert f.style_match.isna().all()  # a busca não pede estilo


def test_attribute_vocab_keeps_only_frequent_single_word_values(products, monkeypatch):
    monkeypatch.setattr(experimental, "MIN_PRODUCTS_IN_VOCAB", 2)
    catalog = products.assign(
        product_features=["color : black|material : chair cushion", "color : black", "material : chair cushion"]
    )
    vocab = product_table(catalog)["vocab"]
    assert vocab["color"] == {"black"}
    assert "chair" not in vocab["material"]


def test_v3_corrections(products):
    catalog = products.assign(
        product_name=["party chair with art print", "wall art with frame", "blue sofa with pillows"],
        product_features=["countryoforigin : china|color : gold", "", "color : blue"],
    )
    table = product_table(catalog)
    f = build("art", np.array([0, 1]), table, np.zeros(3), np.zeros(3))
    assert f.query_in_name.tolist() == [1.0, 1.0]  # v1: 'art' casa dentro de 'party'
    assert f.query_in_name_tokens.tolist() == [1.0, 1.0]  # 'art' é palavra inteira nos dois
    assert build("par", np.array([0]), table, np.zeros(3), np.zeros(3)).query_in_name_tokens.tolist() == [0.0]
    assert "china" not in table["attributes_desc"][0] and "gold" in table["attributes_desc"][0]
    g = build("with zzz", np.array([0]), table, np.zeros(3), np.zeros(3))
    assert g.oov_share.tolist() == [0.5]
    # 'with' é comum no catálogo e 'zzz' não existe: casar só 'with' vale menos que metade no overlap com IDF
    assert g.name_overlap.tolist() == [0.5] and g.name_overlap_idf.iloc[0] < 0.5


def test_bayes_rating_shrinks_few_reviews(products):
    catalog = products.assign(rating_count=[1, np.nan, 1000], average_rating=[5.0, np.nan, 4.6])
    rating = product_table(catalog)["rating_bayes"]
    assert rating[0] < 5.0 and abs(rating[2] - 4.6) < 0.01 and np.isnan(rating[1])
