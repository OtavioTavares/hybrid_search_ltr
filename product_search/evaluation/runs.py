import json
from dataclasses import dataclass

import pandas as pd

from product_search import config
from product_search.data.loading import load_labels, load_queries, to_qrels
from product_search.ranking import experimental


@dataclass
class Dataset:
    qrels: dict
    queries: dict[str, pd.DataFrame]


def load_dataset() -> Dataset:
    """Rótulos limpos e queries de cada split, lidos de reports/data/splits.csv."""
    path = config.REPORTS / "data" / "splits.csv"
    if not path.exists():
        raise SystemExit("rode antes: uv run split-queries")
    labels = load_labels()
    queries = load_queries(labels)
    splits = pd.read_csv(path)
    by_split = {s: queries[queries.query_id.isin(splits.query_id[splits.split == s])] for s in config.SPLIT}
    return Dataset(to_qrels(labels), by_split)


def retrieval_run(engine, queries: pd.DataFrame, method: str) -> dict:
    pids = engine.products.product_id.to_numpy()
    pairs = zip(queries.query_id, queries["query"], strict=True)
    return {qid: pids[engine.retrieve(q, method).candidates].tolist() for qid, q in pairs}


def _featurizer(engine, with_experimental: bool):
    if not with_experimental:
        return engine.features
    table = experimental.product_table(engine.products)
    return lambda q, r: experimental.build(q, r.candidates, table, r.bm25, r.cosine)


def build_frame(engine, queries: pd.DataFrame, qrels: dict, method: str, with_experimental: bool = False) -> pd.DataFrame:
    """Candidatos do retrieval + features + rótulo (0/1/2; sem julgamento = 0).

    `with_experimental` acrescenta as features não adotadas (ranking/experimental.py), para os estudos.
    """
    pids = engine.products.product_id.to_numpy()
    featurize = _featurizer(engine, with_experimental)
    frames = []
    for qid, q in zip(queries.query_id, queries["query"], strict=True):
        r = engine.retrieve(q, method)
        labels = [qrels[qid].get(p) for p in pids[r.candidates]]
        frames.append(
            featurize(q, r).assign(
                query_id=qid,
                product_id=pids[r.candidates],
                judged=[label is not None for label in labels],
                label=[config.GAIN.get(label, 0) for label in labels],
            )
        )
    return pd.concat(frames, ignore_index=True)


def ranked_run(frame: pd.DataFrame, scores=None) -> dict:
    """Sem scores: mantém a ordem do retrieval."""
    if scores is not None:
        frame = frame.assign(s=scores).sort_values(["query_id", "s"], ascending=[True, False], kind="stable")
    return {qid: g.product_id.tolist() for qid, g in frame.groupby("query_id", sort=False)}


def mean(per_query: pd.DataFrame) -> dict:
    return per_query.mean().round(4).to_dict()


def save_json(name: str, data: dict) -> None:
    """`name` é relativo a reports/, ex.: 'ranking/metrics.json'."""
    path = config.REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=1, ensure_ascii=False))


def save_csv(name: str, frame: pd.DataFrame) -> None:
    path = config.REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def load_json(name: str) -> dict:
    return json.loads((config.REPORTS / name).read_text(encoding="utf-8"))
