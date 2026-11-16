import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WANDS_DIR", ROOT / "dataset"))
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SEED = 42
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2}
GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

MAX_QUERY_CHARS = 200
BM25_K1, BM25_B = 1.2, 0.75
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # commit fixo no Hugging Face
RRF_K = 60
N_CANDIDATES = 100

LGBM = dict(
    objective="lambdarank",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)
EARLY_STOPPING = 50
