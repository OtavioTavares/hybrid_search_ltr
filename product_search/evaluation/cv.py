import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from product_search import config

N_TREES = 80  # fixo: escolher por early stopping usaria as partes de avaliação


def cross_val_scores(frame: pd.DataFrame, features: list[str], n_splits: int = 5) -> np.ndarray:
    """Scores fora da amostra para cada linha, com validação cruzada por query."""
    scores = np.zeros(len(frame))
    params = config.LGBM | {"n_estimators": N_TREES}
    for fit_idx, eval_idx in GroupKFold(n_splits=n_splits).split(frame, groups=frame.query_id):
        fit = frame.iloc[fit_idx]
        model = lgb.LGBMRanker(**params)
        model.fit(fit[features], fit.label, group=fit.groupby("query_id", sort=False).size().to_numpy())
        scores[eval_idx] = model.predict(frame.iloc[eval_idx][features])
    return scores
