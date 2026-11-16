import lightgbm as lgb
import numpy as np
import pandas as pd

from product_search import config


class Ranker:
    """LambdaMART (LightGBM). Espera frames com query_id contíguo e coluna `label` (0/1/2)."""

    def __init__(self, features: list[str], retriever: str = "hybrid"):
        self.features, self.retriever = features, retriever

    def _prepare(self, frame: pd.DataFrame):
        groups = frame.groupby("query_id", sort=False).size().to_numpy()
        return frame[self.features], frame.label.to_numpy(), groups

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        X, y, g = self._prepare(train)
        Xv, yv, gv = self._prepare(val)
        self.model = lgb.LGBMRanker(**config.LGBM)
        self.model.fit(
            X,
            y,
            group=g,
            eval_X=(Xv,),
            eval_y=(yv,),
            eval_group=[gv],
            eval_at=[10],
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING, verbose=False)],
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.features])

    def importance(self) -> dict:
        gain = self.model.booster_.feature_importance("gain")
        return dict(sorted(zip(self.features, gain.round(1).tolist(), strict=True), key=lambda x: -x[1]))
