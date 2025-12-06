"""
XGBoost Ranker
==============
Learning-to-rank model using XGBoost (rank:pairwise or rank:ndcg).
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)

# Availability check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed: pip install xgboost")


class XGBoostRanker(BaseRecommender):
    """
    XGBoost-based ranking model.

    • Uses pairwise or NDCG ranking loss
    • Same interface as LightGBMRanker (fit, predict, predict_df, recommend)

    Example:
        >>> model = XGBoostRanker(max_depth=6, learning_rate=0.1)
        >>> model.fit(train_df, feature_names, valid_df)
        >>> scores = model.predict_df(test_df)
    """

    @property
    def name(self) -> str:
        return "XGBoost"

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "rank:ndcg",  # or "rank:ndcg"
        early_stopping_rounds: int = 50,
        verbosity: int = 0
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed: pip install xgboost")

        self.params = {
            "objective": objective,
            "eval_metric": "ndcg",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "verbosity": verbosity,
            "seed": 42,
        }

        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self.is_ml_model = True

        logger.info(f"XGBoostRanker initialized (max_depth={max_depth}, lr={learning_rate})")

    # ----------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        valid_df: pd.DataFrame = None,
        **kwargs
    ):
        """Train model using XGBoost's rank objective."""

        logger.info(f"Fitting {self.name} model...")

        self.feature_names = feature_names

        X_train = train_df[feature_names].values
        y_train = train_df["label"].values

        # group → required by ranking loss
        train_groups = train_df.groupby(["customer_id", "order_idx"]).size().values

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)

        evals = [(dtrain, "train")]

        if valid_df is not None:
            X_valid = valid_df[feature_names].values
            y_valid = valid_df["label"].values
            valid_groups = valid_df.groupby(["customer_id", "order_idx"]).size().values

            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            dvalid.set_group(valid_groups)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        self._is_fitted = True
        logger.info(f"Training complete. Best iteration={self.model.best_iteration}")

        return self

    # ----------------------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------------------
    def predict(self, customer_id: int, products: List[str], features: pd.DataFrame):
        if not self._is_fitted:
            raise RuntimeError("XGBoost model not fitted.")

        X = features[self.feature_names].values
        scores = self.model.predict(xgb.DMatrix(X))

        return scores

    # Convenience version for full DF (used by RankingMetrics)
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        X = df[self.feature_names].values
        ddata = xgb.DMatrix(X)
        return self.model.predict(ddata)

    # ----------------------------------------------------------------------
    # PARAM LOGGING
    # ----------------------------------------------------------------------
    def get_feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        return self.model.get_score(importance_type="gain")

    def get_params(self) -> Dict[str, Any]:
        params = dict(self.params)
        params["n_estimators"] = self.n_estimators
        params["early_stopping_rounds"] = self.early_stopping_rounds
        return params

    @classmethod
    def from_params(cls, params: Dict[str, Any]):
        return cls(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 500),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
        )
