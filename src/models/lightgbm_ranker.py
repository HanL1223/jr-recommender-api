"""
LightGBM Ranker
===============
Modular, production-ready LambdaRank model.

Designed to be drop-in replaceable with:
- XGBoost Ranker
- CatBoost Ranker
- Custom ML rankers
"""

import logging
from typing import List, Dict, Any, Optional, Self

import numpy as np
import pandas as pd

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)

# LightGBM dependency
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Run: pip install lightgbm")


class LightGBMRanker(BaseRecommender):
    """Learning-to-rank model using LightGBM LambdaRank."""

    @property
    def name(self) -> str:
        return "LightGBMRanker"

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.1,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
    ) -> None:

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5, 10],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "verbose": verbose,
            "seed": 42,
        }

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self.is_ml_model = True

        logger.info(f"{self.name} initialized (num_leaves={num_leaves}, lr={learning_rate})")

    # Factory Method of Optuna
    # Refinement needed - Enable GPU,wider parameter range
    # -------------------------------------------------------------------------
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "LightGBMRanker":
        """
        Factory method used by Optuna to build a model instance from a
        hyperparameter dictionary.

        Any missing parameters fall back to the class defaults.
        """
        return cls(
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            feature_fraction=params.get("feature_fraction", 0.8),
            bagging_fraction=params.get("bagging_fraction", 0.8),
            bagging_freq=params.get("bagging_freq", 5),
            min_data_in_leaf=params.get("min_data_in_leaf", 20),
            lambda_l1=params.get("lambda_l1", 0.1),
            lambda_l2=params.get("lambda_l2", 0.1),
            num_boost_round=params.get("num_boost_round", 500),
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
        )
    
    #Fit
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        valid_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Self:

        logger.info(f"Fitting {self.name}...")

        required_cols = {"customer_id", "order_idx", "label"}
        if not required_cols.issubset(train_df.columns):
            raise ValueError(
                f"Training data missing required columns: {required_cols - set(train_df.columns)}"
            )

        # sort by grouping keys
        train_df = train_df.sort_values(["customer_id", "order_idx"])

        self.feature_names = feature_names

        X_train = train_df[feature_names].values
        y_train = train_df["label"].values
        train_groups = train_df.groupby(["customer_id", "order_idx"]).size().values

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            feature_name=feature_names,
        )

        valid_sets = [train_dataset]
        valid_names = ["train"]

        if valid_df is not None:
            valid_df = valid_df.sort_values(["customer_id", "order_idx"])
            X_valid = valid_df[feature_names].values
            y_valid = valid_df["label"].values
            valid_groups = valid_df.groupby(["customer_id", "order_idx"]).size().values

            valid_dataset = lgb.Dataset(
                X_valid,
                label=y_valid,
                group=valid_groups,
                feature_name=feature_names,
                reference=train_dataset,
            )

            valid_sets.append(valid_dataset)
            valid_names.append("valid")

        logger.info(
            f"Training with {len(X_train):,} samples and {len(train_groups):,} ranking groups..."
        )

        self.model = lgb.train(
            self.params,
            train_dataset,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        self._is_fitted = True
        logger.info(f"{self.name} training finished. Best iteration = {self.model.best_iteration}")

        return self


    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame,
    ) -> np.ndarray:

        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

        X = features[self.feature_names].values
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict_df().")

        df = df.sort_values(["customer_id", "order_idx"])
        X = df[self.feature_names].values
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    #Feature importance
    #By gain, currently no use for downstream
    def get_feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        importance = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, importance))


    def get_params(self) -> Dict[str, Any]:
        params = self.params.copy()
        params["num_boost_round"] = self.num_boost_round
        params["early_stopping_rounds"] = self.early_stopping_rounds

        if self._is_fitted:
            params["best_iteration"] = self.model.best_iteration
            params["n_trees"] = self.model.num_trees()

        return params
