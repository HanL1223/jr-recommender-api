"""
BaseRecommender
================
Abstract interface for all recommendation models.

Every model must implement:
- fit()
- predict()
- name (property)

This ensures consistent behaviour across:
- Baselines
- ML models (XGBoost, LightGBM Ranker)
- Deep models (LSTM, transformers)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing import Optional
from typing import Self
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract base class for recommender models.

    Enforces a consistent interface so that all models can be:
    - fitted
    - used for scoring
    - compared against baselines
    - logged in MLflow
    - saved/loaded consistently
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Readable model name."""
        raise NotImplementedError

#Training function
    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> Self:
        """
        Fit model on labeled training data.

        Returns:
            Self, to allow chaining (model.fit(...).predict(...))
        """
        raise NotImplementedError

#Predicting function
    @abstractmethod
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Score candidate products for a given customer.

        Returns:
            np.ndarray of shape (n_products,) with float scores.
        """
        raise NotImplementedError

    # Optional: batch scoring helper (not abstract)
    # -------------------------------------------
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Optional batch prediction for DataFrame.
        Subclasses may override if useful.
        """
        raise NotImplementedError(
            f"{self.name} does not implement batch prediction (predict_df)."
        )

    # Provided: Convert predicted scores -> ranked recommendations
    # -------------------------------------------
    def recommend(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Convert model scores into ranked recommendations.

        Returns:
            List[{product, score, rank}]
        """
        scores = self.predict(customer_id, products, features)

        # Safety checks
        if not isinstance(scores, np.ndarray):
            raise ValueError("predict() must return a numpy array.")
        if len(scores) != len(products):
            raise ValueError(
                f"predict() returned {len(scores)} scores but received {len(products)} products."
            )

        sorted_idx = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for rank, idx in enumerate(sorted_idx, start=1):
            recommendations.append({
                "product": products[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })

        return recommendations

    # Metadata for MLflow / logging
    # -------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        """Return training parameters or model metadata."""
        return {}

    # Save / load
    # -------------------------------------------
    def save(self, path: str) -> None:
        """Serialise model to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseRecommender":
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
