"""
Baseline Models
===============
Simple baseline recommenders for comparison.

- PopularityRecommender: global product popularity
- PersonalFrequencyRecommender: blends personal frequency with global popularity
"""

import logging
from typing import List, Dict, Any, Self
import numpy as np
import pandas as pd

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


# POPULARITY BASELINE  (NO predict_df → avoids ML path in RankingMetrics)
# ======================================================================
class PopularityRecommender(BaseRecommender):
    """
    Simple global popularity recommender.
    Scores = P(product was purchased at least once).
    """

    @property
    def name(self) -> str:
        return "Popularity"

    def __init__(self):
        self.popularity_scores: Dict[str, float] = {}
        self._is_fitted: bool = False

    #fit
    def fit(self, train_df, feature_names: List[str], **kwargs):
        logger.info(f"Fitting {self.name} model")

        if "label" not in train_df.columns or "product" not in train_df.columns:
            raise ValueError("train_df must contain label and product columns")

        positive_df = train_df[train_df["label"] == 1]

        if positive_df.empty:
            logger.warning("No positive samples in training data.")
            self.popularity_scores = {}
        else:
            product_counts = positive_df["product"].value_counts()
            total = float(product_counts.sum())
            self.popularity_scores = (product_counts / total).to_dict()

        self._is_fitted = True
        logger.info(f"Learned popularity for {len(self.popularity_scores)} product")
        return self

    #predict
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame = None,
    ) -> np.ndarray:
        """
        Row-by-row scoring (RankingMetrics baseline mode).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted, call fit() before predict()")

        return np.array(
            [self.popularity_scores.get(p, 0.0) for p in products],
            dtype=float,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "popularity",
            "n_products": len(self.popularity_scores),
        }


# PERSONAL FREQUENCY BASELINE
# ======================================================================
class PersonalFrequencyRecommender(BaseRecommender):
    """
    Combines:
      - personal frequency P(product | customer)
      - global popularity P(product)

    score = (1 - smoothing) * personal_freq + smoothing * global_freq
    """

    @property
    def name(self) -> str:
        return "PersonalFrequency"

    def __init__(self, smoothing: float = 0.3) -> None:
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must be between 0.0 and 1.0")

        self.smoothing = smoothing
        self.customer_frequencies: Dict[int, Dict[str, float]] = {}
        self.global_popularity: Dict[str, float] = {}
        self._is_fitted = False

        logger.info(f"PersonalFrequencyRecommender initialised (smoothing={smoothing})")

    #fit
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs,
    ) -> Self:

        logger.info(f"Fitting {self.name} model...")

        required = {"label", "product", "customer_id"}
        if not required.issubset(train_df.columns):
            raise ValueError("train_df must contain 'label', 'product', and 'customer_id' columns.")

        positive_df = train_df[train_df["label"] == 1]

        if positive_df.empty:
            logger.warning("No positive samples. Frequencies empty.")
            self._is_fitted = True
            return self

        # Global popularity
        product_counts = positive_df["product"].value_counts()
        total = float(product_counts.sum())
        self.global_popularity = (product_counts / total).to_dict()

        # Customer frequencies
        for cid, group in positive_df.groupby("customer_id"):
            counts = group["product"].value_counts()
            c_total = float(counts.sum())
            self.customer_frequencies[cid] = (counts / c_total).to_dict()

        self._is_fitted = True

        logger.info(f"Learned frequencies for {len(self.customer_frequencies)} customers")
        logger.info(f"Global popularity for {len(self.global_popularity)} products")

        return self

    #predict
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame = None,
    ) -> np.ndarray:

        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        personal_freq = self.customer_frequencies.get(customer_id, {})

        scores = []
        for p in products:
            p_personal = personal_freq.get(p, 0.0)
            p_global = self.global_popularity.get(p, 0.0)

            score = (1 - self.smoothing) * p_personal + self.smoothing * p_global
            scores.append(score)

        return np.array(scores, dtype=float)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        scores = []
        for _, row in df.iterrows():
            cid, prod = row["customer_id"], row["product"]
            p_personal = self.customer_frequencies.get(cid, {}).get(prod, 0.0)
            p_global = self.global_popularity.get(prod, 0.0)

            scores.append(
                (1 - self.smoothing) * p_personal + self.smoothing * p_global
            )

        return np.array(scores, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "personal_frequency",
            "smoothing": self.smoothing,
            "n_customers": len(self.customer_frequencies),
            "n_products": len(self.global_popularity),
        }
