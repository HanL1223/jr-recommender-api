"""
Ranking Metrics
===============
Evaluation helpers for ranking/recommendation models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    per_query: pd.DataFrame


class RankingMetrics:
    """
    Compute ranking metrics for recommenders.

    Supported metrics:
    - NDCG@k
    - Hit Rate@k
    - Precision@k
    - MRR (Mean Reciprocal Rank)

    Works with ANY model implementing:
        predict_df(df)   → ML models (LightGBM, XGBoost, CatBoost)
        predict(cust_id, products, features) → Baseline models
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info(f"RankingMetrics initialized (k={self.k_values})")

    # -----------------------------
    # Main evaluation entry point
    # -----------------------------
    def evaluate(self, model, test_df: pd.DataFrame, feature_names: List[str]) -> EvaluationResult:
        """Evaluate ranking performance."""
        logger.info("Evaluating model...")

        df = test_df.copy()

        # 1. Generate predictions
        df["pred_score"] = self._predict_scores(model, df, feature_names)

        # 2. Compute per-query metrics
        per_query_metrics = []
        grouped = df.groupby(["customer_id", "order_idx"])

        for (cust, order), group in grouped:
            labels = group.sort_values("pred_score", ascending=False)["label"].values

            qm = {f"ndcg@{k}": self._ndcg_at_k(labels, k) for k in self.k_values}
            qm.update({f"hit_rate@{k}": self._hit_rate_at_k(labels, k) for k in self.k_values})
            qm.update({f"precision@{k}": self._precision_at_k(labels, k) for k in self.k_values})
            qm["mrr"] = self._mrr(labels)

            per_query_metrics.append(qm)

        per_query_df = pd.DataFrame(per_query_metrics)
        avg_metrics = per_query_df.mean().to_dict()

        return EvaluationResult(metrics=avg_metrics, per_query=per_query_df)

    # -------------------------------------
    # Prediction handling for any model
    # -------------------------------------
    def _predict_scores(self, model, df, feature_names):

        # Detect true ML models → they declare `is_ml_model = True`
        if getattr(model, "is_ml_model", False):
            logger.info("Using predict_df (ML model)")
            return model.predict_df(df[feature_names])

        # Baseline models → row-by-row evaluation
        logger.info("Using predict() row-by-row (baseline model)")
        scores = []
        for _, row in df.iterrows():
            scores.append(
                model.predict(
                    row["customer_id"],
                    [row["product"]],
                    None
                )[0]
            )
        return np.array(scores)


    # -------------------------------------
    # Metric computations
    # -------------------------------------
    def _ndcg_at_k(self, labels, k):
        labels_k = labels[:k]
        dcg = np.sum((2**labels_k - 1) / np.log2(np.arange(2, len(labels_k) + 2)))

        ideal = np.sort(labels)[::-1][:k]
        idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
        return 0.0 if idcg == 0 else dcg / idcg

    def _hit_rate_at_k(self, labels, k):
        return 1.0 if np.sum(labels[:k]) > 0 else 0.0

    def _precision_at_k(self, labels, k):
        return np.mean(labels[:k])

    def _mrr(self, labels):
        for i, lbl in enumerate(labels):
            if lbl == 1:
                return 1.0 / (i + 1)
        return 0.0
