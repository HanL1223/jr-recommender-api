"""
Ranking Metrics
===============
Evaluation metrics for ranking and recommendation models.

This module provides comprehensive evaluation for recommendation systems,
with special focus on distinguishing between:

1. Repurchase prediction - predicting items the customer has bought before
2. Discovery prediction - predicting NEW items the customer will try

The distinction is critical because a simple popularity baseline can perform
well on repurchase, but ML models should demonstrate value on discovery.

Supported metrics:
    - NDCG@k (Normalized Discounted Cumulative Gain)
    - Hit Rate@k (at least one relevant item in top-k)
    - Precision@k (fraction of relevant items in top-k)
    - MRR (Mean Reciprocal Rank)

Example:
    >>> evaluator = RankingMetrics(k_values=[3, 5, 10])
    >>> results = evaluator.evaluate_discovery(
    ...     model=trained_model,
    ...     test_df=test_data,
    ...     train_df=train_data,
    ...     feature_names=features
    ... )
    >>> print(results["discovery"].summary())
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class MetricInterpretation:
    """
    Human-readable interpretation of a metric value.
    
    Attributes:
        value: The raw metric value
        rating: Qualitative rating (poor, fair, good, excellent)
        explanation: Plain-language explanation of what the value means
    """
    value: float
    rating: str
    explanation: str


@dataclass
class DiscoveryStats:
    """
    Statistics about discovery (new item) recommendation performance.
    
    These metrics specifically measure how well the model surfaces items
    that customers have not purchased before.
    
    Attributes:
        avg_new_item_rank: Average position of new items in rankings
        avg_historical_rank: Average position of previously-purchased items
        pct_queries_with_new_in_top3: Percentage of queries with a new item in top 3
        pct_queries_with_new_in_top5: Percentage of queries with a new item in top 5
        discovery_hits_top5: Count of new items that were purchased and ranked in top 5
        total_discovery_opportunities: Total new items that were actually purchased
        discovery_hit_rate: Ratio of discovery_hits to opportunities
    """
    avg_new_item_rank: float
    avg_historical_rank: float
    pct_queries_with_new_in_top3: float
    pct_queries_with_new_in_top5: float
    discovery_hits_top5: int
    total_discovery_opportunities: int
    discovery_hit_rate: float


@dataclass
class EvaluationResult:
    """
    Container for evaluation results with interpretability features.
    
    Attributes:
        metrics: Dictionary of metric_name -> value
        per_query: DataFrame with per-query metric values for distribution analysis
        interpretations: Dictionary of metric_name -> MetricInterpretation
        discovery_stats: Optional discovery-specific statistics
        segment_breakdown: Optional per-segment metric breakdown
    """
    metrics: Dict[str, float]
    per_query: pd.DataFrame
    interpretations: Dict[str, MetricInterpretation] = field(default_factory=dict)
    discovery_stats: Optional[DiscoveryStats] = None
    segment_breakdown: Optional[Dict[str, pd.DataFrame]] = None

    def summary(self) -> str:
        """
        Generate formatted summary of evaluation results.
        
        Returns:
            Multi-line string with metrics, interpretations, and diagnostics.
        """
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            "",
        ]

        # Group metrics by type (ndcg, hit_rate, etc.)
        metric_groups = self._group_metrics()
        
        for group_name, group_metrics in sorted(metric_groups.items()):
            lines.append(f"{group_name.upper()}")
            lines.append("-" * 40)
            
            for metric_name, value in sorted(group_metrics):
                interp = self.interpretations.get(metric_name)
                rating_str = f"[{interp.rating}]" if interp else ""
                lines.append(f"  {metric_name}: {value:.4f} {rating_str}")
                if interp and interp.explanation:
                    lines.append(f"    -> {interp.explanation}")
            lines.append("")

        # Discovery statistics
        if self.discovery_stats:
            lines.extend(self._format_discovery_stats())

        # Distribution insights
        lines.extend(self._format_distribution_insights())

        return "\n".join(lines)

    def _group_metrics(self) -> Dict[str, List[Tuple[str, float]]]:
        """Group metrics by their base name (e.g., ndcg@5 -> ndcg)."""
        groups = {}
        for name, value in self.metrics.items():
            base = name.split("@")[0]
            groups.setdefault(base, []).append((name, value))
        return groups

    def _format_discovery_stats(self) -> List[str]:
        """Format discovery statistics section."""
        ds = self.discovery_stats
        lines = [
            "DISCOVERY PERFORMANCE",
            "-" * 40,
            f"  Average rank of NEW items: {ds.avg_new_item_rank:.1f}",
            f"  Average rank of HISTORICAL items: {ds.avg_historical_rank:.1f}",
            f"  Queries with new item in top-3: {ds.pct_queries_with_new_in_top3:.1f}%",
            f"  Queries with new item in top-5: {ds.pct_queries_with_new_in_top5:.1f}%",
            f"  Discovery hit rate: {ds.discovery_hit_rate:.1%} "
            f"({ds.discovery_hits_top5}/{ds.total_discovery_opportunities})",
            "",
        ]

        # Add diagnostic message
        rank_diff = ds.avg_new_item_rank - ds.avg_historical_rank
        if rank_diff > 2:
            lines.append(
                "  NOTE: Model ranks historical items significantly higher than new items."
            )
            lines.append(
                "  This is expected but limits discovery value."
            )
        elif rank_diff < 0:
            lines.append(
                "  NOTE: Model successfully surfaces new items above historical items."
            )
        lines.append("")

        return lines

    def _format_distribution_insights(self) -> List[str]:
        """Format distribution analysis section."""
        lines = [
            "DISTRIBUTION ANALYSIS",
            "-" * 40,
        ]

        # Use ndcg@5 as the primary metric for distribution analysis
        key_metric = "ndcg@5" if "ndcg@5" in self.per_query.columns else None
        if key_metric is None and len(self.per_query.columns) > 0:
            key_metric = self.per_query.columns[0]

        if key_metric:
            values = self.per_query[key_metric]
            lines.extend([
                f"  {key_metric} distribution:",
                f"    Mean:   {values.mean():.4f}",
                f"    Median: {values.median():.4f}",
                f"    Std:    {values.std():.4f}",
                f"    Min:    {values.min():.4f}",
                f"    Max:    {values.max():.4f}",
            ])

            # Diagnostic insights
            zero_pct = (values == 0).mean() * 100
            perfect_pct = (values == 1.0).mean() * 100

            if zero_pct > 10:
                lines.append(f"  WARNING: {zero_pct:.1f}% of queries have zero {key_metric}")
            lines.append(f"  {perfect_pct:.1f}% of queries achieve perfect ranking")

        return lines

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary with metrics and summary statistics.
        """
        result = {
            "metrics": self.metrics,
            "n_queries": len(self.per_query),
        }
        
        if self.discovery_stats:
            result["discovery_stats"] = {
                "avg_new_item_rank": self.discovery_stats.avg_new_item_rank,
                "avg_historical_rank": self.discovery_stats.avg_historical_rank,
                "discovery_hit_rate": self.discovery_stats.discovery_hit_rate,
            }
        
        return result


# -----------------------------------------------------------------------------
# Main Evaluator Class
# -----------------------------------------------------------------------------

class RankingMetrics:
    """
    Compute ranking metrics for recommendation models.
    
    This evaluator supports both standard evaluation and discovery-aware
    evaluation, which separately measures performance on repurchase vs
    new item prediction.
    
    Works with any model implementing either:
        - predict_df(df) -> array (ML models with is_ml_model=True)
        - predict(customer_id, products, features) -> scores (baseline models)
    
    Attributes:
        k_values: List of k values for @k metrics
        
    Example:
        >>> evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
        >>> result = evaluator.evaluate(model, test_df, feature_names)
        >>> print(result.summary())
    """

    # Thresholds for metric interpretation (domain-specific, adjust as needed)
    METRIC_THRESHOLDS = {
        "ndcg": [(0.3, "poor"), (0.5, "fair"), (0.7, "good"), (1.0, "excellent")],
        "hit_rate": [(0.4, "poor"), (0.6, "fair"), (0.8, "good"), (1.0, "excellent")],
        "precision": [(0.2, "poor"), (0.4, "fair"), (0.6, "good"), (1.0, "excellent")],
        "mrr": [(0.3, "poor"), (0.5, "fair"), (0.7, "good"), (1.0, "excellent")],
    }

    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of k values for computing @k metrics.
                     Defaults to [1, 3, 5, 10].
        """
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info("RankingMetrics initialized with k_values=%s", self.k_values)

    # -------------------------------------------------------------------------
    # Public Evaluation Methods
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        test_df: pd.DataFrame,
        feature_names: List[str]
    ) -> EvaluationResult:
        """
        Evaluate model with standard ranking metrics.
        
        Args:
            model: Trained model with predict_df or predict method
            test_df: Test samples with columns: customer_id, order_idx, 
                    product, label, and feature columns
            feature_names: List of feature column names
            
        Returns:
            EvaluationResult with metrics and per-query breakdown
        """
        logger.info("Evaluating model (standard metrics)")

        df = test_df.copy()
        df["pred_score"] = self._predict_scores(model, df, feature_names)

        per_query_metrics = self._compute_per_query_metrics(df, label_col="label")
        per_query_df = pd.DataFrame(per_query_metrics)
        avg_metrics = per_query_df.mean().to_dict()

        interpretations = {
            name: self._interpret_metric(name, value)
            for name, value in avg_metrics.items()
        }

        logger.info("Evaluation complete: %d queries", len(per_query_df))

        return EvaluationResult(
            metrics=avg_metrics,
            per_query=per_query_df,
            interpretations=interpretations
        )

    def evaluate_discovery(
        self,
        model: Any,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model with discovery-aware metrics.
        
        This method separately evaluates:
        1. Overall performance (all purchases)
        2. Repurchase performance (historical items only)
        3. Discovery performance (new items only)
        
        The distinction is important because:
        - A popularity baseline can achieve good repurchase performance
        - ML models should demonstrate value on discovery
        
        Args:
            model: Trained model
            test_df: Test samples with labels
            train_df: Training data (used to identify historical purchases)
            feature_names: List of feature column names
            
        Returns:
            Dictionary with keys "overall", "repurchase", "discovery",
            each containing an EvaluationResult
        """
        logger.info("Evaluating model (discovery-aware)")

        # Tag each test row as historical or new
        df = self._tag_discovery(test_df, train_df)
        df["pred_score"] = self._predict_scores(model, df, feature_names)

        results = {}

        # 1. Overall performance
        logger.info("Computing overall metrics")
        overall_metrics = self._compute_per_query_metrics(df, label_col="label")
        results["overall"] = EvaluationResult(
            metrics=pd.DataFrame(overall_metrics).mean().to_dict(),
            per_query=pd.DataFrame(overall_metrics),
            interpretations={}
        )

        # 2. Repurchase performance (historical items as positives)
        logger.info("Computing repurchase metrics")
        repurchase_metrics = self._compute_per_query_metrics(df, label_col="repurchase_label")
        results["repurchase"] = EvaluationResult(
            metrics=pd.DataFrame(repurchase_metrics).mean().to_dict(),
            per_query=pd.DataFrame(repurchase_metrics),
            interpretations={}
        )

        # 3. Discovery performance (new items as positives)
        logger.info("Computing discovery metrics")
        discovery_metrics = self._compute_per_query_metrics(df, label_col="discovery_label")
        discovery_stats = self._compute_discovery_stats(df)

        results["discovery"] = EvaluationResult(
            metrics=pd.DataFrame(discovery_metrics).mean().to_dict(),
            per_query=pd.DataFrame(discovery_metrics),
            interpretations={},
            discovery_stats=discovery_stats
        )

        # Add interpretations
        for key, result in results.items():
            result.interpretations = {
                name: self._interpret_metric(name, value)
                for name, value in result.metrics.items()
            }

        logger.info(
            "Discovery evaluation complete: %d overall, %d repurchase, %d discovery queries",
            len(results["overall"].per_query),
            len(results["repurchase"].per_query),
            len(results["discovery"].per_query)
        )

        return results

    # -------------------------------------------------------------------------
    # Discovery Tagging
    # -------------------------------------------------------------------------

    def _tag_discovery(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Tag test rows as historical (repurchase) or discovery (new).
        
        Creates two label columns:
        - repurchase_label: 1 if purchased AND in training history
        - discovery_label: 1 if purchased AND NOT in training history
        
        Args:
            test_df: Test samples
            train_df: Training data
            
        Returns:
            DataFrame with is_historical, repurchase_label, discovery_label columns
        """
        # Build set of (customer, product) pairs from training
        historical_purchases = set(
            zip(train_df["customer_id"], train_df["product"])
        )
        logger.info("Historical (customer, product) pairs: %d", len(historical_purchases))

        df = test_df.copy()

        # Tag each row
        df["is_historical"] = df.apply(
            lambda r: (r["customer_id"], r["product"]) in historical_purchases,
            axis=1
        )

        # Create separate label columns
        df["repurchase_label"] = ((df["label"] == 1) & df["is_historical"]).astype(int)
        df["discovery_label"] = ((df["label"] == 1) & ~df["is_historical"]).astype(int)

        n_repurchase = df["repurchase_label"].sum()
        n_discovery = df["discovery_label"].sum()
        logger.info(
            "Test positives: %d repurchase, %d discovery",
            n_repurchase, n_discovery
        )

        return df

    def _compute_discovery_stats(self, df: pd.DataFrame) -> DiscoveryStats:
        """
        Compute detailed statistics about discovery recommendations.
        
        Analyzes where new items appear in the model's rankings compared
        to historical items.
        """
        stats = []

        for (cust, order), group in df.groupby(["customer_id", "order_idx"]):
            ranked = group.sort_values("pred_score", ascending=False).reset_index(drop=True)
            
            # Add rank column explicitly (0-indexed position becomes the index after reset)
            n_items = len(ranked)
            ranked["_rank"] = np.arange(n_items)

            new_items = ranked[~ranked["is_historical"]]
            historical_items = ranked[ranked["is_historical"]]

            discovery_opportunities = int((ranked["discovery_label"] == 1).sum())

            # Calculate average ranks using the explicit rank column
            # Use numpy to compute mean to avoid pandas Index compatibility issues
            avg_rank_new = np.nan
            avg_rank_hist = np.nan
            
            if len(new_items) > 0:
                avg_rank_new = float(new_items["_rank"].mean()) + 1  # Convert to 1-indexed
            
            if len(historical_items) > 0:
                avg_rank_hist = float(historical_items["_rank"].mean()) + 1

            # Check if any new items are in top positions
            new_in_top_3 = False
            new_in_top_5 = False
            discovery_hit_top5 = 0
            
            if len(new_items) > 0:
                new_ranks = new_items["_rank"].values
                new_in_top_3 = bool(np.any(new_ranks < 3))
                new_in_top_5 = bool(np.any(new_ranks < 5))
                
                # Count discovery hits in top 5
                new_items_top5 = new_items[new_items["_rank"] < 5]
                discovery_hit_top5 = int((new_items_top5["discovery_label"] == 1).sum())

            stats.append({
                "n_candidates": n_items,
                "n_new_items": len(new_items),
                "n_historical": len(historical_items),
                "avg_rank_new": avg_rank_new,
                "avg_rank_historical": avg_rank_hist,
                "new_in_top_3": new_in_top_3,
                "new_in_top_5": new_in_top_5,
                "discovery_hit_top5": discovery_hit_top5,
                "discovery_opportunities": discovery_opportunities,
            })

        summary = pd.DataFrame(stats)

        total_opportunities = int(summary["discovery_opportunities"].sum())
        total_hits = int(summary["discovery_hit_top5"].sum())

        # Calculate averages, handling NaN values
        avg_new = summary["avg_rank_new"].dropna()
        avg_hist = summary["avg_rank_historical"].dropna()

        return DiscoveryStats(
            avg_new_item_rank=float(avg_new.mean()) if len(avg_new) > 0 else 0.0,
            avg_historical_rank=float(avg_hist.mean()) if len(avg_hist) > 0 else 0.0,
            pct_queries_with_new_in_top3=float(summary["new_in_top_3"].mean() * 100),
            pct_queries_with_new_in_top5=float(summary["new_in_top_5"].mean() * 100),
            discovery_hits_top5=total_hits,
            total_discovery_opportunities=total_opportunities,
            discovery_hit_rate=(
                total_hits / total_opportunities if total_opportunities > 0 else 0.0
            ),
        )

    # -------------------------------------------------------------------------
    # Prediction Interface
    # -------------------------------------------------------------------------

    def _predict_scores(
        self,
        model: Any,
        df: pd.DataFrame,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Get prediction scores from model.
        
        Handles two model interfaces:
        - ML models (is_ml_model=True): uses predict_df(features)
        - Baseline models: uses predict(customer_id, products, features) row-by-row
        
        For ML models, tries passing only feature columns first. If that fails
        (e.g., model needs grouping columns), passes the full DataFrame.
        """
        if getattr(model, "is_ml_model", False):
            logger.debug("Using predict_df (ML model)")
            
            # First try with only feature columns
            try:
                return model.predict_df(df[feature_names])
            except KeyError as e:
                # Model might need additional columns (e.g., customer_id for grouping)
                logger.debug("Retrying with full DataFrame due to: %s", e)
                try:
                    return model.predict_df(df)
                except Exception as e2:
                    logger.error("predict_df failed: %s", e2)
                    raise
            except Exception as e:
                logger.error("predict_df failed: %s", e)
                raise

        logger.debug("Using row-by-row predict (baseline model)")
        scores = []
        for _, row in df.iterrows():
            score = model.predict(row["customer_id"], [row["product"]], None)[0]
            scores.append(score)
        return np.array(scores)

    # -------------------------------------------------------------------------
    # Per-Query Metric Computation
    # -------------------------------------------------------------------------

    def _compute_per_query_metrics(
        self,
        df: pd.DataFrame,
        label_col: str = "label"
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each query (customer_id, order_idx combination).
        """
        per_query_metrics = []

        for (cust, order), group in df.groupby(["customer_id", "order_idx"]):
            ranked = group.sort_values("pred_score", ascending=False)
            labels = ranked[label_col].values

            qm = {}
            for k in self.k_values:
                qm[f"ndcg@{k}"] = self._ndcg_at_k(labels, k)
                qm[f"hit_rate@{k}"] = self._hit_rate_at_k(labels, k)
                qm[f"precision@{k}"] = self._precision_at_k(labels, k)
            qm["mrr"] = self._mrr(labels)

            per_query_metrics.append(qm)

        return per_query_metrics

    # -------------------------------------------------------------------------
    # Metric Implementations
    # -------------------------------------------------------------------------

    def _ndcg_at_k(self, labels: np.ndarray, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        
        Measures ranking quality with position-based discounting.
        Perfect ranking = 1.0, random = varies by data.
        """
        labels_k = labels[:k]
        dcg = np.sum((2**labels_k - 1) / np.log2(np.arange(2, len(labels_k) + 2)))

        ideal = np.sort(labels)[::-1][:k]
        idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))

        return 0.0 if idcg == 0 else dcg / idcg

    def _hit_rate_at_k(self, labels: np.ndarray, k: int) -> float:
        """
        Hit Rate at k.
        
        Returns 1 if at least one relevant item is in top-k, else 0.
        """
        return 1.0 if np.sum(labels[:k]) > 0 else 0.0

    def _precision_at_k(self, labels: np.ndarray, k: int) -> float:
        """
        Precision at k.
        
        Fraction of top-k items that are relevant.
        """
        return float(np.mean(labels[:k]))

    def _mrr(self, labels: np.ndarray) -> float:
        """
        Mean Reciprocal Rank.
        
        1/position of first relevant item, or 0 if none found.
        """
        for i, label in enumerate(labels):
            if label == 1:
                return 1.0 / (i + 1)
        return 0.0

    # -------------------------------------------------------------------------
    # Metric Interpretation
    # -------------------------------------------------------------------------

    def _interpret_metric(self, metric_name: str, value: float) -> MetricInterpretation:
        """
        Generate human-readable interpretation of a metric value.
        """
        base_name = metric_name.split("@")[0]
        thresholds = self.METRIC_THRESHOLDS.get(base_name, [(1.0, "unknown")])

        rating = "unknown"
        for threshold, label in thresholds:
            if value <= threshold:
                rating = label
                break

        explanations = {
            "ndcg": f"Model achieves {value*100:.1f}% of ideal ranking quality",
            "hit_rate": f"{value*100:.1f}% of queries have relevant items in top-k",
            "precision": f"{value*100:.1f}% of top-k recommendations are relevant",
            "mrr": (
                f"First relevant item at average position {1/value:.1f}"
                if value > 0 else "No relevant items found in rankings"
            ),
        }

        return MetricInterpretation(
            value=value,
            rating=rating,
            explanation=explanations.get(base_name, f"Value: {value:.4f}")
        )