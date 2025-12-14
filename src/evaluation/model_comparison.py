"""
Model Comparison
================
Compare multiple recommendation models side-by-side.

This module provides utilities for:
1. Comparing models on standard ranking metrics
2. Comparing models on discovery vs repurchase performance
3. Generating comparison reports for stakeholder communication

The key insight is that model value should be measured primarily by
discovery performance, since simple baselines can match repurchase
prediction but ML should excel at surfacing new items.

Example:
    >>> comparator = ModelComparator(k_values=[3, 5, 10])
    >>> comparison = comparator.compare(
    ...     models={"baseline": baseline_model, "lightgbm": lgb_model},
    ...     test_df=test_data,
    ...     train_df=train_data,
    ...     feature_names=features
    ... )
    >>> print(comparator.generate_report(comparison))
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd

from .metrics import RankingMetrics, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """
    Container for model comparison results.
    
    Attributes:
        comparison_df: DataFrame with models as rows, metrics as columns
        detailed_results: Dictionary of model_name -> full EvaluationResult
        best_overall: Name of best model on overall metrics
        best_discovery: Name of best model on discovery metrics
        baseline_name: Name of baseline model for lift calculations
    """
    comparison_df: pd.DataFrame
    detailed_results: Dict[str, Dict[str, EvaluationResult]]
    best_overall: Optional[str] = None
    best_discovery: Optional[str] = None
    baseline_name: Optional[str] = None

    def summary(self) -> str:
        """Generate formatted comparison summary."""
        lines = [
            "=" * 70,
            "MODEL COMPARISON SUMMARY",
            "=" * 70,
            "",
        ]

        if self.best_overall:
            overall_col = "overall_ndcg@5" if "overall_ndcg@5" in self.comparison_df.columns else "overall_ndcg@3"
            if overall_col in self.comparison_df.columns:
                overall_score = self.comparison_df.loc[self.best_overall, overall_col]
                lines.append(f"Best overall model: {self.best_overall} ({overall_col}: {overall_score:.4f})")

        if self.best_discovery:
            discovery_col = "discovery_ndcg@5" if "discovery_ndcg@5" in self.comparison_df.columns else "discovery_ndcg@3"
            if discovery_col in self.comparison_df.columns:
                discovery_score = self.comparison_df.loc[self.best_discovery, discovery_col]
                lines.append(f"Best discovery model: {self.best_discovery} ({discovery_col}: {discovery_score:.4f})")

        lines.append("")
        lines.append("Key metrics by model:")
        lines.append("-" * 50)

        # Select key columns to display
        key_cols = ["overall_ndcg@3", "overall_ndcg@5", "discovery_ndcg@3", "discovery_ndcg@5", "discovery_hit_rate"]
        available_cols = [c for c in key_cols if c in self.comparison_df.columns]

        for model in self.comparison_df.index:
            lines.append(f"  {model}:")
            for col in available_cols:
                value = self.comparison_df.loc[model, col]
                lines.append(f"    {col}: {value:.4f}")
            lines.append("")

        return "\n".join(lines)


class ModelComparator:
    """
    Compare multiple recommendation models.
    
    Provides side-by-side comparison with focus on distinguishing
    overall performance from discovery-specific performance.
    
    Args:
        k_values: List of k values for @k metrics
        
    Example:
        >>> comparator = ModelComparator()
        >>> result = comparator.compare(models, test_df, train_df, features)
        >>> print(result.comparison_df)
    """

    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Initialize comparator.
        
        Args:
            k_values: List of k values for @k metrics. Defaults to [1, 3, 5, 10].
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.evaluator = RankingMetrics(k_values=self.k_values)
        logger.info("ModelComparator initialized with k_values=%s", self.k_values)

    def compare(
        self,
        models: Dict[str, Any],
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        feature_names: List[str],
        baseline_name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare multiple models with discovery-aware evaluation.
        
        Args:
            models: Dictionary of {model_name: model_object}
            test_df: Test samples with labels
            train_df: Training data (for discovery tagging)
            feature_names: Feature column names
            baseline_name: Name of baseline model for lift calculations
            
        Returns:
            ComparisonResult with comparison DataFrame and detailed results
        """
        logger.info("Comparing %d models", len(models))

        rows = []
        detailed_results = {}

        for name, model in models.items():
            logger.info("Evaluating model: %s", name)

            try:
                eval_results = self.evaluator.evaluate_discovery(
                    model=model,
                    test_df=test_df,
                    train_df=train_df,
                    feature_names=feature_names
                )
                detailed_results[name] = eval_results

                row = self._extract_comparison_row(name, eval_results)
                rows.append(row)

            except Exception as e:
                logger.error("Failed to evaluate model %s: %s", name, str(e))
                raise

        comparison_df = pd.DataFrame(rows).set_index("model")

        # Identify best models
        best_overall = None
        best_discovery = None

        # Try different column names for overall metric
        overall_cols = ["overall_ndcg@5", "overall_ndcg@3", "overall_ndcg@1"]
        for col in overall_cols:
            if col in comparison_df.columns:
                best_overall = comparison_df[col].idxmax()
                break

        # Try different column names for discovery metric
        discovery_cols = ["discovery_ndcg@5", "discovery_ndcg@3", "discovery_ndcg@1"]
        for col in discovery_cols:
            if col in comparison_df.columns:
                best_discovery = comparison_df[col].idxmax()
                break

        # Add ranking columns
        comparison_df = self._add_rank_columns(comparison_df)

        # Add lift vs baseline
        if baseline_name and baseline_name in comparison_df.index:
            comparison_df = self._add_lift_columns(comparison_df, baseline_name)

        logger.info("Comparison complete. Best overall: %s, Best discovery: %s",
                   best_overall, best_discovery)

        return ComparisonResult(
            comparison_df=comparison_df,
            detailed_results=detailed_results,
            best_overall=best_overall,
            best_discovery=best_discovery,
            baseline_name=baseline_name
        )

    def compare_standard(
        self,
        models: Dict[str, Any],
        test_df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare models using standard (non-discovery) evaluation.
        
        Use this when you don't need discovery vs repurchase breakdown.
        
        Args:
            models: Dictionary of {model_name: model_object}
            test_df: Test samples
            feature_names: Feature column names
            
        Returns:
            DataFrame with models as rows, metrics as columns
        """
        logger.info("Comparing %d models (standard evaluation)", len(models))

        rows = []
        for name, model in models.items():
            logger.info("Evaluating model: %s", name)
            result = self.evaluator.evaluate(model, test_df, feature_names)
            row = {"model": name, **result.metrics}
            rows.append(row)

        comparison_df = pd.DataFrame(rows).set_index("model")
        comparison_df = self._add_rank_columns(comparison_df)

        return comparison_df

    def _extract_comparison_row(
        self,
        model_name: str,
        eval_results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """Extract a single row of comparison data from evaluation results."""
        row = {"model": model_name}

        # Overall metrics
        for metric, value in eval_results["overall"].metrics.items():
            row[f"overall_{metric}"] = value

        # Repurchase metrics
        for metric, value in eval_results["repurchase"].metrics.items():
            row[f"repurchase_{metric}"] = value

        # Discovery metrics
        for metric, value in eval_results["discovery"].metrics.items():
            row[f"discovery_{metric}"] = value

        # Discovery statistics
        ds = eval_results["discovery"].discovery_stats
        if ds:
            row["discovery_hit_rate"] = ds.discovery_hit_rate
            row["avg_new_item_rank"] = ds.avg_new_item_rank
            row["avg_historical_rank"] = ds.avg_historical_rank
            row["pct_new_in_top5"] = ds.pct_queries_with_new_in_top5

        return row

    def _add_rank_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rank columns for key metrics."""
        key_metrics = [
            "overall_ndcg@5",
            "overall_ndcg@3",
            "discovery_ndcg@5",
            "discovery_ndcg@3",
            "discovery_hit_rate",
            "discovery_hit_rate@5"
        ]

        for col in key_metrics:
            if col in df.columns:
                df[f"{col}_rank"] = df[col].rank(ascending=False).astype(int)

        return df

    def _add_lift_columns(
        self,
        df: pd.DataFrame,
        baseline_name: str
    ) -> pd.DataFrame:
        """Add lift vs baseline columns."""
        key_metrics = ["overall_ndcg@5", "overall_ndcg@3", "discovery_ndcg@5", "discovery_ndcg@3"]

        for metric in key_metrics:
            if metric in df.columns:
                baseline_value = df.loc[baseline_name, metric]
                if baseline_value > 0:
                    lift_col = f"{metric}_lift_pct"
                    df[lift_col] = ((df[metric] - baseline_value) / baseline_value * 100).round(2)

        return df

    def generate_report(
        self,
        result: ComparisonResult,
        include_recommendations: bool = True
    ) -> str:
        """
        Generate a detailed comparison report.
        
        Args:
            result: ComparisonResult from compare()
            include_recommendations: Whether to include actionable recommendations
            
        Returns:
            Formatted multi-line report string
        """
        lines = [
            "=" * 70,
            "MODEL COMPARISON REPORT",
            "=" * 70,
            "",
        ]

        # Summary section
        lines.extend(self._format_summary_section(result))

        # Detailed metrics table
        lines.extend(self._format_metrics_section(result))

        # Discovery analysis
        lines.extend(self._format_discovery_section(result))

        # Recommendations
        if include_recommendations:
            lines.extend(self._format_recommendations_section(result))

        return "\n".join(lines)

    def _format_summary_section(self, result: ComparisonResult) -> List[str]:
        """Format the summary section of the report."""
        lines = [
            "SUMMARY",
            "-" * 50,
        ]

        df = result.comparison_df

        if result.best_overall:
            col = "overall_ndcg@5" if "overall_ndcg@5" in df.columns else "overall_ndcg@3"
            if col in df.columns:
                score = df.loc[result.best_overall, col]
                lines.append(f"  Best overall model: {result.best_overall} ({col}: {score:.4f})")

        if result.best_discovery:
            col = "discovery_ndcg@5" if "discovery_ndcg@5" in df.columns else "discovery_ndcg@3"
            if col in df.columns:
                score = df.loc[result.best_discovery, col]
                lines.append(f"  Best discovery model: {result.best_discovery} ({col}: {score:.4f})")

        # Lift vs baseline
        lift_col = "discovery_ndcg@5_lift_pct" if "discovery_ndcg@5_lift_pct" in df.columns else "discovery_ndcg@3_lift_pct"
        if result.baseline_name and lift_col in df.columns:
            lines.append("")
            lines.append(f"  Improvement over baseline ({result.baseline_name}):")
            for model in df.index:
                if model != result.baseline_name:
                    lift = df.loc[model, lift_col]
                    status = "+" if lift > 0 else ""
                    lines.append(f"    {model}: {status}{lift:.1f}% on discovery")

        lines.append("")
        return lines

    def _format_metrics_section(self, result: ComparisonResult) -> List[str]:
        """Format the detailed metrics section."""
        lines = [
            "DETAILED METRICS",
            "-" * 50,
        ]

        df = result.comparison_df

        # Select key columns to display
        display_cols = [
            "overall_ndcg@3", "overall_ndcg@5", "overall_hit_rate@5",
            "discovery_ndcg@3", "discovery_ndcg@5", "discovery_hit_rate@5",
            "repurchase_ndcg@5"
        ]
        available_cols = [c for c in display_cols if c in df.columns]

        if not available_cols:
            # Fallback to any available columns
            available_cols = [c for c in df.columns if not c.endswith("_rank") and not c.endswith("_lift_pct")][:5]

        # Format as table
        col_width = 15
        header = "  Model".ljust(20) + "".join(c[:col_width-1].ljust(col_width) for c in available_cols)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for model in df.index:
            row = f"  {model}".ljust(20)
            for col in available_cols:
                row += f"{df.loc[model, col]:.4f}".ljust(col_width)
            lines.append(row)

        lines.append("")
        return lines

    def _format_discovery_section(self, result: ComparisonResult) -> List[str]:
        """Format the discovery analysis section."""
        lines = [
            "DISCOVERY ANALYSIS",
            "-" * 50,
            "  Discovery measures ability to recommend NEW items customers will like.",
            "  This is where ML models should demonstrate value over simple baselines.",
            "",
        ]

        df = result.comparison_df

        if "avg_new_item_rank" in df.columns:
            lines.append("  Average rank of new items (lower is better):")
            for model in df.index:
                rank = df.loc[model, "avg_new_item_rank"]
                lines.append(f"    {model}: {rank:.1f}")
            lines.append("")

        if "discovery_hit_rate" in df.columns:
            lines.append("  Discovery hit rate (new items purchased that were in top-5):")
            for model in df.index:
                rate = df.loc[model, "discovery_hit_rate"]
                lines.append(f"    {model}: {rate:.1%}")
            lines.append("")

        return lines

    def _format_recommendations_section(self, result: ComparisonResult) -> List[str]:
        """Format the recommendations section."""
        lines = [
            "RECOMMENDATIONS",
            "-" * 50,
        ]

        df = result.comparison_df

        # Check if ML model outperforms baseline on discovery
        lift_col = "discovery_ndcg@5_lift_pct" if "discovery_ndcg@5_lift_pct" in df.columns else "discovery_ndcg@3_lift_pct"
        
        if result.baseline_name and lift_col in df.columns:
            best_lift = df[lift_col].max()
            best_model = df[lift_col].idxmax()

            if best_lift > 10:
                lines.append(
                    f"  RECOMMENDED: Deploy {best_model} for discovery recommendations."
                )
                lines.append(
                    f"  It achieves {best_lift:.1f}% improvement over baseline on new item "
                    "prediction."
                )
            elif best_lift > 0:
                lines.append(
                    f"  {best_model} shows modest improvement ({best_lift:.1f}%) over baseline."
                )
                lines.append(
                    "  Consider additional feature engineering or model tuning."
                )
            else:
                lines.append(
                    "  WARNING: No model significantly outperforms baseline on discovery."
                )
                lines.append(
                    "  Consider: feature engineering, more training data, or different model architectures."
                )
        else:
            if result.best_discovery:
                lines.append(f"  Best discovery model: {result.best_discovery}")
                lines.append(
                    "  Recommendation: Compare against a popularity baseline to quantify lift."
                )

        lines.append("")
        lines.append("  NOTE: Model value is measured by discovery performance.")
        lines.append("  Simple popularity baselines can match repurchase prediction,")
        lines.append("  but ML should significantly outperform on surfacing new items.")
        lines.append("")

        return lines


def compare_models(
    models: Dict[str, Any],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_names: List[str],
    baseline_name: Optional[str] = None,
    k_values: Optional[List[int]] = None
) -> ComparisonResult:
    """
    Convenience function for model comparison.
    
    Args:
        models: Dictionary of {model_name: model_object}
        test_df: Test samples
        train_df: Training data
        feature_names: Feature column names
        baseline_name: Name of baseline model
        k_values: k values for @k metrics
        
    Returns:
        ComparisonResult with comparison DataFrame and detailed results
    """
    comparator = ModelComparator(k_values=k_values)
    return comparator.compare(
        models=models,
        test_df=test_df,
        train_df=train_df,
        feature_names=feature_names,
        baseline_name=baseline_name
    )