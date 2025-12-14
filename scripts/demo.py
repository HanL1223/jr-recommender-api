"""
evaluate_demo.py
================

Comprehensive evaluation and demonstration of the trained recommender.

This script:
1. Loads trained model and rebuilds test data
2. Runs discovery-aware evaluation (repurchase vs new items)
3. Compares ML model against baselines
4. Shows detailed recommendations for sample customers
5. Explains recommendation reasons

Usage:
    python scripts/evaluate_demo.py
    python scripts/evaluate_demo.py --customer-id 123
    python scripts/evaluate_demo.py --top-k 10
"""

import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.inference.recommender_predictor import RecommenderPredictor
from src.inference.cold_start import ColdStartHandler
from src.evaluation.metrics import RankingMetrics
from src.evaluation.model_comparison import ModelComparator
from src.models.baseline_models import PopularityRecommender, PersonalFrequencyRecommender
from src.features.training_data_builder import TrainingDataBuilder
from src.training.data_splitter import TemporalDataSplitter

ARTIFACT_DIR = Path("models/artifacts")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EVALUATE_DEMO")


# -----------------------------------------------------------------------------
# Artifact Loading
# -----------------------------------------------------------------------------

def load_artifacts():
    """Load all training artifacts."""
    logger.info("Loading artifacts from %s", ARTIFACT_DIR)
    
    # Load model
    with open(ARTIFACT_DIR / "recommender.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    
    # Load metadata
    with open(ARTIFACT_DIR / "model_info.json", "r") as f:
        model_info = json.load(f)
    
    # Load features and data
    with open(ARTIFACT_DIR / "product_features.pkl", "rb") as f:
        product_features = pickle.load(f)
    
    with open(ARTIFACT_DIR / "customer_profiles.pkl", "rb") as f:
        customer_profiles = pickle.load(f)
    
    with open(ARTIFACT_DIR / "prepared_data.pkl", "rb") as f:
        prepared_data = pickle.load(f)
    
    logger.info("Loaded model: %s", model_info.get("model_name", type(model).__name__))
    logger.info("Test NDCG@3: %.4f", model_info.get("test_metrics", {}).get("ndcg@3", 0))
    
    return {
        "model": model,
        "feature_names": feature_names,
        "model_info": model_info,
        "product_features": product_features,
        "customer_profiles": customer_profiles,
        "prepared_data": prepared_data,
    }


def rebuild_test_data(artifacts: dict) -> dict:
    """Rebuild train/test split for evaluation."""
    logger.info("Rebuilding training data for evaluation...")
    
    prepared_data = artifacts["prepared_data"]
    customer_profiles = artifacts["customer_profiles"]
    product_features = artifacts["product_features"]
    
    # Rebuild training samples
    builder = TrainingDataBuilder(negative_ratio=5)
    if hasattr(builder, 'set_cooccurrence'):
        builder.set_cooccurrence(product_features.cooccurrence)
    
    training_data = builder.build(
        prepared_data=prepared_data,
        customer_profiles=customer_profiles,
        product_features=product_features,
    )
    
    # Split with stratified temporal (guarantees customer overlap)
    splitter = TemporalDataSplitter(
        valid_ratio=0.1,
        test_ratio=0.2,
        method="stratified_temporal",
        min_orders_per_customer=2,
    )
    split = splitter.split(training_data, date_column="order_date")
    
    logger.info("Train: %d, Valid: %d, Test: %d samples", 
                split.n_train_samples, split.n_valid_samples, split.n_test_samples)
    logger.info("Test customer coverage: %.1f%%", split.customer_coverage_test * 100)
    
    return {
        "train_df": split.train_df,
        "valid_df": split.valid_df,
        "test_df": split.test_df,
        "feature_names": split.feature_names,
        "split": split,
    }


# -----------------------------------------------------------------------------
# Evaluation Functions
# -----------------------------------------------------------------------------

def run_discovery_evaluation(artifacts: dict, split_data: dict) -> dict:
    """Run discovery-aware evaluation on the trained model."""
    print("\n" + "=" * 70)
    print("DISCOVERY-AWARE EVALUATION")
    print("=" * 70)
    
    model = artifacts["model"]
    evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
    
    # Run discovery evaluation
    results = evaluator.evaluate_discovery(
        model=model,
        test_df=split_data["test_df"],
        train_df=split_data["train_df"],
        feature_names=split_data["feature_names"],
    )
    
    # Print overall metrics
    print("\n--- OVERALL PERFORMANCE ---")
    overall = results["overall"]
    print(f"  NDCG@3:     {overall.metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5:     {overall.metrics['ndcg@5']:.4f}")
    print(f"  Hit Rate@5: {overall.metrics['hit_rate@5']:.4f}")
    print(f"  MRR:        {overall.metrics['mrr']:.4f}")
    
    # Print repurchase metrics
    print("\n--- REPURCHASE PERFORMANCE (items customer bought before) ---")
    repurchase = results["repurchase"]
    print(f"  NDCG@3:     {repurchase.metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5:     {repurchase.metrics['ndcg@5']:.4f}")
    print(f"  Hit Rate@5: {repurchase.metrics['hit_rate@5']:.4f}")
    
    # Print discovery metrics
    print("\n--- DISCOVERY PERFORMANCE (NEW items customer will try) ---")
    discovery = results["discovery"]
    print(f"  NDCG@3:     {discovery.metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5:     {discovery.metrics['ndcg@5']:.4f}")
    print(f"  Hit Rate@5: {discovery.metrics['hit_rate@5']:.4f}")
    
    # Print discovery statistics
    if discovery.discovery_stats:
        ds = discovery.discovery_stats
        print("\n--- DISCOVERY STATISTICS ---")
        print(f"  Avg rank of NEW items:        {ds.avg_new_item_rank:.1f}")
        print(f"  Avg rank of HISTORICAL items: {ds.avg_historical_rank:.1f}")
        print(f"  Discovery hit rate:           {ds.discovery_hit_rate:.1%}")
        print(f"  Queries with new in top-5:    {ds.pct_queries_with_new_in_top5:.1f}%")
        
        # Interpretation
        rank_diff = ds.avg_new_item_rank - ds.avg_historical_rank
        if rank_diff > 5:
            print("\n  INTERPRETATION: Model strongly favors historical items.")
            print("  Consider tuning for better discovery if that's a business goal.")
        elif rank_diff > 2:
            print("\n  INTERPRETATION: Model moderately favors historical items.")
            print("  This is typical behavior for personalized recommenders.")
        else:
            print("\n  INTERPRETATION: Model balances historical and new items well.")
    
    return results


def run_model_comparison(artifacts: dict, split_data: dict) -> dict:
    """Compare trained model against baselines."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON VS BASELINES")
    print("=" * 70)
    
    train_df = split_data["train_df"]
    test_df = split_data["test_df"]
    feature_names = split_data["feature_names"]
    
    # Train baselines
    logger.info("Training baseline models...")
    
    popularity = PopularityRecommender()
    popularity.fit(train_df, feature_names)
    
    personal_freq = PersonalFrequencyRecommender(smoothing=0.3)
    personal_freq.fit(train_df, feature_names)
    
    # Build models dict
    models = {
        "Popularity": popularity,
        "PersonalFreq": personal_freq,
        "TrainedModel": artifacts["model"],
    }
    
    # Run comparison
    comparator = ModelComparator(k_values=[1, 3, 5, 10])
    comparison = comparator.compare(
        models=models,
        test_df=test_df,
        train_df=train_df,
        feature_names=feature_names,
        baseline_name="Popularity",
    )
    
    # Print comparison table
    print("\n--- COMPARISON TABLE ---")
    display_cols = ["overall_ndcg@3", "overall_ndcg@5", "discovery_ndcg@3", "discovery_ndcg@5"]
    available_cols = [c for c in display_cols if c in comparison.comparison_df.columns]
    
    print(comparison.comparison_df[available_cols].to_string())
    
    # Print lift over baseline
    print("\n--- LIFT OVER POPULARITY BASELINE ---")
    baseline_overall = comparison.comparison_df.loc["Popularity", "overall_ndcg@3"]
    baseline_discovery = comparison.comparison_df.loc["Popularity", "discovery_ndcg@3"]
    
    for model_name in comparison.comparison_df.index:
        if model_name != "Popularity":
            overall_lift = (comparison.comparison_df.loc[model_name, "overall_ndcg@3"] - baseline_overall) / baseline_overall * 100
            discovery_lift = (comparison.comparison_df.loc[model_name, "discovery_ndcg@3"] - baseline_discovery) / baseline_discovery * 100
            print(f"  {model_name}:")
            print(f"    Overall lift:   {overall_lift:+.1f}%")
            print(f"    Discovery lift: {discovery_lift:+.1f}%")
    
    # Business recommendation
    print("\n--- RECOMMENDATION ---")
    trained_discovery = comparison.comparison_df.loc["TrainedModel", "discovery_ndcg@3"]
    if trained_discovery > baseline_discovery * 1.1:
        print("  The trained ML model outperforms baselines on discovery.")
        print("  RECOMMENDED: Use ML model for 'Try Something New' recommendations.")
    elif trained_discovery > baseline_discovery:
        print("  The trained ML model shows modest improvement on discovery.")
        print("  Consider additional feature engineering or tuning.")
    else:
        print("  WARNING: ML model does not outperform baseline on discovery.")
        print("  Review feature engineering and model architecture.")
    
    return comparison


# -----------------------------------------------------------------------------
# Recommendation Demo
# -----------------------------------------------------------------------------

def demo_recommendations(
    artifacts: dict,
    customer_ids: Optional[List[int]] = None,
    top_k: int = 5,
):
    """Show detailed recommendations for sample customers."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION EXAMPLES")
    print("=" * 70)
    
    # Build predictor
    cold_handler = ColdStartHandler(
        product_features=artifacts["product_features"],
        prepared_data=artifacts["prepared_data"],
        customer_profiles=artifacts["customer_profiles"],
    )
    
    predictor = RecommenderPredictor(
        ml_model=artifacts["model"],
        baseline_model=None,
        product_features=artifacts["product_features"],
        prepared_data=artifacts["prepared_data"],
        customer_profiles=artifacts["customer_profiles"],
        feature_names=artifacts["feature_names"],
        cold_start_handler=cold_handler,
    )
    
    prepared_data = artifacts["prepared_data"]
    customer_profiles = artifacts["customer_profiles"]
    
    # Select customers to demo
    if customer_ids is None:
        # Pick diverse customers
        all_customers = list(prepared_data.customer_histories.keys())
        
        # Longest history
        long_hist = max(all_customers, key=lambda c: len(prepared_data.customer_histories[c]))
        
        # Shortest history
        short_hist = min(all_customers, key=lambda c: len(prepared_data.customer_histories[c]))
        
        # Middle history
        sorted_by_hist = sorted(all_customers, key=lambda c: len(prepared_data.customer_histories[c]))
        mid_hist = sorted_by_hist[len(sorted_by_hist) // 2]
        
        customer_ids = [long_hist, mid_hist, short_hist]
    
    for cid in customer_ids:
        print_customer_recommendation(
            predictor=predictor,
            customer_id=cid,
            prepared_data=prepared_data,
            customer_profiles=customer_profiles,
            top_k=top_k,
        )
    
    # Cold start demo
    print("\n" + "-" * 70)
    print("COLD START CUSTOMER (New User)")
    print("-" * 70)
    
    cold_pred = predictor.recommend(customer_id=99999999, top_k=top_k)
    print(f"Customer ID: 99999999 (unknown)")
    print(f"Model used:  {cold_pred.model_used}")
    print("\nRecommendations:")
    for i, item in enumerate(cold_pred.primary_items, 1):
        print(f"  {i}. {item.product:35s} | Score: {item.score:.4f} | Reason: {item.reason}")


def print_customer_recommendation(
    predictor,
    customer_id: int,
    prepared_data,
    customer_profiles: dict,
    top_k: int = 5,
):
    """Print detailed recommendation for a single customer."""
    print("\n" + "-" * 70)
    print(f"CUSTOMER ID: {customer_id}")
    print("-" * 70)
    
    # Get customer profile
    profile = customer_profiles.get(customer_id, {})
    history = prepared_data.customer_histories.get(customer_id, [])
    if profile is None:
        archetype = 'unknown'
        avg_order_value = 0
        order_frequency = 0
    elif isinstance(profile, dict):
        archetype = profile.get('archetype', 'unknown')
        avg_order_value = profile.get('avg_order_value', 0)
        order_frequency = profile.get('order_frequency', 0)
    else:
        # It's a dataclass/object
        archetype = getattr(profile, 'archetype', 'unknown')
        avg_order_value = getattr(profile, 'avg_order_value', 0)
        order_frequency = getattr(profile, 'order_frequency', 0)
    
    # Print customer context
    print("\nCUSTOMER PROFILE:")
    print(f"  Total orders:     {len(history)}")
    print(f"  Archetype:        {profile.get('archetype', 'unknown')}")
    print(f"  Avg order value:  ${profile.get('avg_order_value', 0):.2f}")
    print(f"  Order frequency:  {profile.get('order_frequency', 0):.2f} days between orders")
    
    # Print purchase history
    print("\nPURCHASE HISTORY (recent 5 orders):")
    recent_orders = history[-5:] if len(history) > 5 else history
    for order in recent_orders:
        products = order.get("products", [])
        date = order.get("order_date", "unknown")
        print(f"  [{date}] {', '.join(products[:3])}{'...' if len(products) > 3 else ''}")
    
    # Get recommendation
    pred = predictor.recommend(customer_id=customer_id, top_k=top_k)
    
    # Print recommendations with detailed reasons
    print(f"\nRECOMMENDATIONS (model: {pred.model_used}):")
    print("-" * 50)
    
    # Get customer's historical products for context
    historical_products = set()
    for order in history:
        historical_products.update(order.get("products", []))
    
    for i, item in enumerate(pred.primary_items, 1):
        is_new = item.product not in historical_products
        new_tag = "[NEW]" if is_new else "[REPEAT]"
        
        print(f"\n  {i}. {item.product}")
        print(f"     Score:  {item.score:.4f}")
        print(f"     Type:   {new_tag}")
        print(f"     Reason: {item.reason}")
        
        # Add more context based on reason
        if "favorite" in item.reason.lower():
            # Count how many times purchased
            count = sum(1 for o in history if item.product in o.get("products", []))
            print(f"     Context: Purchased {count} times previously")
        elif "popular" in item.reason.lower():
            print(f"     Context: High popularity among all customers")
        elif "archetype" in item.reason.lower() or "similar" in item.reason.lower():
            print(f"     Context: Matches {profile.get('archetype', 'unknown')} preferences")
    
    # Print add-ons if available
    if pred.addon_items:
        print("\n  ADD-ON SUGGESTIONS:")
        for item in pred.addon_items:
            print(f"    + {item.product:30s} | {item.reason}")


# -----------------------------------------------------------------------------
# Summary Report
# -----------------------------------------------------------------------------

def print_final_summary(artifacts: dict, eval_results: dict, comparison):
    """Print final summary report."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    
    model_info = artifacts["model_info"]
    
    print("\nMODEL INFORMATION")
    print("-" * 40)
    print(f"  Model name:       {model_info.get('model_name', 'Unknown')}")
    print(f"  Training date:    {model_info.get('training_date', 'Unknown')}")
    print(f"  Version:          {model_info.get('version', 'Unknown')}")
    
    print("\nPERFORMANCE SUMMARY")
    print("-" * 40)
    
    overall = eval_results["overall"]
    discovery = eval_results["discovery"]
    
    print(f"  Overall NDCG@3:    {overall.metrics['ndcg@3']:.4f}")
    print(f"  Discovery NDCG@3:  {discovery.metrics['ndcg@3']:.4f}")
    
    if discovery.discovery_stats:
        print(f"  Discovery hit rate: {discovery.discovery_stats.discovery_hit_rate:.1%}")
    
    # Lift over baseline
    if comparison:
        baseline_discovery = comparison.comparison_df.loc["Popularity", "discovery_ndcg@3"]
        trained_discovery = comparison.comparison_df.loc["TrainedModel", "discovery_ndcg@3"]
        lift = (trained_discovery - baseline_discovery) / baseline_discovery * 100
        print(f"  Lift over baseline: {lift:+.1f}%")
    
    print("\nBUSINESS VALUE")
    print("-" * 40)
    
    ds = discovery.discovery_stats
    if ds and ds.discovery_hit_rate > 0.2:
        print("  STRONG: Model effectively recommends new items customers will buy.")
        print("  Use for 'Try Something New' feature with high confidence.")
    elif ds and ds.discovery_hit_rate > 0.1:
        print("  MODERATE: Model shows reasonable discovery performance.")
        print("  Good for personalized recommendations, room for improvement on discovery.")
    else:
        print("  WEAK: Model primarily recommends repeat purchases.")
        print("  Consider feature engineering to improve discovery.")
    
    print("\n" + "=" * 70)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained recommender model")
    parser.add_argument("--customer-id", type=int, nargs="+", help="Specific customer IDs to demo")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only show demos")
    parser.add_argument("--skip-demo", action="store_true", help="Skip demos, only show evaluation")
    
    args = parser.parse_args()
    
    # Load artifacts
    artifacts = load_artifacts()
    
    eval_results = None
    comparison = None
    
    if not args.skip_eval:
        # Rebuild test data
        split_data = rebuild_test_data(artifacts)
        
        # Run discovery evaluation
        eval_results = run_discovery_evaluation(artifacts, split_data)
        
        # Run model comparison
        comparison = run_model_comparison(artifacts, split_data)
    
    if not args.skip_demo:
        # Demo recommendations
        demo_recommendations(
            artifacts=artifacts,
            customer_ids=args.customer_id,
            top_k=args.top_k,
        )
    
    # Print summary
    if eval_results:
        print_final_summary(artifacts, eval_results, comparison)
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()