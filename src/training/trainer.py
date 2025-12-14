"""
TRAINING PIPELINE V2 - Discovery Optimized
==========================================

Uses TrainingDataBuilderV2 with:
- Co-occurrence features
- Archetype-product matching
- Stratified negative sampling
- Exploration rate features

This trains a model that can predict DISCOVERY, not just reorders.
"""

import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

from src.data_ingestion.data_loader import IngestionFactory, DataLoader
from src.data_ingestion.data_validator import ValidationFactory, DataValidator
from src.data_ingestion.data_preprocessor import PreprocessingFactory

from src.features.customer_features import CustomerFeatureExtractor
from src.features.product_features import ProductFeatureExtractor
from src.features.training_data_builder import TrainingDataBuilder  # V2!

from src.training.data_splitter import TemporalDataSplitter
from src.models.baseline_models import PopularityRecommender, PersonalFrequencyRecommender
from src.models.lightgbm_ranker import LightGBMRanker
from src.evaluation.metrics import RankingMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train(
    data_path: str,
    do_tuning: bool = False,
    n_trials: int = 20,
    timeout: int = 300
):
    """
    Train recommender with discovery-optimized features.
    
    Key differences from V1:
    1. Uses TrainingDataBuilderV2 with co-occurrence features
    2. Stratified negative sampling
    3. Tracks first-purchase vs reorder statistics
    """

    # ----------------------------------------------------------------------
    # 1. INGESTION
    # ----------------------------------------------------------------------
    logger.info("=== STEP 1: Data Ingestion ===")
    ingestion = IngestionFactory.create(
        source_type="csv",
        file_path=data_path,
        date_columns=["order_date", "first_order_date", "last_order_date"],
    )
    df = DataLoader(ingestion).load().transactions

    # ----------------------------------------------------------------------
    # 2. VALIDATION
    # ----------------------------------------------------------------------
    logger.info("=== STEP 2: Validation ===")
    validator = DataValidator(
        rules=ValidationFactory.default_rules(),
        strict_mode=False
    )
    report = validator.validate(df)

    if not report.is_valid:
        raise ValueError("Dataset validation failed")

    # ----------------------------------------------------------------------
    # 3. PREPROCESSING
    # ----------------------------------------------------------------------
    logger.info("=== STEP 3: Preprocessing ===")
    prepared_data = PreprocessingFactory.create(
        method="sequence",
        min_orders=2
    ).transform(df)

    # ----------------------------------------------------------------------
    # 4. FEATURE EXTRACTION
    # ----------------------------------------------------------------------
    logger.info("=== STEP 4: Feature Extraction ===")
    customer_profiles = CustomerFeatureExtractor().extract(prepared_data)
    product_features = ProductFeatureExtractor().extract(prepared_data)
    
    logger.info(f"  Products: {len(product_features.popularity)}")
    logger.info(f"  Customers: {len(customer_profiles)}")
    logger.info(f"  Co-occurrence pairs: {sum(len(v) for v in product_features.cooccurrence.values())}")

    # ----------------------------------------------------------------------
    # 5. TRAINING DATA (V2 - Discovery Optimized)
    # ----------------------------------------------------------------------
    logger.info("=== STEP 5: Building Training Data (V2) ===")
    
    builder = TrainingDataBuilder(negative_ratio=5)
    builder.set_cooccurrence(product_features.cooccurrence)  # Pass co-occurrence!
    
    training_data = builder.build(
        prepared_data=prepared_data,
        product_features=product_features,
        customer_profiles=customer_profiles
    )
    
    logger.info(f"  Total samples: {len(training_data.samples_df)}")
    logger.info(f"  Features: {len(training_data.feature_names)}")

    # ----------------------------------------------------------------------
    # 6. TEMPORAL SPLIT
    # ----------------------------------------------------------------------
    logger.info("=== STEP 6: Temporal Split ===")
    split = TemporalDataSplitter(test_ratio=0.2).split(training_data)

    train_df = split.train_df
    test_df = split.test_df
    feature_names = split.feature_names

    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    evaluator = RankingMetrics(k_values=[1, 3, 5])

    # ----------------------------------------------------------------------
    # 7. CANDIDATE MODELS
    # ----------------------------------------------------------------------
    logger.info("=== STEP 7: Training Models ===")
    
    candidates = [
        ("Popularity", PopularityRecommender(), False),
        ("PersonalFreq_0.3", PersonalFrequencyRecommender(0.3), False),
        ("LightGBM", LightGBMRanker(), True),
    ]

    results = []

    for name, model, supports_tuning in candidates:
        logger.info(f"Training: {name}")
        model.fit(train_df, feature_names, valid_df=test_df)

        metrics = evaluator.evaluate(model, test_df, feature_names)

        results.append({
            "name": name,
            "model": model,
            "supports_tuning": supports_tuning,
            "metrics": metrics
        })

        logger.info(f"  {name} NDCG@3 = {metrics.metrics['ndcg@3']:.4f}")

    # ----------------------------------------------------------------------
    # 8. SELECT BEST MODEL
    # ----------------------------------------------------------------------
    best = max(results, key=lambda r: r["metrics"].metrics["ndcg@3"])
    best_model = best["model"]
    best_name = best["name"]

    logger.info(f"=== Best model: {best_name} ===")

    # ----------------------------------------------------------------------
    # 9. HYPERPARAMETER TUNING (OPTIONAL)
    # ----------------------------------------------------------------------
    if do_tuning and best["supports_tuning"]:
        logger.info("=== STEP 9: Hyperparameter Tuning ===")
        from recommander_model_202511.recommender_model_202511.src.tuning.hyperparameter_tuning import HyperparameterTuner
        tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout)
        tuning_result = tuner.tune_lightgbm(split)
        best_model = LightGBMRanker.from_params(tuning_result.best_params)
        best_model.fit(train_df, feature_names, valid_df=test_df)
        tuned_metrics = evaluator.evaluate(best_model, test_df, feature_names)
    else:
        tuned_metrics = best["metrics"]

    # ----------------------------------------------------------------------
    # 10. ANALYZE FEATURE IMPORTANCE
    # ----------------------------------------------------------------------
    logger.info("=== Feature Importance (Top 10) ===")
    if hasattr(best_model, 'model') and hasattr(best_model.model, 'feature_importance'):
        importance = best_model.model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat}: {imp:.1f}")
        
        # Check if discovery features are being used
        discovery_feats = ['cooccur_with_history', 'cooccur_max', 'archetype_product_match']
        logger.info("=== Discovery Feature Importance ===")
        for feat in discovery_feats:
            idx = feature_names.index(feat) if feat in feature_names else -1
            if idx >= 0:
                logger.info(f"  {feat}: {importance[idx]:.1f}")

    # ----------------------------------------------------------------------
    # 11. SAVE ARTIFACTS
    # ----------------------------------------------------------------------
    logger.info("=== STEP 11: Saving Artifacts ===")
    out_dir = Path("models/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name ,
        "feature_names": feature_names,
        "metrics": tuned_metrics.metrics,
        "training_date": datetime.now().isoformat(),
        "version": "discovery_optimized",
    }

    # Save main bundle
    with open(out_dir / "recommender.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # Save supporting artifacts
    with open(out_dir / "product_features.pkl", "wb") as f:
        pickle.dump(product_features, f)

    with open(out_dir / "customer_profiles.pkl", "wb") as f:
        pickle.dump(customer_profiles, f)

    with open(out_dir / "prepared_data.pkl", "wb") as f:
        pickle.dump(prepared_data, f)

    # Save model info
    model_info = {
        "model_name": best_name ,
        "version": "discovery_optimized",
        "training_date": datetime.now().isoformat(),
        "metrics": tuned_metrics.metrics,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_customers": len(customer_profiles),
        "n_products": len(product_features.popularity),
    }

    with open(out_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info("=== Training Complete ===")
    logger.info(f"Artifacts saved to {out_dir}")

    return bundle


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/jr_cafe_data.csv"
    train(data_path, do_tuning=False)