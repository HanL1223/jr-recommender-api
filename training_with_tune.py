"""
Training Pipeline (Production-Ready) - V2 Discovery Optimized
--------------------------------------------------------------
Implements full model training workflow with discovery features:

1. Ingestion (ABC: CSV / BigQuery)
2. Validation
3. Preprocessing
4. Feature extraction
5. Training data construction (V2 - with co-occurrence & archetype features)
6. Temporal split
7. Train baseline + ML models
8. Select best pre-tuning model
9. Hyperparameter tuning (Strategy Pattern)
10. Final model selection
11. Artifact saving

V2 Changes:
- Uses TrainingDataBuilderV2 with discovery features
- Passes co-occurrence matrix for cross-sell signal
- Stratified negative sampling
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime

# Import from src
from src.data_ingestion.data_loader import IngestionFactory, DataLoader
from src.data_ingestion.data_validator import ValidationFactory, DataValidator
from src.data_ingestion.data_preprocessor import PreprocessingFactory
from src.features.customer_features import CustomerFeatureExtractor
from src.features.product_features import ProductFeatureExtractor

# V2: Use discovery-optimized training data builder
from src.features.training_data_builder import TrainingDataBuilder

from src.training.data_splitter import TemporalDataSplitter

from src.models.baseline_models import (
    PopularityRecommender,
    PersonalFrequencyRecommender,
)
from src.models.lightgbm_ranker import LightGBMRanker
from src.models.xgboost_ranker import XGBoostRanker

from src.evaluation.metrics import RankingMetrics

from src.tuning.hyperparameter_tuning import HyperparameterTuner
from src.tuning.model_tuning_strategy import (
    LightGBMTuningStrategy,
    XGBoostTuningStrategy,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Main Pipeline

def run_training_pipeline(
    source_type: str = "csv",
    file_path: str | None = None,
    project_id: str | None = None,
    query: str | None = None,
    tune: bool = False,
    n_trials: int = 20,
    timeout: int = 600,
    strict_validation: bool = False,
) -> dict:
    """
    Execute the full training pipeline.
    """

    # STEP 1 — INGEST DATA
    
    logger.info("STEP 1: Ingestion")

    ingestion_args = {"source_type": source_type}

    if source_type == "csv":
        if not file_path:
            raise ValueError("file_path is required for CSV ingestion.")
        ingestion_args["file_path"] = file_path

    elif source_type == "bigquery":
        if not project_id or not query:
            raise ValueError("project_id and query are required for BigQuery ingestion.")
        ingestion_args["project_id"] = project_id
        ingestion_args["query"] = query

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    ingestion = IngestionFactory.create(**ingestion_args)
    raw = DataLoader(ingestion).load()
    df = raw.transactions

    logger.info(
        "Loaded %s rows, %s customers, %s orders",
        f"{raw.n_rows:,}",
        f"{raw.n_customers:,}",
        f"{raw.n_orders:,}",
    )


    # STEP 2 — VALIDATE DATA

    logger.info("STEP 2: Validation")

    validator = DataValidator(
        rules=ValidationFactory.default_rules(),
        strict_mode=strict_validation,
    )
    report = validator.validate(df)

    if not getattr(report, "is_valid", False):
        logger.error("Validation failed. Halting pipeline.")
        return {"status": "failed_validation", "validation_report": report}


    # STEP 3 — PREPROCESSING

    logger.info("STEP 3: Preprocessing")

    preprocessor = PreprocessingFactory.create(method="sequence", min_orders=2)
    prepared = preprocessor.transform(df)

    logger.info(
        "Prepared dataset contains %s customers, %s products, %s orders",
        f"{prepared.n_customers:,}",
        f"{prepared.n_products:,}",
        f"{prepared.n_orders:,}",
    )


    # STEP 4 — FEATURE EXTRACTION

    logger.info("STEP 4: Feature extraction")

    cust_ext = CustomerFeatureExtractor()
    customer_profiles = cust_ext.extract(prepared)

    prod_ext = ProductFeatureExtractor()
    product_features = prod_ext.extract(prepared)

    # V2: Log co-occurrence stats
    n_cooccur = sum(len(v) for v in product_features.cooccurrence.values())
    logger.info(
        "Extracted features for %s customers, %s products, %s co-occurrence pairs",
        f"{len(customer_profiles):,}",
        f"{len(product_features.popularity):,}",
        f"{n_cooccur:,}",
    )


    # STEP 5 — TRAINING DATA BUILD (V2)

    logger.info("STEP 5: Building training samples (V2 - discovery optimized)")

    # V2: Use discovery-optimized builder with co-occurrence
    builder = TrainingDataBuilder(negative_ratio=5)
    builder.set_cooccurrence(product_features.cooccurrence)  # KEY: Pass co-occurrence!
    
    training_data = builder.build(
        prepared_data=prepared,
        customer_profiles=customer_profiles,
        product_features=product_features,
    )

    logger.info(
        "Training samples: %s rows and %s feature columns",
        f"{len(training_data.samples_df):,}",
        f"{len(training_data.feature_names):,}",
    )
    
    # V2: Log new discovery features
    logger.info("Discovery features included: cooccur_with_history, cooccur_max, archetype_product_match")


    # STEP 6 — TEMPORAL SPLIT

    logger.info("STEP 6: Train/Valid/Test split")

    splitter = TemporalDataSplitter(valid_ratio=0.1, test_ratio=0.2)
    split = splitter.split(training_data, date_column="order_date")

    train_df = split.train_df
    valid_df = split.valid_df
    test_df = split.test_df
    feature_names = split.feature_names

    logger.info(
        "Split sizes: train=%s, valid=%s, test=%s",
        f"{split.n_train_samples:,}",
        f"{split.n_valid_samples:,}",
        f"{split.n_test_samples:,}",
    )


    # STEP 7 — TRAIN CANDIDATE MODELS

    logger.info("STEP 7: Training candidate models")

    evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
    candidates = []

    def train_and_record(model, name):
        logger.info("Training: %s", name)

        try:
            model.fit(train_df, feature_names, valid_df)
        except TypeError:
            model.fit(train_df, feature_names)

        valid_scores = evaluator.evaluate(model, valid_df, feature_names).metrics
        test_scores = evaluator.evaluate(model, test_df, feature_names).metrics

        logger.info(
            "%s | VALID NDCG@3=%.4f | TEST NDCG@3=%.4f",
            name,
            valid_scores["ndcg@3"],
            test_scores["ndcg@3"],
        )

        candidates.append(
            {
                "name": name,
                "model": model,
                "valid_metrics": valid_scores,
                "test_metrics": test_scores,
                "params": getattr(model, "get_params", lambda: {})(),
            }
        )

    # Baselines
    train_and_record(PopularityRecommender(), "Popularity")
    train_and_record(PersonalFrequencyRecommender(0.2), "PersonalFreq_0.2")
    train_and_record(PersonalFrequencyRecommender(0.3), "PersonalFreq_0.3")
    train_and_record(PersonalFrequencyRecommender(0.5), "PersonalFreq_0.5")

    # LightGBM
    try:
        train_and_record(
            LightGBMRanker(
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                min_data_in_leaf=20,
                num_boost_round=300,
                early_stopping_rounds=50,
            ),
            "LightGBM_default",
        )
    except Exception as e:
        logger.warning("LightGBM unavailable: %s", e)

    # XGBoost
    try:
        train_and_record(
            XGBoostRanker(
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=500,
            ),
            "XGBoost_default",
        )
    except Exception as e:
        logger.warning("XGBoost unavailable: %s", e)


    # STEP 8 — BEST MODEL (PRE-TUNING)

    logger.info("STEP 8: Selecting best pre-tuning model")

    candidates.sort(key=lambda m: m["valid_metrics"]["ndcg@3"], reverse=True)
    best_pre = candidates[0]

    logger.info(
        "Selected model: %s (NDCG@3=%.4f)",
        best_pre["name"],
        best_pre["valid_metrics"]["ndcg@3"],
    )


    # STEP 9 — HYPERPARAMETER TUNING (Strategy Pattern)

    logger.info("STEP 9: Hyperparameter tuning")

    strategy_map = {
        "LightGBM_default": LightGBMTuningStrategy,
        "LightGBMRanker": LightGBMTuningStrategy,
        "XGBoost_default": XGBoostTuningStrategy,
        "XGBoost": XGBoostTuningStrategy,
    }

    tuned_entry = None
    tuning_result = None

    if tune and best_pre["name"] in strategy_map:
        strategy = strategy_map[best_pre["name"]]()
        tuner = HyperparameterTuner(
            metric="ndcg@3",
            direction="maximize",
            n_trials=n_trials,
            timeout=timeout,
        )

        tuning_result = tuner.tune(split, strategy)
        best_params = tuning_result.best_params

        tuned_model = strategy.create_model(best_params)
        tuned_model.fit(train_df, feature_names, valid_df)

        tuned_valid = evaluator.evaluate(tuned_model, valid_df, feature_names).metrics
        tuned_test = evaluator.evaluate(tuned_model, test_df, feature_names).metrics

        tuned_entry = {
            "name": f"{best_pre['name']}_tuned",
            "model": tuned_model,
            "valid_metrics": tuned_valid,
            "test_metrics": tuned_test,
            "params": best_params,
        }
        candidates.append(tuned_entry)


    # STEP 10 — FINAL MODEL SELECTION

    logger.info("STEP 10: Selecting final model")

    candidates.sort(key=lambda m: m["valid_metrics"]["ndcg@3"], reverse=True)
    best_final = candidates[0]

    logger.info(
        "Final model: %s (VALID NDCG@3=%.4f, TEST NDCG@3=%.4f)",
        best_final["name"],
        best_final["valid_metrics"]["ndcg@3"],
        best_final["test_metrics"]["ndcg@3"],
    )


    # STEP 11 — SAVE ARTIFACTS
    logger.info("STEP 11: Saving model artifacts")

    output_dir = Path("models/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "recommender.pkl"
    info_path = output_dir / "model_info.json"

    # Save trained model
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_final["model"],
                "feature_names": feature_names,
                "params": best_final["params"],
            },
            f,
        )

    # Save metadata
    model_info = {
        "model_name": best_final["name"],
        "training_date": datetime.now().isoformat(),
        "valid_metrics": best_final["valid_metrics"],
        "test_metrics": best_final["test_metrics"],
        "params": best_final["params"],
        "tuning_enabled": tune,
        "tuning_trials": n_trials if tune else 0,
        "all_models": [c["name"] for c in candidates],
        "version": "discovery_optimized",  # V2: Add version tag
    }

    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info("Model + metadata saved.")

    # Save additional required artifacts
    with open(output_dir / "product_features.pkl", "wb") as f:
        pickle.dump(product_features, f)

    with open(output_dir / "customer_profiles.pkl", "wb") as f:
        pickle.dump(customer_profiles, f)

    with open(output_dir / "prepared_data.pkl", "wb") as f:
        pickle.dump(prepared, f)

    with open(output_dir / "encoders.pkl", "wb") as f:
        pickle.dump(training_data.encoders, f)

    with open(output_dir / "category_map.pkl", "wb") as f:
        pickle.dump(prepared.category_map, f)

    logger.info("Saved: product_features, customer_profiles, prepared_data, encoders, category_map")


    # STEP 12 — FEATURE IMPORTANCE ANALYSIS (V2)
    
    logger.info("STEP 12: Feature importance analysis")
    
    if hasattr(best_final["model"], 'model') and hasattr(best_final["model"].model, 'feature_importance'):
        importance = best_final["model"].model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        
        logger.info("\nTop 10 Features by Importance:")
        logger.info("-" * 40)
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat:30s}: {imp:,.1f}")
        
        # V2: Check discovery features specifically
        discovery_feats = ['cooccur_with_history', 'cooccur_max', 'archetype_product_match']
        logger.info("\nDiscovery Feature Importance:")
        logger.info("-" * 40)
        for feat in discovery_feats:
            if feat in feature_names:
                idx = feature_names.index(feat)
                logger.info(f"  {feat:30s}: {importance[idx]:,.1f}")
            else:
                logger.info(f"  {feat:30s}: NOT FOUND")

 
    # STEP 13 — PROFESSIONAL HUMAN-READABLE SUMMARY

    logger.info("\n" + "=" * 72)
    logger.info("FINAL TRAINING SUMMARY (V2 - Discovery Optimized)")
    logger.info("=" * 72)

    logger.info("Best Model")
    logger.info("-----------")
    logger.info(f"Name               : {best_final['name']}")
    logger.info(f"Validation NDCG@3  : {best_final['valid_metrics']['ndcg@3']:.4f}")
    logger.info(f"Test NDCG@3        : {best_final['test_metrics']['ndcg@3']:.4f}")

    logger.info("\nBest Hyperparameters")
    logger.info("---------------------")
    if best_final["params"]:
        for k, v in best_final["params"].items():
            if isinstance(v, float):
                logger.info(f"{k:22s}: {v:.6f}")
            else:
                logger.info(f"{k:22s}: {v}")
    else:
        logger.info("Model has no trainable hyperparameters (baseline model).")

    logger.info("\nModel Artifact Paths")
    logger.info("--------------------")
    logger.info(f"Serialized Model   : {model_path}")
    logger.info(f"Model Metadata JSON: {info_path}")

    logger.info("\nTraining Summary Complete.")
    logger.info("=" * 72 + "\n")

    # FINAL RETURN
    return {
        "status": "success",
        "best_model": best_final["model"],
        "best_model_name": best_final["name"],
        "best_valid_metrics": best_final["valid_metrics"],
        "best_test_metrics": best_final["test_metrics"],
        "best_params": best_final["params"],
        "tuning_result": tuning_result,
        "artifacts": {
            "model_path": str(model_path),
            "info_path": str(info_path),
        },
    }
       


# CLI ENTRYPOINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JR Recommender model (V2)")

    parser.add_argument("--source_type", default="csv")
    parser.add_argument("--file_path")
    parser.add_argument("--project_id")
    parser.add_argument("--query")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--strict_validation", action="store_true")

    args = parser.parse_args()

    run_training_pipeline(
        source_type=args.source_type,
        file_path=args.file_path,
        project_id=args.project_id,
        query=args.query,
        tune=args.tune,
        n_trials=args.trials,
        timeout=args.timeout,
        strict_validation=args.strict_validation,
    )