"""
TRAINING PIPELINE
=================
Clean end-to-end trainer aligned with ColdStartHandler + RecommenderPredictor.
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
from src.features.training_data_builder import TrainingDataBuilder

from src.training.data_splitter import TemporalDataSplitter
from src.models.baseline_models import PopularityRecommender, PersonalFrequencyRecommender
from src.models.lightgbm_ranker import LightGBMRanker
from recommander_model_202511.recommender_model_202511.src.tuning.hyperparameter_tuning import HyperparameterTuner
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
    Trains the best recommender model and stores all components
    required for predictor + cold-start inference.
    """

    # ----------------------------------------------------------------------
    # 1. INGESTION
    # ----------------------------------------------------------------------
    ingestion = IngestionFactory.create(
        source_type="csv",
        file_path=data_path,
        date_columns=["order_date", "first_order_date", "last_order_date"],
    )
    df = DataLoader(ingestion).load().transactions

    # ----------------------------------------------------------------------
    # 2. VALIDATION
    # ----------------------------------------------------------------------
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
    prepared_data = PreprocessingFactory.create(
        method="sequence",
        min_orders=2
    ).transform(df)

    # ----------------------------------------------------------------------
    # 4. FEATURE EXTRACTION
    # ----------------------------------------------------------------------
    customer_profiles = CustomerFeatureExtractor().extract(prepared_data)
    product_features = ProductFeatureExtractor().extract(prepared_data)

    # ----------------------------------------------------------------------
    # 5. TRAINING DATA
    # ----------------------------------------------------------------------
    builder = TrainingDataBuilder(negative_ratio=5)
    training_data = builder.build(
        prepared_data=prepared_data,
        product_features=product_features,
        customer_profiles=customer_profiles
    )

    # ----------------------------------------------------------------------
    # 6. TEMPORAL SPLIT
    # ----------------------------------------------------------------------
    split = TemporalDataSplitter(test_ratio=0.2).split(training_data)

    train_df = split.train_df
    test_df = split.test_df
    feature_names = split.feature_names

    evaluator = RankingMetrics(k_values=[1, 3, 5])
    tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout)

    # ----------------------------------------------------------------------
    # 7. CANDIDATE MODELS
    # ----------------------------------------------------------------------
    candidates = [
        ("Popularity", PopularityRecommender(), False),
        ("PersonalFreq_0.3", PersonalFrequencyRecommender(0.3), False),
        ("LightGBM_default", LightGBMRanker(), True),
    ]

    results = []

    for name, model, supports_tuning in candidates:
        logger.info(f"Training model: {name}")
        model.fit(train_df, feature_names, valid_df=test_df)

        metrics = evaluator.evaluate(model, test_df, feature_names)

        results.append({
            "name": name,
            "model": model,
            "supports_tuning": supports_tuning,
            "metrics": metrics
        })

        logger.info(f"{name} NDCG@3 = {metrics.metrics['ndcg@3']:.4f}")

    # ----------------------------------------------------------------------
    # 8. SELECT BEST BASE MODEL
    # ----------------------------------------------------------------------
    best = max(results, key=lambda r: r["metrics"].metrics["ndcg@3"])
    best_model = best["model"]
    best_name = best["name"]

    logger.info(f"Best base model: {best_name}")

    # ----------------------------------------------------------------------
    # 9. HYPERPARAMETER TUNING (OPTIONAL)
    # ----------------------------------------------------------------------
    if do_tuning and best["supports_tuning"]:
        tuning_result = tuner.tune_lightgbm(split)
        best_model = LightGBMRanker.from_params(tuning_result.best_params)
        best_model.fit(train_df, feature_names, valid_df=test_df)

        tuned_metrics = evaluator.evaluate(best_model, test_df, feature_names)
    else:
        tuned_metrics = best["metrics"]

    # ----------------------------------------------------------------------
    # 10. SAVE ARTIFACTS FOR INFERENCE
    # ----------------------------------------------------------------------
    out_dir = Path("models/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "metrics": tuned_metrics.metrics,
        "customer_profiles": customer_profiles,
        "product_features": product_features,
        "prepared_data": prepared_data,
        "training_date": datetime.now().isoformat(),
    }

    with open(out_dir / "recommender.pkl", "wb") as f:
        pickle.dump(bundle, f)

    logger.info("Training pipeline complete. Saved recommender.pkl")

    return bundle
