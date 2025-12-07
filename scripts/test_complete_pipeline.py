"""
================================================================================
FULL SYSTEM INTEGRATION TEST (PRODUCTION PIPELINE)
================================================================================

This script validates the entire recommender system, end-to-end:

1. Ingestion (CSV via IngestionFactory + DataLoader)
2. Validation
3. Preprocessing (sequence-based histories)
4. Feature extraction (product + customer)
5. Training data construction
6. Temporal train/valid/test split
7. Model training (baselines + LightGBM + XGBoost)
8. Evaluation (NDCG@k)
9. Hyperparameter tuning (Strategy Pattern)
10. Final model selection
11. Inference (RecommenderPredictor)
12. Cold Start (ColdStartHandler)
13. Model save/load
14. Optional MLflow logging
15. Summary and CI-friendly exit code

Run (from project root):
    python scripts/test_complete_pipeline.py --data data/raw/data_raw.csv
"""

import sys
import os
import logging
import tempfile
import pickle
from pathlib import Path
from datetime import datetime

# Optional MLflow
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None  # gracefully handle absence


# -----------------------------------------------------------------------------
# Make `src/` importable
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SYSTEM_TEST")


def print_header(text: str):
    logger.info("\n" + "=" * 72)
    logger.info(text)
    logger.info("=" * 72)


def print_result(name: str, ok: bool, details: str = ""):
    status = "PASS" if ok else "FAIL"
    logger.info(f"{status} - {name}")
    if details and not ok:
        logger.info(f"  Details: {details}")


# -----------------------------------------------------------------------------
# MAIN TEST FUNCTION
# -----------------------------------------------------------------------------
def run_all_tests(data_path: str):
    """
    Run full end-to-end integration tests.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file used for the training pipeline.
    """
    results: list[tuple[str, bool]] = []

    print_header("FULL SYSTEM TEST STARTED")
    logger.info(datetime.now().isoformat())

    # -------------------------------------------------------------------------
    # TEST 1 — INGESTION
    # -------------------------------------------------------------------------
    print_header("TEST 1: DATA INGESTION")

    try:
        from src.data_ingestion.data_loader import IngestionFactory, DataLoader

        ingestion = IngestionFactory.create(source_type="csv", file_path=data_path)
        raw = DataLoader(ingestion).load()

        assert raw.transactions is not None
        assert raw.n_rows > 0

        print_result("IngestionFactory + DataLoader", True)
        results.append(("Ingestion", True))
    except Exception as e:
        print_result("IngestionFactory + DataLoader", False, str(e))
        results.append(("Ingestion", False))
        return results  # hard fail – can't proceed without data

    # -------------------------------------------------------------------------
    # TEST 2 — VALIDATION
    # -------------------------------------------------------------------------
    print_header("TEST 2: DATA VALIDATION")

    try:
        from src.data_ingestion.data_validator import ValidationFactory, DataValidator

        validator = DataValidator(
            rules=ValidationFactory.default_rules(),
            strict_mode=False,  # WARNING allowed; ERROR/CRITICAL should fail in pipeline
        )
        report = validator.validate(raw.transactions)

        assert hasattr(report, "is_valid")

        print_result("DataValidator.validate", True)
        results.append(("Validation", True))
    except Exception as e:
        print_result("DataValidator.validate", False, str(e))
        results.append(("Validation", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 3 — PREPROCESSING
    # -------------------------------------------------------------------------
    print_header("TEST 3: PREPROCESSING")

    try:
        from src.data_ingestion.data_preprocessor import PreprocessingFactory

        preprocessor = PreprocessingFactory.create(method="sequence", min_orders=2)
        prepared = preprocessor.transform(raw.transactions)

        assert prepared.n_customers > 0
        assert prepared.n_products > 0
        assert len(prepared.customer_histories) > 0

        print_result("PreprocessingFactory.transform", True)
        results.append(("Preprocessing", True))
    except Exception as e:
        print_result("PreprocessingFactory.transform", False, str(e))
        results.append(("Preprocessing", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 4 — PRODUCT FEATURES
    # -------------------------------------------------------------------------
    print_header("TEST 4: PRODUCT FEATURE EXTRACTION")

    try:
        from src.features.product_features import ProductFeatureExtractor

        prod_ext = ProductFeatureExtractor()
        product_features = prod_ext.extract(prepared)

        assert product_features.popularity is not None
        assert len(product_features.popularity) > 0

        print_result("ProductFeatureExtractor.extract", True)
        results.append(("ProductFeatures", True))
    except Exception as e:
        print_result("ProductFeatureExtractor.extract", False, str(e))
        results.append(("ProductFeatures", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 5 — CUSTOMER FEATURES
    # -------------------------------------------------------------------------
    print_header("TEST 5: CUSTOMER FEATURE EXTRACTION")

    try:
        from src.features.customer_features import CustomerFeatureExtractor

        cust_ext = CustomerFeatureExtractor()
        customer_profiles = cust_ext.extract(prepared)

        assert customer_profiles is not None
        assert len(customer_profiles) > 0

        print_result("CustomerFeatureExtractor.extract", True)
        results.append(("CustomerFeatures", True))
    except Exception as e:
        print_result("CustomerFeatureExtractor.extract", False, str(e))
        results.append(("CustomerFeatures", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 6 — TRAINING DATA BUILDER
    # -------------------------------------------------------------------------
    print_header("TEST 6: TRAINING DATA BUILDER")

    try:
        from src.features.training_data_builder import TrainingDataBuilder

        builder = TrainingDataBuilder(negative_ratio=5)
        training_data = builder.build(
            prepared_data=prepared,
            customer_profiles=customer_profiles,
            product_features=product_features,
        )

        assert training_data.samples_df is not None
        assert len(training_data.samples_df) > 0
        assert training_data.feature_names is not None
        assert len(training_data.feature_names) > 0

        print_result("TrainingDataBuilder.build", True)
        results.append(("TrainingData", True))
    except Exception as e:
        print_result("TrainingDataBuilder.build", False, str(e))
        results.append(("TrainingData", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 7 — TEMPORAL SPLIT
    # -------------------------------------------------------------------------
    print_header("TEST 7: TEMPORAL TRAIN/VALID/TEST SPLIT")

    try:
        from src.training.data_splitter import TemporalDataSplitter

        splitter = TemporalDataSplitter(valid_ratio=0.1, test_ratio=0.2)
        split = splitter.split(training_data, date_column="order_date")

        assert split.train_df is not None
        assert split.valid_df is not None
        assert split.test_df is not None
        assert split.n_train_samples > 0
        assert split.n_test_samples > 0

        print_result("TemporalDataSplitter.split", True)
        results.append(("TemporalSplit", True))
    except Exception as e:
        print_result("TemporalDataSplitter.split", False, str(e))
        results.append(("TemporalSplit", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 8 — BASELINE MODELS
    # -------------------------------------------------------------------------
    print_header("TEST 8: BASELINE MODELS")

    try:
        from src.models.baseline_models import (
            PopularityRecommender,
            PersonalFrequencyRecommender,
        )
        from src.evaluation.metrics import RankingMetrics

        evaluator = RankingMetrics(k_values=[1, 3, 5])

        pop = PopularityRecommender()
        pop.fit(split.train_df, split.feature_names)
        pop_eval = evaluator.evaluate(pop, split.test_df, split.feature_names)
        pop_metrics = pop_eval.metrics

        pf = PersonalFrequencyRecommender(smoothing=0.3)
        pf.fit(split.train_df, split.feature_names)
        pf_eval = evaluator.evaluate(pf, split.test_df, split.feature_names)
        pf_metrics = pf_eval.metrics

        assert "ndcg@3" in pop_metrics
        assert "ndcg@3" in pf_metrics

        print_result("Baseline Models (Popularity + PersonalFrequency)", True)
        results.append(("Baselines", True))
    except Exception as e:
        print_result("Baseline Models", False, str(e))
        results.append(("Baselines", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 9 — ML MODELS: LIGHTGBM + XGBOOST
    # -------------------------------------------------------------------------
    print_header("TEST 9: ML MODELS (LightGBM + XGBoost)")

    from src.models.lightgbm_ranker import LightGBMRanker
    from src.models.xgboost_ranker import XGBoostRanker

    best_model = pf
    best_score = pf_metrics["ndcg@3"]

    # LightGBM
    try:
        lgb = LightGBMRanker(
            num_leaves=3,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_data_in_leaf=20,
            num_boost_round=5,
            early_stopping_rounds=3,
        )
        lgb.fit(split.train_df, split.feature_names, split.valid_df)
        lgb_eval = evaluator.evaluate(lgb, split.test_df, split.feature_names)
        lgb_metrics = lgb_eval.metrics

        print_result("LightGBMRanker", True)
        results.append(("LightGBM", True))

        if lgb_metrics["ndcg@3"] > best_score:
            best_model = lgb
            best_score = lgb_metrics["ndcg@3"]
    except Exception as e:
        print_result("LightGBMRanker", False, str(e))
        results.append(("LightGBM", False))

    # XGBoost
    try:
        xgb = XGBoostRanker(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=10,
        )
        xgb.fit(split.train_df, split.feature_names, split.valid_df)
        xgb_eval = evaluator.evaluate(xgb, split.test_df, split.feature_names)
        xgb_metrics = xgb_eval.metrics

        print_result("XGBoostRanker", True)
        results.append(("XGBoost", True))

        if xgb_metrics["ndcg@3"] > best_score:
            best_model = xgb
            best_score = xgb_metrics["ndcg@3"]
    except Exception as e:
        print_result("XGBoostRanker", False, str(e))
        results.append(("XGBoost", False))

    # -------------------------------------------------------------------------
    # TEST 10 — HYPERPARAMETER TUNING
    # -------------------------------------------------------------------------
    print_header("TEST 10: HYPERPARAMETER TUNING (Strategy Pattern)")

    try:
        from src.tuning.hyperparameter_tuning import HyperparameterTuner
        from src.tuning.model_tuning_strategy import (
            LightGBMTuningStrategy,
            XGBoostTuningStrategy,
        )

        tuner = HyperparameterTuner(
            metric="ndcg@3",
            n_trials=2,          # keep small for test
            direction="maximize",
            timeout=21,
        )

        if isinstance(best_model, LightGBMRanker):
            strategy = LightGBMTuningStrategy()
        elif isinstance(best_model, XGBoostRanker):
            strategy = XGBoostTuningStrategy()
        else:
            # fall back: try LightGBM strategy even if baseline was best
            strategy = LightGBMTuningStrategy()

        tuning_result = tuner.tune(split, strategy)
        assert tuning_result.best_params is not None

        print_result("HyperparameterTuner.tune()", True)
        results.append(("Tuning", True))
    except Exception as e:
        print_result("HyperparameterTuner.tune()", False, str(e))
        results.append(("Tuning", False))

    # -------------------------------------------------------------------------
    # TEST 11 — INFERENCE (RecommenderPredictor)
    # -------------------------------------------------------------------------
    print_header("TEST 11: INFERENCE (RecommenderPredictor)")

    try:
        from src.inference.recommender_predictor import RecommenderPredictor
        from src.inference.cold_start import ColdStartHandler

        cold_handler = ColdStartHandler(
            product_features=product_features,
            prepared_data=prepared,
            customer_profiles=customer_profiles,
        )

        predictor = RecommenderPredictor(
            ml_model=best_model,
            baseline_model=pf,
            product_features=product_features,
            prepared_data=prepared,
            customer_profiles=customer_profiles,
            feature_names=split.feature_names,
            cold_start_handler=cold_handler,
        )

        test_customer = list(prepared.customer_histories.keys())[0]
        prediction = predictor.recommend(test_customer, top_k=5)

        assert prediction is not None
        assert len(prediction.primary_items) > 0

        print_result("RecommenderPredictor.recommend()", True)
        results.append(("Inference", True))
    except Exception as e:
        print_result("RecommenderPredictor.recommend()", False, str(e))
        results.append(("Inference", False))

    # -------------------------------------------------------------------------
    # TEST 12 — COLD START
    # -------------------------------------------------------------------------
    print_header("TEST 12: COLD START RECOMMENDATIONS")

    try:
        from src.inference.cold_start import ColdStartHandler

        cold = ColdStartHandler(
            product_features=product_features,
            prepared_data=prepared,
            customer_profiles=customer_profiles,
        )

        recs = cold.recommend(top_k=5)
        assert recs is not None
        assert len(recs) > 0

        print_result("ColdStartHandler.recommend()", True)
        results.append(("ColdStart", True))
    except Exception as e:
        print_result("ColdStartHandler.recommend()", False, str(e))
        results.append(("ColdStart", False))

    # -------------------------------------------------------------------------
    # TEST 13 — MODEL SAVE / LOAD
    # -------------------------------------------------------------------------
    print_header("TEST 13: MODEL SAVE/LOAD")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        path = tmp.name
        tmp.close()

        bundle = {
            "model": best_model,
            "feature_names": split.feature_names,
            "params": getattr(best_model, "get_params", lambda: {})(),
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert "model" in loaded

        os.remove(path)

        print_result("Model Save/Load", True)
        results.append(("SaveLoad", True))
    except Exception as e:
        print_result("Model Save/Load", False, str(e))
        results.append(("SaveLoad", False))

    # -------------------------------------------------------------------------
    # TEST 14 — MLFLOW (OPTIONAL)
    # -------------------------------------------------------------------------
    print_header("TEST 14: MLFLOW TRACKING (OPTIONAL)")

    try:
        if mlflow is None:
            raise ImportError("MLflow not installed")

        import shutil

        tmpdir = tempfile.mkdtemp()
        tracking_uri = f"sqlite:///{Path(tmpdir).joinpath('mlflow.db')}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("system-test")

        with mlflow.start_run():
            mlflow.log_param("sample_param", "value")
            mlflow.log_metric("score", 0.99)

        print_result("MLflow Tracking", True)
        results.append(("MLflow", True))

        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        print_result("MLflow Tracking", False, str(e))
        results.append(("MLflow", False))

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print_header("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    logger.info(f"Total tests : {total}")
    logger.info(f"Passed      : {passed}")
    logger.info(f"Failed      : {failed}")
    logger.info(f"Success rate: {passed / total:.1%}")

    for name, ok in results:
        logger.info(f"{'PASS' if ok else 'FAIL'}  {name}")

    return results


# -----------------------------------------------------------------------------
# CLI ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full system integration test for JR Recommender"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/data_raw.csv",
        help="Path to raw transactions CSV",
    )
    args = parser.parse_args()

    results = run_all_tests(data_path=args.data)
    failed = sum(1 for _, ok in results if not ok)

    sys.exit(0 if failed == 0 else 1)
