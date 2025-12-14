"""
================================================================================
FULL SYSTEM INTEGRATION TEST (PRODUCTION PIPELINE) - V3
================================================================================

This script validates the entire recommender system, end-to-end:

1. Ingestion (CSV via IngestionFactory + DataLoader)
2. Validation
3. Preprocessing (sequence-based histories)
4. Feature extraction (product + customer)
5. Training data construction
6. Temporal train/valid/test split (both methods)
7. Model training (baselines + LightGBM + XGBoost)
8. Standard evaluation (NDCG@k)
9. Discovery-aware evaluation (repurchase vs new items)
10. Model comparison with lift calculation
11. Hyperparameter tuning (Strategy Pattern)
12. Final model selection
13. Inference (RecommenderPredictor)
14. Cold Start (ColdStartHandler)
15. Model save/load
16. Optional MLflow logging
17. Summary and CI-friendly exit code

V3 Changes:
- Tests both temporal and stratified_temporal split methods
- Tests discovery-aware evaluation
- Tests model comparison functionality
- Validates customer coverage metrics

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
from typing import List, Tuple, Any

# Optional MLflow
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None


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
def run_all_tests(data_path: str, skip_tuning: bool = False) -> List[Tuple[str, bool]]:
    """
    Run full end-to-end integration tests.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file used for the training pipeline.
    skip_tuning : bool
        If True, skip hyperparameter tuning test (faster CI runs).

    Returns
    -------
    List of (test_name, passed) tuples.
    """
    results: List[Tuple[str, bool]] = []

    print_header("FULL SYSTEM TEST STARTED (V3 - Discovery Evaluation)")
    logger.info(datetime.now().isoformat())

    # -------------------------------------------------------------------------
    # TEST 1 — INGESTION
    # -------------------------------------------------------------------------
    print_header("TEST 1: DATA INGESTION")

    try:
        from src.data_ingestion.data_loader import IngestionFactory, DataLoader

        ingestion = IngestionFactory.create(source_type="csv", file_path=data_path)
        raw = DataLoader(ingestion).load()

        assert raw.transactions is not None, "Transactions DataFrame is None"
        assert raw.n_rows > 0, "No rows loaded"

        logger.info(f"  Loaded {raw.n_rows:,} rows, {raw.n_customers:,} customers")

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
            strict_mode=False,
        )
        report = validator.validate(raw.transactions)

        assert hasattr(report, "is_valid"), "ValidationReport missing is_valid attribute"

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

        assert prepared.n_customers > 0, "No customers after preprocessing"
        assert prepared.n_products > 0, "No products after preprocessing"
        assert len(prepared.customer_histories) > 0, "No customer histories"

        logger.info(f"  {prepared.n_customers:,} customers, {prepared.n_products:,} products")

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

        assert product_features.popularity is not None, "Popularity is None"
        assert len(product_features.popularity) > 0, "No popularity scores"

        # V3: Check co-occurrence extraction
        n_cooccur = sum(len(v) for v in product_features.cooccurrence.values())
        logger.info(f"  {len(product_features.popularity):,} products, {n_cooccur:,} co-occurrence pairs")

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

        assert customer_profiles is not None, "Customer profiles is None"
        assert len(customer_profiles) > 0, "No customer profiles"

        logger.info(f"  {len(customer_profiles):,} customer profiles")

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
        
        # V3: Set co-occurrence for discovery features
        if hasattr(builder, 'set_cooccurrence'):
            builder.set_cooccurrence(product_features.cooccurrence)
        
        training_data = builder.build(
            prepared_data=prepared,
            customer_profiles=customer_profiles,
            product_features=product_features,
        )

        assert training_data.samples_df is not None, "samples_df is None"
        assert len(training_data.samples_df) > 0, "No training samples"
        assert training_data.feature_names is not None, "feature_names is None"
        assert len(training_data.feature_names) > 0, "No features"

        logger.info(f"  {len(training_data.samples_df):,} samples, {len(training_data.feature_names)} features")

        print_result("TrainingDataBuilder.build", True)
        results.append(("TrainingData", True))
    except Exception as e:
        print_result("TrainingDataBuilder.build", False, str(e))
        results.append(("TrainingData", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 7A — TEMPORAL SPLIT (Standard)
    # -------------------------------------------------------------------------
    print_header("TEST 7A: TEMPORAL SPLIT (Standard)")

    try:
        from src.training.data_splitter import TemporalDataSplitter

        splitter = TemporalDataSplitter(
            valid_ratio=0.1,
            test_ratio=0.2,
            method="temporal",
        )
        split_temporal = splitter.split(training_data, date_column="order_date")

        assert split_temporal.train_df is not None, "train_df is None"
        assert split_temporal.valid_df is not None, "valid_df is None"
        assert split_temporal.test_df is not None, "test_df is None"
        assert split_temporal.n_train_samples > 0, "No training samples"
        assert split_temporal.n_test_samples > 0, "No test samples"
        
        # V3: Check new attributes
        assert hasattr(split_temporal, 'split_method'), "Missing split_method attribute"
        assert hasattr(split_temporal, 'customer_coverage_test'), "Missing customer_coverage_test"
        assert split_temporal.split_method == "temporal", "Wrong split method"

        logger.info(f"  Train: {split_temporal.n_train_samples:,}, Test: {split_temporal.n_test_samples:,}")
        logger.info(f"  Test coverage: {split_temporal.customer_coverage_test:.1%}")

        print_result("TemporalDataSplitter (temporal)", True)
        results.append(("TemporalSplit_Standard", True))
    except Exception as e:
        print_result("TemporalDataSplitter (temporal)", False, str(e))
        results.append(("TemporalSplit_Standard", False))
        return results

    # -------------------------------------------------------------------------
    # TEST 7B — TEMPORAL SPLIT (Stratified)
    # -------------------------------------------------------------------------
    print_header("TEST 7B: TEMPORAL SPLIT (Stratified - Guaranteed Overlap)")

    try:
        from src.training.data_splitter import TemporalDataSplitter

        splitter_stratified = TemporalDataSplitter(
            valid_ratio=0.1,
            test_ratio=0.2,
            method="stratified_temporal",
            min_orders_per_customer=2,
        )
        split = splitter_stratified.split(training_data, date_column="order_date")

        assert split.train_df is not None, "train_df is None"
        assert split.n_train_samples > 0, "No training samples"
        assert split.split_method == "stratified_temporal", "Wrong split method"
        
        # V3: Stratified split should have high coverage
        assert split.customer_coverage_test >= 0.99, (
            f"Stratified split should have ~100% coverage, got {split.customer_coverage_test:.1%}"
        )

        logger.info(f"  Train: {split.n_train_samples:,}, Test: {split.n_test_samples:,}")
        logger.info(f"  Test coverage: {split.customer_coverage_test:.1%} (should be ~100%)")
        logger.info(f"  New customers in test: {split.new_customers_test}")

        print_result("TemporalDataSplitter (stratified_temporal)", True)
        results.append(("TemporalSplit_Stratified", True))
    except Exception as e:
        print_result("TemporalDataSplitter (stratified_temporal)", False, str(e))
        results.append(("TemporalSplit_Stratified", False))
        # Continue with standard split
        split = split_temporal

    # -------------------------------------------------------------------------
    # TEST 7C — SPLIT DATA SUMMARY
    # -------------------------------------------------------------------------
    print_header("TEST 7C: SPLIT DATA SUMMARY")

    try:
        # V3: Test summary() method
        summary = split.summary()
        assert isinstance(summary, str), "summary() should return string"
        assert len(summary) > 0, "summary() returned empty string"
        assert "SPLIT SUMMARY" in summary, "summary() missing header"

        # V3: Test to_dict() method
        split_dict = split.to_dict()
        assert isinstance(split_dict, dict), "to_dict() should return dict"
        assert "split_method" in split_dict, "to_dict() missing split_method"
        assert "customer_coverage_test" in split_dict, "to_dict() missing coverage"

        logger.info("  summary() and to_dict() methods work correctly")

        print_result("SplitData.summary() and to_dict()", True)
        results.append(("SplitData_Methods", True))
    except Exception as e:
        print_result("SplitData.summary() and to_dict()", False, str(e))
        results.append(("SplitData_Methods", False))

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

        assert "ndcg@3" in pop_metrics, "Missing ndcg@3 in popularity metrics"
        assert "ndcg@3" in pf_metrics, "Missing ndcg@3 in personal freq metrics"

        logger.info(f"  Popularity NDCG@3: {pop_metrics['ndcg@3']:.4f}")
        logger.info(f"  PersonalFreq NDCG@3: {pf_metrics['ndcg@3']:.4f}")

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
    lgb_model = None
    xgb_model = None

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

        logger.info(f"  LightGBM NDCG@3: {lgb_metrics['ndcg@3']:.4f}")

        print_result("LightGBMRanker", True)
        results.append(("LightGBM", True))

        lgb_model = lgb
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

        logger.info(f"  XGBoost NDCG@3: {xgb_metrics['ndcg@3']:.4f}")

        print_result("XGBoostRanker", True)
        results.append(("XGBoost", True))

        xgb_model = xgb
        if xgb_metrics["ndcg@3"] > best_score:
            best_model = xgb
            best_score = xgb_metrics["ndcg@3"]
    except Exception as e:
        print_result("XGBoostRanker", False, str(e))
        results.append(("XGBoost", False))

    # -------------------------------------------------------------------------
    # TEST 10 — DISCOVERY-AWARE EVALUATION (V3 NEW)
    # -------------------------------------------------------------------------
    print_header("TEST 10: DISCOVERY-AWARE EVALUATION (V3)")

    try:
        from src.evaluation.metrics import RankingMetrics, EvaluationResult, DiscoveryStats

        evaluator = RankingMetrics(k_values=[1, 3, 5])

        # Test discovery evaluation on best model
        discovery_results = evaluator.evaluate_discovery(
            model=best_model,
            test_df=split.test_df,
            train_df=split.train_df,
            feature_names=split.feature_names,
        )

        # Verify result structure
        assert "overall" in discovery_results, "Missing 'overall' in discovery results"
        assert "repurchase" in discovery_results, "Missing 'repurchase' in discovery results"
        assert "discovery" in discovery_results, "Missing 'discovery' in discovery results"

        # Verify each result is EvaluationResult
        assert isinstance(discovery_results["overall"], EvaluationResult), "overall not EvaluationResult"
        assert isinstance(discovery_results["discovery"], EvaluationResult), "discovery not EvaluationResult"

        # Verify discovery stats
        discovery_eval = discovery_results["discovery"]
        assert discovery_eval.discovery_stats is not None, "Missing discovery_stats"
        assert isinstance(discovery_eval.discovery_stats, DiscoveryStats), "Wrong discovery_stats type"

        ds = discovery_eval.discovery_stats
        logger.info(f"  Overall NDCG@3: {discovery_results['overall'].metrics['ndcg@3']:.4f}")
        logger.info(f"  Repurchase NDCG@3: {discovery_results['repurchase'].metrics['ndcg@3']:.4f}")
        logger.info(f"  Discovery NDCG@3: {discovery_results['discovery'].metrics['ndcg@3']:.4f}")
        logger.info(f"  Discovery hit rate: {ds.discovery_hit_rate:.2%}")
        logger.info(f"  Avg new item rank: {ds.avg_new_item_rank:.1f}")

        # Verify summary() method
        summary = discovery_eval.summary()
        assert isinstance(summary, str), "summary() should return string"
        assert len(summary) > 0, "summary() returned empty string"

        print_result("Discovery-Aware Evaluation", True)
        results.append(("DiscoveryEvaluation", True))
    except Exception as e:
        print_result("Discovery-Aware Evaluation", False, str(e))
        results.append(("DiscoveryEvaluation", False))

    # -------------------------------------------------------------------------
    # TEST 11 — MODEL COMPARISON (V3 NEW)
    # -------------------------------------------------------------------------
    print_header("TEST 11: MODEL COMPARISON (V3)")

    try:
        from src.evaluation.model_comparison import ModelComparator, ComparisonResult

        comparator = ModelComparator(k_values=[1, 3, 5])

        # Build models dict
        models_to_compare = {
            "Popularity": pop,
            "PersonalFreq": pf,
        }
        if lgb_model is not None:
            models_to_compare["LightGBM"] = lgb_model
        if xgb_model is not None:
            models_to_compare["XGBoost"] = xgb_model

        comparison = comparator.compare(
            models=models_to_compare,
            test_df=split.test_df,
            train_df=split.train_df,
            feature_names=split.feature_names,
            baseline_name="Popularity",
        )

        # Verify result structure
        assert isinstance(comparison, ComparisonResult), "compare() should return ComparisonResult"
        assert comparison.comparison_df is not None, "comparison_df is None"
        assert len(comparison.comparison_df) == len(models_to_compare), "Wrong number of models"
        assert comparison.best_overall is not None, "best_overall is None"
        assert comparison.best_discovery is not None, "best_discovery is None"

        # Verify comparison DataFrame has expected columns
        expected_cols = ["overall_ndcg@3", "discovery_ndcg@3"]
        for col in expected_cols:
            assert col in comparison.comparison_df.columns, f"Missing column: {col}"

        # Verify lift calculation
        if "discovery_ndcg@3_lift_pct" in comparison.comparison_df.columns:
            logger.info("  Lift calculation verified")

        logger.info(f"  Best overall: {comparison.best_overall}")
        logger.info(f"  Best discovery: {comparison.best_discovery}")
        logger.info(f"  Models compared: {list(comparison.comparison_df.index)}")

        # Verify summary() method
        summary = comparison.summary()
        assert isinstance(summary, str), "summary() should return string"

        # Verify generate_report() method
        report = comparator.generate_report(comparison)
        assert isinstance(report, str), "generate_report() should return string"
        assert "MODEL COMPARISON REPORT" in report, "Report missing header"

        print_result("Model Comparison", True)
        results.append(("ModelComparison", True))
    except Exception as e:
        print_result("Model Comparison", False, str(e))
        results.append(("ModelComparison", False))

    # -------------------------------------------------------------------------
    # TEST 12 — HYPERPARAMETER TUNING
    # -------------------------------------------------------------------------
    print_header("TEST 12: HYPERPARAMETER TUNING (Strategy Pattern)")

    if skip_tuning:
        logger.info("  Skipping tuning test (--skip-tuning flag)")
        results.append(("Tuning", True))
    else:
        try:
            from src.tuning.hyperparameter_tuning import HyperparameterTuner
            from src.tuning.model_tuning_strategy import (
                LightGBMTuningStrategy,
                XGBoostTuningStrategy,
            )

            tuner = HyperparameterTuner(
                metric="ndcg@3",
                n_trials=2,  # keep small for test
                direction="maximize",
                timeout=21,
            )

            if isinstance(best_model, LightGBMRanker):
                strategy = LightGBMTuningStrategy()
            elif isinstance(best_model, XGBoostRanker):
                strategy = XGBoostTuningStrategy()
            else:
                strategy = LightGBMTuningStrategy()

            tuning_result = tuner.tune(split, strategy)
            assert tuning_result.best_params is not None, "best_params is None"

            logger.info(f"  Best params: {tuning_result.best_params}")

            print_result("HyperparameterTuner.tune()", True)
            results.append(("Tuning", True))
        except Exception as e:
            print_result("HyperparameterTuner.tune()", False, str(e))
            results.append(("Tuning", False))

    # -------------------------------------------------------------------------
    # TEST 13 — INFERENCE (RecommenderPredictor)
    # -------------------------------------------------------------------------
    print_header("TEST 13: INFERENCE (RecommenderPredictor)")

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

        assert prediction is not None, "Prediction is None"
        assert len(prediction.primary_items) > 0, "No primary items"

        logger.info(f"  Recommendations for customer {test_customer}: {prediction.primary_items[:3]}")

        print_result("RecommenderPredictor.recommend()", True)
        results.append(("Inference", True))
    except Exception as e:
        print_result("RecommenderPredictor.recommend()", False, str(e))
        results.append(("Inference", False))

    # -------------------------------------------------------------------------
    # TEST 14 — COLD START
    # -------------------------------------------------------------------------
    print_header("TEST 14: COLD START RECOMMENDATIONS")

    try:
        from src.inference.cold_start import ColdStartHandler

        cold = ColdStartHandler(
            product_features=product_features,
            prepared_data=prepared,
            customer_profiles=customer_profiles,
        )

        recs = cold.recommend(top_k=5)
        assert recs is not None, "Cold start recs is None"
        assert len(recs) > 0, "No cold start recs"

        logger.info(f"  Cold start recommendations: {recs[:3]}")

        print_result("ColdStartHandler.recommend()", True)
        results.append(("ColdStart", True))
    except Exception as e:
        print_result("ColdStartHandler.recommend()", False, str(e))
        results.append(("ColdStart", False))

    # -------------------------------------------------------------------------
    # TEST 15 — MODEL SAVE / LOAD
    # -------------------------------------------------------------------------
    print_header("TEST 15: MODEL SAVE/LOAD")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        path = tmp.name
        tmp.close()

        bundle = {
            "model": best_model,
            "feature_names": split.feature_names,
            "params": getattr(best_model, "get_params", lambda: {})(),
            "split_info": split.to_dict(),  # V3: Include split info
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert "model" in loaded, "Missing 'model' in loaded bundle"
        assert "split_info" in loaded, "Missing 'split_info' in loaded bundle"

        os.remove(path)

        print_result("Model Save/Load", True)
        results.append(("SaveLoad", True))
    except Exception as e:
        print_result("Model Save/Load", False, str(e))
        results.append(("SaveLoad", False))

    # -------------------------------------------------------------------------
    # TEST 16 — MLFLOW (OPTIONAL)
    # -------------------------------------------------------------------------
    print_header("TEST 16: MLFLOW TRACKING (OPTIONAL)")

    try:
        if mlflow is None:
            raise ImportError("MLflow not installed")

        import shutil

        tmpdir = tempfile.mkdtemp()
        tracking_uri = f"sqlite:///{Path(tmpdir).joinpath('mlflow.db')}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("system-test-v3")

        with mlflow.start_run():
            mlflow.log_param("sample_param", "value")
            mlflow.log_param("split_method", split.split_method)
            mlflow.log_metric("score", 0.99)
            mlflow.log_metric("customer_coverage", split.customer_coverage_test)

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

    logger.info("")
    logger.info("Results by test:")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        logger.info(f"  {status}  {name}")

    # V3 specific summary
    logger.info("")
    logger.info("V3 Feature Tests:")
    v3_tests = [
        "TemporalSplit_Stratified",
        "SplitData_Methods",
        "DiscoveryEvaluation",
        "ModelComparison",
    ]
    for test in v3_tests:
        result = next((ok for name, ok in results if name == test), None)
        if result is not None:
            status = "PASS" if result else "FAIL"
            logger.info(f"  {status}  {test}")

    return results


# -----------------------------------------------------------------------------
# CLI ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full system integration test for JR Recommender (V3)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/data_raw.csv",
        help="Path to raw transactions CSV",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning test (faster CI)",
    )
    args = parser.parse_args()

    results = run_all_tests(data_path=args.data, skip_tuning=args.skip_tuning)
    failed = sum(1 for _, ok in results if not ok)

    sys.exit(0 if failed == 0 else 1)