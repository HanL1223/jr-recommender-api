"""
demo.py
=======

Demonstrates inference using the trained JR recommender.

Loads:
- Trained model (ranker)
- Product features
- PreparedData
- Customer profiles
- ColdStartHandler
- RecommenderPredictor

Runs 6 representative test cases.
"""

import sys
from pathlib import Path
import json
import pickle
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.recommender_predictor import RecommenderPredictor
from src.inference.cold_start import ColdStartHandler

ARTIFACT_DIR = Path("models/artifacts")
MODEL_PATH = ARTIFACT_DIR / "recommender.pkl"
INFO_PATH = ARTIFACT_DIR / "model_info.json"
PRODUCT_FEATURES_PATH = ARTIFACT_DIR / "product_features.pkl"
CUSTOMER_PROFILES_PATH = ARTIFACT_DIR / "customer_profiles.pkl"
PREPARED_DATA_PATH = ARTIFACT_DIR / "prepared_data.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("DEMO")


# ---------------------------------------------------------------------
# Printing helper
# ---------------------------------------------------------------------
def pretty_print(pred, title):
    print(f"\n==================== {title} ====================")
    print(f"MODEL USED : {pred.model_used}")
    print(f"CUSTOMER   : {pred.customer_id}")
    print("-------------------------------------------------")
    print("PRIMARY RECOMMENDATIONS:")
    for i, item in enumerate(pred.primary_items, start=1):
        print(f"  {i}. {item.product:30s} score={item.score:.4f}  reason={item.reason}")

    if pred.addon_items:
        print("\nADD-ON SUGGESTIONS:")
        for item in pred.addon_items:
            print(f"  + {item.product:30s} score={item.score:.4f}  reason={item.reason}")

    print("=================================================\n")


# ---------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------
def load_artifacts():
    bundle = pickle.load(open(MODEL_PATH, "rb"))
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    product_features = pickle.load(open(PRODUCT_FEATURES_PATH, "rb"))
    customer_profiles = pickle.load(open(CUSTOMER_PROFILES_PATH, "rb"))
    prepared_data = pickle.load(open(PREPARED_DATA_PATH, "rb"))

    info = json.load(open(INFO_PATH))

    logger.info("Loaded trained model: %s", info["model_name"])
    logger.info("VALID NDCG@3: %.4f", info["valid_metrics"]["ndcg@3"])
    logger.info("TEST  NDCG@3: %.4f", info["test_metrics"]["ndcg@3"])

    return model, feature_names, product_features, customer_profiles, prepared_data


# ---------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------
def main():
    ml_model, feature_names, product_features, customer_profiles, prepared_data = load_artifacts()

    cold_handler = ColdStartHandler(
        product_features=product_features,
        prepared_data=prepared_data,
        customer_profiles=customer_profiles,
    )

    predictor = RecommenderPredictor(
        ml_model=ml_model,
        baseline_model=None,
        product_features=product_features,
        prepared_data=prepared_data,
        customer_profiles=customer_profiles,
        feature_names=feature_names,
        cold_start_handler=cold_handler,
    )

    # CASE 1 — Known customer
    known = next(iter(prepared_data.customer_histories.keys()))
    pretty_print(predictor.recommend(known, 5), "CASE 1 — Known Customer")

    # CASE 2 — Long history
    long_hist = max(prepared_data.customer_histories, key=lambda c: len(prepared_data.customer_histories[c]))
    pretty_print(predictor.recommend(long_hist, 5), "CASE 2 — Long History Customer")

    # CASE 3 — Short history
    short_hist = min(prepared_data.customer_histories, key=lambda c: len(prepared_data.customer_histories[c]))
    pretty_print(predictor.recommend(short_hist, 5), "CASE 3 — Short History Customer")

    # CASE 4 — Cold start
    pretty_print(predictor.recommend(99999999, 5), "CASE 4 — Cold Start (Unknown Customer)")

    # CASE 5 — Archetype-driven cold start
    latte_items = cold_handler.recommend(archetype_hint="latte_lover", time_of_day=9, top_k=5)
    print("\n===== CASE 5 — Archetype Cold Start: LATTE LOVER =====")
    for item in latte_items:
        print(f"- {item.product:30s} score={item.score:.4f} reason={item.reason}")

    # CASE 6 — Segment-driven inference override
    prepared_data.customer_histories[known][0]["segment"] = "VIP"
    pretty_print(predictor.recommend(known, 5), "CASE 6 — Segment-driven Recommendation (VIP)")

    print("\nAll demo test cases completed successfully.")


if __name__ == "__main__":
    main()
