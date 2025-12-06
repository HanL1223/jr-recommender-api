import pickle
from pathlib import Path

ARTIFACT_DIR = Path("models/artifacts")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    print("Loading saved artifacts...")

    # Load all components saved by your training pipeline
    model_obj        = load_pickle(ARTIFACT_DIR / "recommender.pkl")
    product_features = load_pickle(ARTIFACT_DIR / "product_features.pkl")
    customer_profiles = load_pickle(ARTIFACT_DIR / "customer_profiles.pkl")
    prepared_data    = load_pickle(ARTIFACT_DIR / "prepared_data.pkl")
    encoders         = load_pickle(ARTIFACT_DIR / "encoders.pkl")

    # Build the unified bundle expected by RecommenderPredictor
    bundle = {
        "ml_model": model_obj["model"],       # can be baseline or ranker
        "baseline_model": None,               # optional, leave None if using ML
        "product_features": product_features,
        "customer_profiles": customer_profiles,
        "prepared_data": prepared_data,
        "feature_names": model_obj["feature_names"],
        "cold_start_handler": None,  # OPTIONAL: add your ColdStart handler if exists
        "encoders": encoders,
    }

    # Save bundle
    out_path = ARTIFACT_DIR / "model_bundle.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model bundle created at: {out_path}")

if __name__ == "__main__":
    main()
