"""
api.py
======
FastAPI server for JR Café Recommender.
Loads artifacts the same way as demo.py for consistency.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pickle
import json
import logging
import traceback

from src.inference.recommender_predictor import RecommenderPredictor
from src.inference.cold_start import ColdStartHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cafe Recommender API")

# CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Load artifacts (same as demo.py)
# ---------------------------------------------------------------------
ARTIFACT_DIR = Path("models/artifacts")

logger.info("Loading artifacts...")

# Load model bundle (contains model + feature_names)
with open(ARTIFACT_DIR / "recommender.pkl", "rb") as f:
    bundle = pickle.load(f)
ml_model = bundle["model"]
feature_names = bundle["feature_names"]

# Load other artifacts
with open(ARTIFACT_DIR / "product_features.pkl", "rb") as f:
    product_features = pickle.load(f)

with open(ARTIFACT_DIR / "customer_profiles.pkl", "rb") as f:
    customer_profiles = pickle.load(f)

with open(ARTIFACT_DIR / "prepared_data.pkl", "rb") as f:
    prepared_data = pickle.load(f)

# Load model info for logging
with open(ARTIFACT_DIR / "model_info.json", "r") as f:
    info = json.load(f)

logger.info("Loaded model: %s", info["model_name"])
logger.info("VALID NDCG@3: %.4f", info["valid_metrics"]["ndcg@3"])
logger.info("TEST  NDCG@3: %.4f", info["test_metrics"]["ndcg@3"])

# Create cold start handler
cold_handler = ColdStartHandler(
    product_features=product_features,
    prepared_data=prepared_data,
    customer_profiles=customer_profiles,
)

# Create predictor
predictor = RecommenderPredictor(
    ml_model=ml_model,
    baseline_model=None,
    product_features=product_features,
    prepared_data=prepared_data,
    customer_profiles=customer_profiles,
    feature_names=feature_names,
    cold_start_handler=cold_handler,
)

logger.info("Predictor ready!")


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/popular")
def recommend_popular(top_k: int = 5):
    """
    Cold-start endpoint for new customers.
    GET /popular?top_k=5
    """
    logger.info(f"GET /popular called with top_k={top_k}")
    
    try:
        pred = predictor.recommend_cold_start(top_k=top_k)
        logger.info(f"Returning {len(pred.primary_items)} items")
        return pred.__dict__
    
    except Exception as e:
        logger.error(f"Error in /popular: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, top_k: int = 5):
    """
    Personalized recommendations.
    GET /recommend/123?top_k=5
    """
    logger.info(f"GET /recommend/{customer_id} called with top_k={top_k}")
    
    try:
        pred = predictor.recommend(customer_id=customer_id, top_k=top_k)
        logger.info(f"Returning {len(pred.primary_items)} items for customer {customer_id}")
        return pred.__dict__
    
    except Exception as e:
        logger.error(f"Error in /recommend/{customer_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))