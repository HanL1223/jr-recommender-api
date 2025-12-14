"""
api.py
======
FastAPI server for JR Café Recommender.

Endpoints:
- GET /health - Health check
- GET /popular - Cold-start recommendations
- GET /recommend/{customer_id} - Personalized recommendations (legacy)
- GET /recommend/{customer_id}/split - Split favorites/discovery recommendations
- GET /customers - List available customer IDs
- GET /customer/{customer_id}/profile - Customer profile and history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from collections import Counter
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

app = FastAPI(title="JR Café Recommender API")

# CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://jr-recommender-api-frontend.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------
class RecommendationItem(BaseModel):
    product: str
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    customer_id: Optional[int]
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


class SplitRecommendationResponse(BaseModel):
    customer_id: int
    model_used: str
    favorites: List[RecommendationItem]
    discovery: List[RecommendationItem]
    addon_items: List[RecommendationItem]


class CustomerProfile(BaseModel):
    customer_id: int
    total_orders: int
    unique_products: int
    top_products: List[tuple]
    archetype: str
    segment: str


# ---------------------------------------------------------------------
# Load artifacts
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

# Check for V2 features
if "cooccur_with_history" in feature_names:
    logger.info("V2 discovery features detected ✓")
else:
    logger.warning("V2 discovery features NOT found - using legacy features")

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
# Helper Functions
# ---------------------------------------------------------------------
def get_customer_purchased_products(customer_id: int) -> set:
    """Get set of products a customer has purchased."""
    history = prepared_data.customer_histories.get(customer_id, [])
    purchased = set()
    for order in history:
        for p in order.get("basket", []):
            if isinstance(p, str):
                purchased.add(p)
    return purchased


def get_customer_product_counts(customer_id: int) -> Counter:
    """Get purchase counts for each product."""
    history = prepared_data.customer_histories.get(customer_id, [])
    counts = Counter()
    for order in history:
        for p in order.get("basket", []):
            if isinstance(p, str):
                counts[p] += 1
    return counts


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": info["model_name"],
        "version": info.get("version", "v1"),
    }


@app.get("/popular", response_model=RecommendationResponse)
def recommend_popular(top_k: int = 5):
    """
    Cold-start endpoint for new customers.
    
    GET /popular?top_k=5
    """
    logger.info(f"GET /popular called with top_k={top_k}")
    
    try:
        pred = predictor.recommend_cold_start(top_k=top_k)
        
        return RecommendationResponse(
            customer_id=pred.customer_id,
            model_used=pred.model_used,
            primary_items=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in pred.primary_items
            ],
            addon_items=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in pred.addon_items
            ],
        )
    
    except Exception as e:
        logger.error(f"Error in /popular: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: int, top_k: int = 5):
    """
    Personalized recommendations (legacy format).
    
    GET /recommend/123?top_k=5
    
    Returns combined primary_items list.
    """
    logger.info(f"GET /recommend/{customer_id} called with top_k={top_k}")
    
    try:
        pred = predictor.recommend(customer_id=customer_id, top_k=top_k)
        
        return RecommendationResponse(
            customer_id=pred.customer_id,
            model_used=pred.model_used,
            primary_items=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in pred.primary_items
            ],
            addon_items=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in pred.addon_items
            ],
        )
    
    except Exception as e:
        logger.error(f"Error in /recommend/{customer_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{customer_id}/split", response_model=SplitRecommendationResponse)
def recommend_split(
    customer_id: int,
    n_favorites: int = 2,
    n_discovery: int = 2,
    n_addons: int = 2
):
    """
    Split recommendations into favorites vs discovery.
    
    GET /recommend/123/split?n_favorites=2&n_discovery=2
    
    Returns:
    - favorites: Items they've purchased before (1-2)
    - discovery: Items they've NEVER purchased (1-2)
    - addon_items: Cross-sell items
    
    This is where the ML model adds value over baseline!
    """
    logger.info(f"GET /recommend/{customer_id}/split called")
    
    try:
        # Check if customer exists
        if customer_id not in prepared_data.customer_histories:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        # Get more recommendations than needed so we can split
        total_needed = max((n_favorites + n_discovery) * 2,30) # Get extra to ensure we have enough of each
        pred = predictor.recommend(customer_id=customer_id, top_k=max(total_needed, 10))
        
        # Get customer's purchase history
        purchased = get_customer_purchased_products(customer_id)
        
        # Split into favorites (purchased before) and discovery (never purchased)
        favorites = []
        discovery = []
        
        for item in pred.primary_items:
            if item.product in purchased:
                if len(favorites) < n_favorites:
                    favorites.append(item)
            else:
                if len(discovery) < n_discovery:
                    discovery.append(item)
            
            # Stop if we have enough of both
            if len(favorites) >= n_favorites and len(discovery) >= n_discovery:
                break
        
        # Get addon items (exclude items already in favorites/discovery)
        used_products = {i.product for i in favorites + discovery}
        addon_items = [
            i for i in pred.addon_items 
            if i.product not in used_products
        ][:n_addons]
        
        logger.info(f"Split: {len(favorites)} favorites, {len(discovery)} discovery, {len(addon_items)} addons")
        
        return SplitRecommendationResponse(
            customer_id=customer_id,
            model_used=pred.model_used,
            favorites=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in favorites
            ],
            discovery=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in discovery
            ],
            addon_items=[
                RecommendationItem(product=i.product, score=i.score, reason=i.reason)
                for i in addon_items
            ],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /recommend/{customer_id}/split: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers")
def list_customers(limit: int = 20):
    """
    List available customer IDs for testing.
    
    GET /customers?limit=20
    """
    customer_ids = list(prepared_data.customer_histories.keys())[:limit]
    
    return {
        "total_customers": len(prepared_data.customer_histories),
        "sample_ids": customer_ids,
    }


@app.get("/customer/{customer_id}/profile")
def get_customer_profile(customer_id: int):
    """
    Get customer profile and purchase history summary.
    
    GET /customer/123/profile
    """
    if customer_id not in prepared_data.customer_histories:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    history = prepared_data.customer_histories[customer_id]
    profile = customer_profiles.get(customer_id)
    
    # Count purchases
    product_counts = get_customer_product_counts(customer_id)
    
    return {
        "customer_id": customer_id,
        "total_orders": len(history),
        "unique_products": len(product_counts),
        "top_products": product_counts.most_common(10),
        "archetype": profile.archetype if profile else "unknown",
        "segment": history[0].get("segment", "Regular") if history else "unknown",
    }


@app.get("/products")
def list_products(limit: int = 50):
    """
    List available products.
    
    GET /products?limit=50
    """
    products = list(product_features.popularity.keys())[:limit]
    
    return {
        "total_products": len(product_features.popularity),
        "sample_products": products,
    }
@app.get("/recommend/{customer_id}/split")
def recommend_split(
    customer_id: int,
    n_favorites: int = 2,
    n_discovery: int = 2,
    n_addons: int = 2
):
    """
    Split recommendations into favorites vs discovery.
    """
    from collections import Counter
    
    # Check if customer exists
    if customer_id not in prepared_data.customer_histories:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    # Get recommendations
    total_needed = (n_favorites + n_discovery) * 2
    pred = predictor.recommend(customer_id=customer_id, top_k=max(total_needed, 10))
    
    # Get customer's purchase history
    history = prepared_data.customer_histories.get(customer_id, [])
    purchased = set()
    for order in history:
        for p in order.get("basket", []):
            if isinstance(p, str):
                purchased.add(p)
    
    # Split into favorites (purchased before) and discovery (never purchased)
    favorites = []
    discovery = []
    
    for item in pred.primary_items:
        if item.product in purchased:
            if len(favorites) < n_favorites:
                favorites.append(item)
        else:
            if len(discovery) < n_discovery:
                discovery.append(item)
        
        if len(favorites) >= n_favorites and len(discovery) >= n_discovery:
            break
    
    # Get addon items
    used_products = {i.product for i in favorites + discovery}
    addon_items = [i for i in pred.addon_items if i.product not in used_products][:n_addons]
    
    return {
        "customer_id": customer_id,
        "model_used": pred.model_used,
        "favorites": [{"product": i.product, "score": i.score, "reason": i.reason} for i in favorites],
        "discovery": [{"product": i.product, "score": i.score, "reason": i.reason} for i in discovery],
        "addon_items": [{"product": i.product, "score": i.score, "reason": i.reason} for i in addon_items],
    }

@app.get("/products/popular")
def get_popular_products(top_k: int = 20):
    """
    Get most popular products.
    
    GET /products/popular?top_k=20
    """
    sorted_products = sorted(
        product_features.popularity.items(),
        key=lambda x: -x[1]
    )[:top_k]
    
    return {
        "products": [
            {"product": p, "popularity": score}
            for p, score in sorted_products
        ]
    }