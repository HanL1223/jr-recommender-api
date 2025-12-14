"""
RecommenderPredictor (V2 Compatible)
====================================

Production-ready unified recommendation predictor.

V2 Update: Uses FeatureMatrixBuilderV2 to match TrainingDataBuilderV2 features.

Key principles:
- Uses the SAME feature engineering as TrainingDataBuilderV2 via FeatureMatrixBuilderV2.
- No ad-hoc / partial feature logic at inference.
- Ensures all training features (feature_names) are present in the matrix.
- Supports ML model or fallback baseline model.
- Supports addon recommendations using product co-occurrence.
- Supports cold-start fallback.
- Generates human-friendly recommendation reasons.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# V2: Use discovery-optimized feature builder
from src.features.feature_matrix_builder import FeatureMatrixBuilder

logger = logging.getLogger(__name__)


# OUTPUT DATA STRUCTURES
# ----------------------------------------------------------------------
@dataclass
class RecommendationItem:
    product: str
    score: float
    reason: str


@dataclass
class Prediction:
    customer_id: Optional[int]
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


# MAIN PREDICTOR
# ----------------------------------------------------------------------
class RecommenderPredictor:
    """
    Final unified predictor (V2 compatible).

    Responsibilities:
    - Construct ML feature matrix via FeatureMatrixBuilderV2
    - Score with ML or baseline model
    - Generate human-friendly recommendation reasons
    - Add-on recommendations using product co-occurrence
    - Cold-start fallback
    """

    def __init__(
        self,
        ml_model,
        baseline_model,
        product_features,
        prepared_data,
        customer_profiles,
        feature_names,
        cold_start_handler,
        encoders: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.ml_model = ml_model
        self.baseline_model = baseline_model
        self.product_features = product_features
        self.prepared_data = prepared_data
        self.customer_profiles = customer_profiles
        self.feature_names = list(feature_names) if feature_names else []
        self.cold_start_handler = cold_start_handler

        # If encoders not provided, rebuild them deterministically
        self.encoders = encoders or self._build_encoders_from_prepared(prepared_data)

        # V2: Use FeatureMatrixBuilderV2 with co-occurrence support
        self.feature_builder = FeatureMatrixBuilder(
            prepared_data=prepared_data,
            product_features=product_features,
            customer_profiles=customer_profiles,
            category_map=getattr(prepared_data, "category_map", None),
            encoders=self.encoders,
            cooccurrence=getattr(product_features, "cooccurrence", None),  # V2: Pass co-occurrence
        )

        self.use_ml = ml_model is not None
        
        # Detect feature column names for reason generation
        self._detect_feature_columns()

        logger.info("RecommenderPredictor initialized (V2) with ML=%s", bool(self.use_ml))

    def _detect_feature_columns(self):
        """Detect actual feature column names for reason generation."""
        feature_set = set(self.feature_names)
        
        # Common patterns for purchase count features
        self._purchase_count_col = None
        for pattern in ["purchase_count", "history_count", "order_count", "buy_count", 
                        "num_purchases", "customer_product_count", "hist_count", "frequency"]:
            if pattern in feature_set:
                self._purchase_count_col = pattern
                break
        
        # Common patterns for popularity features
        self._popularity_col = None
        for pattern in ["global_popularity", "popularity", "popularity_scaled",
                        "product_popularity", "pop_score", "global_pop", "item_popularity"]:
            if pattern in feature_set:
                self._popularity_col = pattern
                break
        
        # Common patterns for recency features
        self._recency_col = None
        for pattern in ["days_since_last", "recency", "days_since", "last_purchase_days",
                        "days_since_purchase", "recency_days", "orders_since_last_purchase"]:
            if pattern in feature_set:
                self._recency_col = pattern
                break
        
        # V2: Detect discovery features
        self._cooccur_col = "cooccur_with_history" if "cooccur_with_history" in feature_set else None
        self._archetype_match_col = "archetype_product_match" if "archetype_product_match" in feature_set else None
        
        logger.info(f"Detected columns - purchase: {self._purchase_count_col}, "
                    f"popularity: {self._popularity_col}, recency: {self._recency_col}, "
                    f"cooccur: {self._cooccur_col}")

    # PUBLIC API
    # ------------------------------------------------------------------
    def recommend(self, customer_id: int, top_k: int = 5) -> Prediction:
        """Generate top-K recommendations for a customer."""

        # Cold-start detection
        if customer_id not in self.prepared_data.customer_histories:
            logger.info(
                "Customer %s not found in histories. Using cold-start strategy.",
                customer_id,
            )
            return self._cold_start(top_k)

        # Get customer profile for personalized reasons
        profile = self._get_customer_profile(customer_id)
        
        # Get customer history for reason generation
        history = self.prepared_data.customer_histories.get(customer_id, [])
        purchased_products = self._extract_purchased_products(history)

        # Build feature matrix (V2 with discovery features)
        feature_df = self._build_feature_matrix(customer_id)

        # Score with appropriate model
        if self.use_ml:
            scores = self.ml_model.predict_df(feature_df)
            model_used = getattr(self.ml_model, "name", "MLModel")
        else:
            scores = self.baseline_model.predict_df(feature_df)
            model_used = getattr(self.baseline_model, "name", "BaselineModel")

        feature_df = feature_df.copy()
        feature_df["score"] = scores

        # Sort by score descending and take top_k
        top_df = feature_df.sort_values("score", ascending=False).head(top_k)

        # Format outputs with human-friendly reasons
        primary_items = [
            RecommendationItem(
                product=row["product"],
                score=float(row["score"]),
                reason=self._generate_reason(row, profile, purchased_products),
            )
            for _, row in top_df.iterrows()
        ]

        addon_items = self._get_addon_recommendations(primary_items)

        return Prediction(
            customer_id=customer_id,
            model_used=model_used,
            primary_items=primary_items,
            addon_items=addon_items,
        )

    def recommend_cold_start(self, top_k: int = 5) -> Prediction:
        """
        Public method for cold-start recommendations.
        Use this for new customers without a customer ID.
        """
        return self._cold_start(top_k)

    # CUSTOMER PROFILE & HISTORY HELPERS
    # ------------------------------------------------------------------
    def _get_customer_profile(self, customer_id: int) -> Dict[str, Any]:
        """Safely get customer profile as a dictionary."""
        profile = self.customer_profiles.get(customer_id)
        
        if profile is None:
            return {}
        
        # Handle both dict and dataclass/object profiles
        if isinstance(profile, dict):
            return profile
        
        # Convert object to dict
        if hasattr(profile, "__dict__"):
            return profile.__dict__
        
        # Try common attributes
        return {
            "segment": getattr(profile, "segment", None),
            "archetype": getattr(profile, "archetype", None),
            "total_orders": getattr(profile, "total_orders", 0),
            "total_spend": getattr(profile, "total_spend", 0),
        }

    def _extract_purchased_products(self, history: List) -> Dict[str, int]:
        """Extract product purchase counts from customer history."""
        counts = {}
        for item in history:
            # Handle different history formats
            if isinstance(item, dict):
                # Check for basket format (list of products per order)
                basket = item.get("basket", [])
                if basket:
                    for product in basket:
                        if isinstance(product, str):
                            counts[product] = counts.get(product, 0) + 1
                else:
                    # Single product format
                    product = item.get("product") or item.get("product_name") or item.get("item")
                    if product:
                        counts[product] = counts.get(product, 0) + 1
            elif hasattr(item, "product"):
                product = item.product
                if product:
                    counts[product] = counts.get(product, 0) + 1
        
        return counts

    # REASON GENERATION (V2 - with discovery reasons)
    # ------------------------------------------------------------------
    def _generate_reason(
        self, 
        row: pd.Series, 
        profile: Dict[str, Any],
        purchased_products: Dict[str, int]
    ) -> str:
        """
        Generate a human-friendly reason for why this item is recommended.
        V2: Includes discovery-specific reasons.
        """
        product = row.get("product", "")
        
        # 1. Check direct purchase history (most reliable)
        if product in purchased_products:
            count = purchased_products[product]
            if count >= 5:
                return "One of your favorites"
            elif count >= 3:
                return "You order this often"
            elif count >= 2:
                return "You've enjoyed this before"
            else:
                return "Based on your order history"
        
        # 2. Check purchase count feature if available
        if self._purchase_count_col:
            purchase_count = self._safe_get_numeric(row, self._purchase_count_col)
            if purchase_count and purchase_count > 0:
                if purchase_count >= 5:
                    return "One of your favorites"
                elif purchase_count >= 2:
                    return "You've enjoyed this before"
                else:
                    return "Based on your order history"
        
        # === V2: DISCOVERY REASONS ===
        
        # 3. Check if it's a variant of something they buy
        has_variant = self._safe_get_numeric(row, "has_bought_variant")
        if has_variant and has_variant > 0:
            base = product.split(" (")[0] if " (" in product else product
            return f"A new take on {base}"
        
        # 4. Check co-occurrence with history (V2 feature)
        if self._cooccur_col:
            cooccur = self._safe_get_numeric(row, self._cooccur_col)
            if cooccur and cooccur > 0.3:
                return "Pairs with your favorites"
        
        # 5. Check archetype-product match (V2 feature)
        if self._archetype_match_col:
            archetype_match = self._safe_get_numeric(row, self._archetype_match_col)
            if archetype_match and archetype_match > 0:
                return "Matches your taste"
        
        # 6. Check archetype-category match (fallback)
        archetype = profile.get("archetype", "")
        if archetype:
            category = self._get_product_category(product)
            if self._check_archetype_category_match(archetype, category, product):
                return "Matches your taste"
        
        # 7. Check preferred category
        is_pref_cat = self._safe_get_numeric(row, "is_preferred_category")
        if is_pref_cat and is_pref_cat > 0:
            return "From your favorite category"
        
        # 8. Check popularity
        if self._popularity_col:
            popularity = self._safe_get_numeric(row, self._popularity_col)
            if popularity and popularity > 0.05:
                return "Customer favorite"
        
        # 9. Check recency
        if self._recency_col:
            days_since = self._safe_get_numeric(row, self._recency_col)
            if days_since and 0 < days_since < 14:
                return "Recently caught your eye"
        
        # 10. Segment-based fallback
        segment = profile.get("segment", "")
        if segment == "VIP":
            return "Curated for you"
        elif segment == "Regular":
            return "Popular with regulars"
        elif segment == "New":
            return "Great choice to try"
        
        # 11. Score-based fallback
        score = row.get("score", 0)
        if score > 0.8:
            return "Top pick for you"
        elif score > 0.5:
            return "Recommended for you"
        
        return "You might like this"

    def _safe_get_numeric(self, row: pd.Series, col: str) -> Optional[float]:
        """Safely get a numeric value from a row."""
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _get_product_category(self, product: str) -> str:
        """Get category for a product."""
        category_map = getattr(self.product_features, "category_map", {})
        if not category_map:
            category_map = getattr(self.prepared_data, "category_map", {})
        return category_map.get(product, "")

    def _check_archetype_category_match(self, archetype: str, category: str, product: str) -> bool:
        """Check if archetype matches the product category or product name."""
        archetype_keywords = {
            "coffee_purist": ["coffee", "espresso", "black", "long black", "americano", "ristretto"],
            "latte_lover": ["latte", "cappuccino", "flat white", "mocha", "milk"],
            "health_conscious": ["healthy", "salad", "smoothie", "juice", "fresh", "acai", "green"],
            "food_focused": ["food", "meal", "sandwich", "toastie", "breakfast", "lunch", "bacon", "egg"],
            "parent": ["kids", "family", "baby", "babyccino"],
        }
        
        keywords = archetype_keywords.get(archetype, [])
        if not keywords:
            return False
        
        # Check both category and product name
        search_text = f"{category} {product}".lower()
        return any(kw in search_text for kw in keywords)

    # FEATURE MATRIX (using FeatureMatrixBuilderV2)
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, customer_id: int) -> pd.DataFrame:
        """
        Build ML-ready feature matrix for a given customer.
        V2: Includes discovery features (co-occurrence, archetype match, etc.)
        """
        df = self.feature_builder.build(customer_id)

        # Ensure all training features exist as columns
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        return df

    # ENCODERS
    # ------------------------------------------------------------------
    def _build_encoders_from_prepared(self, prepared_data) -> Dict[str, Dict[str, int]]:
        """Rebuild category/segment/archetype encoders."""
        category_map = getattr(prepared_data, "category_map", {}) or {}
        categories = sorted(set(category_map.values()))
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        segment_to_idx = {"New": 1, "Regular": 2, "VIP": 3}
        archetype_to_idx = {
            "casual": 0,
            "parent": 1,
            "coffee_purist": 2,
            "latte_lover": 3,
            "health_conscious": 4,
            "food_focused": 5,
        }

        return {
            "category": cat_to_idx,
            "segment": segment_to_idx,
            "archetype": archetype_to_idx,
        }

    # ADD-ON RECOMMENDATIONS
    # ------------------------------------------------------------------
    def _get_addon_recommendations(
        self, primary_items: List[RecommendationItem], top_k: int = 2
    ) -> List[RecommendationItem]:
        """Generate add-on recommendations based on co-occurrence."""
        co = getattr(self.product_features, "cooccurrence", {})
        scores: Dict[str, float] = {}

        primary_set = {item.product for item in primary_items}

        for item in primary_items:
            p = item.product
            if p not in co:
                continue

            for other, lift in co[p].items():
                if other in primary_set:
                    continue
                scores[other] = max(scores.get(other, 0.0), float(lift))

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        return [
            RecommendationItem(
                product=prod,
                score=lift,
                reason="Often paired together",
            )
            for prod, lift in sorted_items
        ]

    # COLD START
    # ------------------------------------------------------------------
    def _cold_start(self, top_k: int) -> Prediction:
        """Generate cold-start recommendations for unknown customers."""
        cold_items = self.cold_start_handler.recommend(top_k=top_k)

        primary_items = [
            RecommendationItem(
                product=i.product,
                score=float(i.score),
                reason=i.reason,
            )
            for i in cold_items
        ]

        return Prediction(
            customer_id=None,
            model_used="ColdStart",
            primary_items=primary_items,
            addon_items=[],
        )