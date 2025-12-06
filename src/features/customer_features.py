"""
Customer Feature Engineering
============================

Builds per-customer profiles used for:
- Cold start recommendations
- Online inference feature matrices (via to_ml_dict)
"""

from abc import ABC, abstractmethod
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import datetime

from src.data_ingestion.data_preprocessor import PreparedData  # pipeline dependency

logger = logging.getLogger(__name__)


class CustomerArchetypes:
    """Customer segmentation labels used for cold start logic."""
    PARENT = "parent"
    COFFEE_PURIST = "coffee_purist"
    LATTE_LOVER = "latte_lover"
    HEALTH_CONSCIOUS = "health_conscious"
    FOOD_FOCUSED = "food_focused"
    CASUAL = "casual"

    ALL = [
        PARENT,
        COFFEE_PURIST,
        LATTE_LOVER,
        HEALTH_CONSCIOUS,
        FOOD_FOCUSED,
        CASUAL,
    ]


@dataclass
class CustomerProfile:
    customer_id: int
    archetype: str
    total_orders: int
    avg_basket_size: float
    avg_spend: float
    preferred_category: Optional[str]
    preferred_size: Optional[str]
    segment: str  # derived from first order

    def to_dict(self) -> Dict:
        """Raw profile dictionary (mainly for debugging / export)."""
        return asdict(self)

    def to_ml_dict(self) -> Dict:
        """
        ML-ready feature dictionary.

        This is used by RecommenderPredictor to construct
        per-(customer, product) feature rows at inference time.

        NOTE:
        - We **approximate** a few features used in training
        - Any missing features will be zero-filled in the predictor
        """
        features = {
            # History / volume
            "history_length": self.total_orders,
            "avg_basket_size": float(self.avg_basket_size),
            "avg_spend": float(self.avg_spend),

            # Encoded categorical fields
            "segment_encoded": hash(self.segment) % 1000 if self.segment else 0,
            "archetype_encoded": hash(self.archetype) % 1000 if self.archetype else 0,

            # Simple affinity proxies
            "category_affinity": 1.0 if self.preferred_category else 0.0,
            "is_preferred_category": 1 if self.preferred_category else 0,
            "size_affinity": 1.0 if self.preferred_size else 0.0,
            "is_preferred_size": 1 if self.preferred_size else 0,
        }

        # Time-based features (used by training_data_builder)
        now = datetime.datetime.now()
        features["hour_of_day"] = now.hour
        features["day_of_week"] = now.weekday()

        return features


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, prepared: PreparedData) -> Dict[int, CustomerProfile]:
        """Return mapping {customer_id -> CustomerProfile}."""
        raise NotImplementedError


class CustomerFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        logger.info("CustomerFeatureExtractor initialised")

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def extract_size(self, product: str) -> str:
        p = product.lower()
        if "extra large" in p:
            return "XL"
        if "large" in p:
            return "L"
        if "regular" in p:
            return "R"
        if "small" in p or "mini" in p:
            return "S"
        return "unknown"

    def classify_archetype(self, products: List[str]) -> str:
        """
        Rule-based archetype labelling from product strings.
        """
        if not products:
            return CustomerArchetypes.CASUAL

        s = " ".join(products).lower()

        if "babychino" in s or "kids" in s:
            return CustomerArchetypes.PARENT

        if "espresso" in s or "long black" in s or "macchiato" in s:
            return CustomerArchetypes.COFFEE_PURIST

        if s.count("latte") > len(products) * 0.3:
            return CustomerArchetypes.LATTE_LOVER

        if "smoothie" in s or "juice" in s or "acai" in s:
            return CustomerArchetypes.HEALTH_CONSCIOUS

        food_kw = ["breakfast", "eggs", "toast", "burger", "sandwich", "salad", "roll"]
        if any(k in s for k in food_kw):
            return CustomerArchetypes.FOOD_FOCUSED

        return CustomerArchetypes.CASUAL

    # --------------------------------------------------------------------- #
    # Core profile builder
    # --------------------------------------------------------------------- #
    def build_profile(self, customer_id: int, orders: List[dict]) -> CustomerProfile:
        all_products: List[str] = []
        all_categories: List[str] = []
        all_sizes: List[str] = []
        total_spend = 0.0

        for order in orders:
            for product in order.get("basket", []):
                if isinstance(product, str):
                    all_products.append(product)
                    all_sizes.append(self.extract_size(product))

            if "categories" in order:
                for c in order["categories"]:
                    if isinstance(c, str):
                        all_categories.append(c)

            total_spend += float(order.get("order_total", 0.0))

        total_orders = len(orders)
        avg_basket_size = len(all_products) / total_orders if total_orders else 0.0
        avg_spend = total_spend / total_orders if total_orders else 0.0

        preferred_category = (
            Counter(all_categories).most_common(1)[0][0] if all_categories else None
        )
        preferred_size = (
            Counter(all_sizes).most_common(1)[0][0] if all_sizes else None
        )

        archetype = self.classify_archetype(all_products)
        segment = orders[0].get("segment", "Regular") if orders else "Regular"

        return CustomerProfile(
            customer_id=customer_id,
            archetype=archetype,
            total_orders=total_orders,
            avg_basket_size=avg_basket_size,
            avg_spend=avg_spend,
            preferred_category=preferred_category,
            preferred_size=preferred_size,
            segment=segment,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def extract(self, prepared: PreparedData) -> Dict[int, CustomerProfile]:
        logger.info("Extracting customer profiles")

        profiles: Dict[int, CustomerProfile] = {}
        archetype_counts = Counter()

        for cid, orders in prepared.customer_histories.items():
            profile = self.build_profile(cid, orders)
            profiles[cid] = profile
            archetype_counts[profile.archetype] += 1

        logger.info("Extracted %s customer profiles", f"{len(profiles):,}")
        logger.info("Archetype distribution: %s", dict(archetype_counts))

        return profiles


if __name__ == "__main__":
    pass
