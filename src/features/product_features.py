"""
Product Feature Engineering
===========================

Computes product-level features:
  1. Product popularity
  2. Product–product co-occurrence (lift)
  3. Category popularity

Provides a ProductFeatures dataclass with a `to_row(product)` helper
used by the RecommenderPredictor to build ML feature matrices.
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict

from src.data_ingestion.data_preprocessor import PreparedData

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PRODUCT FEATURES CONTAINER
# -----------------------------------------------------------------------------
@dataclass
class ProductFeatures:
    """
    Container for product-level features.

    popularity:          P(product) over all orders
    cooccurrence:        nested dict of lift(product_i, product_j)
    category_popularity: P(category) over all orders

    `to_row(product)` converts these into a flat feature dict for ML models.
    """
    popularity: Dict[str, float]
    cooccurrence: Dict[str, Dict[str, float]]
    category_popularity: Dict[str, float]

    def to_dict(self) -> Dict:
        """Return all features as a nested dictionary."""
        return asdict(self)

    def to_row(self, product: str) -> Dict[str, float]:
        """
        Return a flat feature dict for a single product, used by ML models.

        This is what RecommenderPredictor expects when building its
        feature matrix:

            prod_dict = product_features.to_row(product)
        """
        # Base popularity
        pop = self.popularity.get(product, 0.0)

        # Co-occurrence: strength of best "also bought" partner
        co_dict = self.cooccurrence.get(product, {})
        top_co = max(co_dict.values()) if co_dict else 0.0

        # Category popularity – if you later map products -> category,
        # you can look up category_popularity here. For now, default to 0.
        cat_pop = 0.0

        return {
            "product_popularity": pop,
            "product_co_lift": top_co,
            "category_popularity": cat_pop,
        }


# -----------------------------------------------------------------------------
# BASE EXTRACTOR INTERFACE
# -----------------------------------------------------------------------------
class BaseProductFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, prepared: PreparedData) -> ProductFeatures:
        """Return ProductFeatures from PreparedData."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# MAIN EXTRACTOR IMPLEMENTATION
# -----------------------------------------------------------------------------
class ProductFeatureExtractor(BaseProductFeatureExtractor):
    """
    Computes:
      1. Product popularity
      2. Co-occurrence (lift)
      3. Category popularity
    """

    def __init__(self, min_support: float = 0.001):
        self.min_support = min_support
        logger.info(
            "ProductFeatureExtractor initialised (min_support=%s)",
            min_support,
        )

    # -------------------------------------------------------------------------
    # POPULARITY
    # -------------------------------------------------------------------------
    def compute_popularity(self, prepared: PreparedData) -> Dict[str, float]:
        """
        Compute product popularity as fraction of orders containing product.

        P(product) = (# orders containing product) / (total # orders)
        """
        counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                basket = [p for p in order.get("basket", []) if isinstance(p, str)]
                if not basket:
                    continue
                counts.update(set(basket))  # order-level presence
                total_orders += 1          # count this order once

        if total_orders == 0:
            return {}

        return {p: round(c / total_orders, 5) for p, c in counts.items()}

    # -------------------------------------------------------------------------
    # COOCCURRENCE (LIFT)
    # -------------------------------------------------------------------------
    def compute_cooccurrence(self, prepared: PreparedData) -> Dict[str, Dict[str, float]]:
        """
        Compute product co-occurrence (lift).

        For each pair (p1, p2):

            lift(p1, p2) = P(p1 & p2) / (P(p1) * P(p2))

        Only keep pairs where:
            - each product's frequency >= min_support
            - lift > 1  (positive association)
        """
        pair_counts = defaultdict(Counter)
        single_counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                basket = [p for p in set(order.get("basket", [])) if isinstance(p, str)]
                if not basket:
                    continue

                single_counts.update(basket)
                total_orders += 1

                for i, p1 in enumerate(basket):
                    for p2 in basket[i + 1:]:
                        pair_counts[p1][p2] += 1
                        pair_counts[p2][p1] += 1

        cooccurrence: Dict[str, Dict[str, float]] = {}
        if total_orders == 0:
            return cooccurrence

        for p1, pairs in pair_counts.items():
            p1_freq = single_counts[p1] / total_orders
            if p1_freq < self.min_support:
                continue

            for p2, count in pairs.items():
                p2_freq = single_counts[p2] / total_orders
                if p2_freq < self.min_support:
                    continue

                joint = count / total_orders
                expected_joint = p1_freq * p2_freq
                if expected_joint <= 0:
                    continue

                lift = joint / expected_joint

                if lift > 1:  # positive association only
                    cooccurrence.setdefault(p1, {})[p2] = round(lift, 3)

        return cooccurrence

    # -------------------------------------------------------------------------
    # CATEGORY POPULARITY
    # -------------------------------------------------------------------------
    def compute_category_popularity(self, prepared: PreparedData) -> Dict[str, float]:
        """
        Compute popularity of categories, if present in each order.

        P(category) = (# orders containing category) / (total # orders)
        """
        counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                cats = [c for c in order.get("categories", []) if isinstance(c, str)]
                if not cats:
                    continue
                counts.update(set(cats))
                total_orders += 1

        if total_orders == 0 or not counts:
            return {}

        return {c: cnt / total_orders for c, cnt in counts.items()}

    # -------------------------------------------------------------------------
    # MAIN ENTRYPOINT
    # -------------------------------------------------------------------------
    def extract(self, prepared: PreparedData) -> ProductFeatures:
        logger.info("Extracting product features")

        popularity = self.compute_popularity(prepared)
        cooccurrence = self.compute_cooccurrence(prepared)
        category_popularity = self.compute_category_popularity(prepared)

        # Logging top 5 products by popularity
        if popularity:
            top_5 = sorted(popularity.items(), key=lambda x: -x[1])[:5]
            logger.info("Top 5 products:")
            for rank, (prod, score) in enumerate(top_5, 1):
                logger.info("  %d. %s (%.2f%%)", rank, prod, score * 100)

        logger.info(
            "Computed %s co-occurrence pairs",
            f"{sum(len(v) for v in cooccurrence.values()):,}",
        )

        return ProductFeatures(
            popularity=popularity,
            cooccurrence=cooccurrence,
            category_popularity=category_popularity,
        )


if __name__ == "__main__":
    # Module is intended to be imported. No CLI behaviour.
    pass
