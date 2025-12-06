"""
Cold Start Handler (Refactored)
===============================
Handles recommendations for customers with no purchase history.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

from src.inference.base_predictor import BaseColdStart

logger = logging.getLogger(__name__)


@dataclass
class ColdStartRecommendation:
    product: str
    score: float
    reason: str
    confidence: str   # high / medium / low


class ColdStartHandler(BaseColdStart):
    """
    Modular cold-start handler.
    
    Strategies included:
        - Global popularity
        - Archetype preferences
        - Time-of-day adjustments
    """

    def __init__(self, product_features, prepared_data, customer_profiles=None):
        self.popularity = product_features.popularity
        self.cooccurrence = product_features.cooccurrence

        self.archetypes = self._build_archetype_profiles(
            prepared_data,
            customer_profiles or {}
        )

        logger.info(f"ColdStartHandler initialized with {len(self.archetypes)} archetypes")

    # ------------------------------------------------------------------
    # BUILD ARCHETYPE PROFILES
    # ------------------------------------------------------------------
    def _build_archetype_profiles(self, prepared_data, customer_profiles) -> Dict:
        archetype_products = defaultdict(Counter)

        for cid, orders in prepared_data.customer_histories.items():
            profile = customer_profiles.get(cid)
            archetype = profile.archetype if profile else "casual"

            for order in orders:
                for product in order["basket"]:
                    if isinstance(product, str):
                        archetype_products[archetype][product] += 1

        # Normalize to probability distributions
        profiles = {}
        for arch, counts in archetype_products.items():
            total = sum(counts.values())
            profiles[arch] = {p: c / total for p, c in counts.items()}

        return profiles

    # ------------------------------------------------------------------
    # MAIN RECOMMENDATION ENTRY POINT
    # ------------------------------------------------------------------
    def recommend(
        self,
        top_k: int = 5,
        archetype_hint: Optional[str] = None,
        time_of_day: Optional[int] = None,
        **kwargs
    ) -> List[ColdStartRecommendation]:
        """
        Generate cold-start recommendations.

        Args:
            archetype_hint: Optional customer-type (parent, tradie, etc.)
            time_of_day: Optional hour-of-day (0-23)
        """
        scores = {}

        # 1. Popularity baseline
        for product, pop in self.popularity.items():
            scores[product] = pop * 0.3

        # 2. Archetype preferences
        if archetype_hint and archetype_hint in self.archetypes:
            arch_profile = self.archetypes[archetype_hint]
            for p, pref in arch_profile.items():
                scores[p] = scores.get(p, 0) + pref * 0.5
        else:
            # fallback to "casual"
            if "casual" in self.archetypes:
                for p, pref in self.archetypes["casual"].items():
                    scores[p] = scores.get(p, 0) + pref * 0.3

        # 3. Time-of-day adjustments
        if time_of_day is not None:
            scores = self._apply_time_boost(scores, time_of_day)

        # Sort
        sorted_products = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for product, score in sorted_products:
            results.append(ColdStartRecommendation(
                product=product,
                score=round(score, 4),
                reason=(
                    f"Popular among {archetype_hint} customers"
                    if archetype_hint else
                    "General new-customer popularity"
                ),
                confidence="medium" if archetype_hint else "low"
            ))

        return results

    # ------------------------------------------------------------------
    # TIME BOOST
    # ------------------------------------------------------------------
    def _apply_time_boost(self, scores: Dict[str, float], hour: int):
        boosted = scores.copy()

        for product in scores.keys():
            p = product.lower()

            if 6 <= hour <= 11:   # morning
                if any(k in p for k in ["latte", "coffee", "cappuccino"]):
                    boosted[product] *= 1.3

            if 11 <= hour <= 14:  # lunch
                if any(k in p for k in ["burger", "sandwich", "roll"]):
                    boosted[product] *= 1.3

            if 14 <= hour <= 17:  # afternoon
                if any(k in p for k in ["iced", "cold", "frappe"]):
                    boosted[product] *= 1.2

        return boosted
