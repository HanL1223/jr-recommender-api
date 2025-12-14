"""
Training Data Builder V2 - Discovery Optimized
==============================================

Key improvements over V1:
1. CO-OCCURRENCE FEATURES - Signal for items they've never bought
2. FIRST-PURCHASE TRACKING - Learn patterns of trying new things
3. ARCHETYPE-PRODUCT MATCH - Does this fit their customer type?
4. BETTER NEGATIVE SAMPLING - Stratified by relevance

The goal: Give the model signal to predict DISCOVERY, not just reorders.
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    samples_df: pd.DataFrame
    feature_names: List[str]
    n_positive: int
    n_negative: int
    product_to_idx: Dict[str, int]
    idx_to_product: Dict[int, str]
    encoders: Dict[str, Dict[str, int]]


class TrainingDataBuilder:
    """
    Discovery-optimized training data builder.
    
    Key insight: To predict discovery, we need features that have signal
    even when history_count = 0.
    
    New features:
    - cooccur_with_history: Does this product co-occur with their past purchases?
    - archetype_product_match: Does this fit their customer archetype?
    - is_first_purchase: Label indicating this is a NEW item for them
    - category_exploration_rate: How often do they try new categories?
    - similar_to_favorites: Cosine similarity to their top items
    """

    FEATURE_NAMES = [
        # === HISTORY FEATURES (existing) ===
        "in_history", "history_count", "log_history_count", "history_freq",
        "frequency_rank_pct", "recent_purchase_flag",
        "orders_since_last_purchase", "time_decay_score", "never_purchased",
        
        # === CATEGORY/SIZE AFFINITY (existing) ===
        "category_affinity", "is_preferred_category",
        "size_affinity", "is_preferred_size",
        "has_bought_variant",
        
        # === NEW: DISCOVERY FEATURES ===
        "cooccur_with_history",      # Co-occurrence score with past purchases
        "cooccur_max",               # Max co-occurrence with any single past item
        "archetype_product_match",   # Does product match customer archetype?
        "category_exploration_rate", # How often they try new categories
        "product_exploration_rate",  # How often they try new products
        "is_first_purchase",         # 1 if this is first time buying (for analysis)
        
        # === TIME FEATURES (existing) ===
        "hour_of_day", "day_of_week", "days_since_last_order",
        
        # === CUSTOMER FEATURES (existing) ===
        "history_length", "avg_basket_size", "avg_spend",
        "segment_encoded", "archetype_encoded",
        
        # === PRODUCT FEATURES (existing) ===
        "popularity_scaled", "adjusted_popularity", "category_encoded",
    ]

    # Archetype -> product keyword mapping
    ARCHETYPE_PRODUCTS = {
        "coffee_purist": ["espresso", "long black", "americano", "ristretto", "black coffee", "pourover"],
        "latte_lover": ["latte", "cappuccino", "flat white", "mocha", "chai", "hot chocolate"],
        "health_conscious": ["smoothie", "juice", "acai", "salad", "healthy", "green", "protein"],
        "food_focused": ["sandwich", "toastie", "bacon", "egg", "burger", "wrap", "pie", "sausage"],
        "parent": ["kids", "babyccino", "small", "mini", "fluffy"],
        "casual": [],  # No specific preferences
    }

    def __init__(
        self, 
        negative_ratio: int = 5, 
        random_seed: int = 42,
        cooccurrence_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.negative_ratio = negative_ratio
        self.cooccurrence = cooccurrence_matrix or {}
        np.random.seed(random_seed)
        logger.info("TrainingDataBuilderV2 initialized (discovery-optimized)")

    def set_cooccurrence(self, cooccurrence_matrix: Dict[str, Dict[str, float]]):
        """Set co-occurrence matrix from ProductFeatureExtractor."""
        self.cooccurrence = cooccurrence_matrix

    def build(self, prepared_data, product_features, customer_profiles) -> TrainingData:
        """Build training samples with discovery features."""
        logger.info("Building training samples (V2 - discovery optimized)...")

        # Get co-occurrence from product features if not set
        if not self.cooccurrence and hasattr(product_features, 'cooccurrence'):
            self.cooccurrence = product_features.cooccurrence or {}
            logger.info(f"Loaded co-occurrence matrix: {len(self.cooccurrence)} products")

        samples = []
        product_set = set(prepared_data.product_list)
        category_map = prepared_data.category_map or {}
        popularity = product_features.popularity or {}

        # Build encoders
        categories = sorted(set(category_map.values()))
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        segment_to_idx = {"New": 1, "Regular": 2, "VIP": 3}
        archetype_to_idx = {
            "casual": 0, "parent": 1, "coffee_purist": 2,
            "latte_lover": 3, "health_conscious": 4, "food_focused": 5,
        }

        encoders = {
            "category": cat_to_idx,
            "segment": segment_to_idx,
            "archetype": archetype_to_idx,
        }

        # Track stats
        n_first_purchases = 0
        n_reorders = 0

        for idx, (cid, orders) in enumerate(prepared_data.customer_histories.items()):
            if len(orders) < 2:
                continue

            if idx % 200 == 0:
                logger.info(f"Processing {idx}/{len(prepared_data.customer_histories)} customers")

            profile = customer_profiles.get(cid)
            archetype = profile.archetype if profile else "casual"
            archetype_encoded = archetype_to_idx.get(archetype, 0)
            segment = orders[0].get("segment", "Regular")
            segment_encoded = segment_to_idx.get(segment, 2)

            customer_samples, first, reorder = self._build_customer_samples(
                cid, orders, product_set, popularity,
                category_map, cat_to_idx,
                segment_encoded, archetype_encoded, archetype
            )
            
            samples.extend(customer_samples)
            n_first_purchases += first
            n_reorders += reorder

        if not samples:
            raise ValueError("No training samples generated!")

        df = pd.DataFrame(samples).replace([np.inf, -np.inf], 0).fillna(0)
        df["order_date"] = pd.to_datetime(df["order_date"])

        n_pos = int(df["label"].sum())
        n_neg = len(df) - n_pos

        logger.info(f"Built {len(df)} samples (pos={n_pos}, neg={n_neg})")
        logger.info(f"  First purchases: {n_first_purchases} ({100*n_first_purchases/n_pos:.1f}% of positives)")
        logger.info(f"  Reorders: {n_reorders} ({100*n_reorders/n_pos:.1f}% of positives)")

        product_to_idx = {p: i for i, p in enumerate(prepared_data.product_list)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}

        return TrainingData(
            samples_df=df,
            feature_names=self.FEATURE_NAMES,
            n_positive=n_pos,
            n_negative=n_neg,
            product_to_idx=product_to_idx,
            idx_to_product=idx_to_product,
            encoders=encoders,
        )

    def _build_customer_samples(
        self, cid, orders, product_set, popularity,
        category_map, cat_to_idx, segment_encoded, archetype_encoded, archetype
    ):
        """Build samples for one customer."""
        samples = []
        n_first = 0
        n_reorder = 0
        
        # Track exploration rates across all orders
        all_past_products = set()
        all_past_categories = set()
        new_product_count = 0
        new_category_count = 0
        total_purchases = 0

        for order_idx in range(1, len(orders)):
            past = orders[:order_idx]
            curr = orders[order_idx]

            basket = {p for p in curr["basket"] if isinstance(p, str)}
            if not basket:
                continue

            # Compute features from past orders
            past_feats = self._compute_past_features(past, category_map)
            time_feats = self._compute_time_features(curr, past)
            
            # Compute exploration rates
            for p in basket:
                total_purchases += 1
                if p not in all_past_products:
                    new_product_count += 1
                if category_map.get(p, "Unknown") not in all_past_categories:
                    new_category_count += 1
                    
            product_exploration_rate = new_product_count / max(total_purchases, 1)
            category_exploration_rate = new_category_count / max(total_purchases, 1)

            cust_feats = {
                "history_length": order_idx,
                "avg_basket_size": past_feats["avg_basket_size"],
                "avg_spend": past_feats["avg_spend"],
                "segment_encoded": segment_encoded,
                "archetype_encoded": archetype_encoded,
                "category_exploration_rate": category_exploration_rate,
                "product_exploration_rate": product_exploration_rate,
            }

            # === POSITIVE SAMPLES ===
            for p in basket:
                is_first = p not in past_feats["past_products"]
                if is_first:
                    n_first += 1
                else:
                    n_reorder += 1

                samples.append(
                    self._create_sample(
                        cid, p, label=1, order_idx=order_idx,
                        order_date=curr["order_date"],
                        past_feats=past_feats, time_feats=time_feats,
                        cust_feats=cust_feats,
                        popularity=popularity, category_map=category_map,
                        cat_to_idx=cat_to_idx, archetype=archetype,
                        is_first_purchase=is_first
                    )
                )

            # === NEGATIVE SAMPLES (stratified) ===
            neg_pool = list(product_set - basket)
            n_neg = min(len(basket) * self.negative_ratio, len(neg_pool))

            if n_neg > 0:
                negs = self._stratified_negative_sample(
                    neg_pool, n_neg, past_feats, category_map, popularity
                )

                for p in negs:
                    samples.append(
                        self._create_sample(
                            cid, p, label=0, order_idx=order_idx,
                            order_date=curr["order_date"],
                            past_feats=past_feats, time_feats=time_feats,
                            cust_feats=cust_feats,
                            popularity=popularity, category_map=category_map,
                            cat_to_idx=cat_to_idx, archetype=archetype,
                            is_first_purchase=False
                        )
                    )

            # Update running totals
            all_past_products.update(basket)
            for p in basket:
                all_past_categories.add(category_map.get(p, "Unknown"))

        return samples, n_first, n_reorder

    def _stratified_negative_sample(
        self, neg_pool, n_neg, past_feats, category_map, popularity
    ):
        """
        Stratified negative sampling:
        - 40% from preferred category (hard negatives - similar but not bought)
        - 30% from other categories they've tried
        - 30% random (popularity weighted)
        
        This helps the model learn to distinguish within categories.
        """
        pref_cat = past_feats["preferred_category"]
        past_cats = set(past_feats["past_categories"].keys())
        
        # Split pool by category
        same_cat = [p for p in neg_pool if category_map.get(p) == pref_cat]
        other_tried = [p for p in neg_pool if category_map.get(p) in past_cats and category_map.get(p) != pref_cat]
        other = [p for p in neg_pool if category_map.get(p) not in past_cats]
        
        # Allocate samples
        n_same = min(int(n_neg * 0.4), len(same_cat))
        n_other_tried = min(int(n_neg * 0.3), len(other_tried))
        n_random = n_neg - n_same - n_other_tried
        
        selected = []
        
        if n_same > 0 and same_cat:
            selected.extend(np.random.choice(same_cat, size=n_same, replace=False))
        
        if n_other_tried > 0 and other_tried:
            selected.extend(np.random.choice(other_tried, size=min(n_other_tried, len(other_tried)), replace=False))
        
        # Fill rest with popularity-weighted random
        remaining_pool = [p for p in neg_pool if p not in selected]
        if n_random > 0 and remaining_pool:
            weights = np.array([popularity.get(p, 0.0001) for p in remaining_pool])
            weights /= weights.sum()
            n_to_sample = min(n_random, len(remaining_pool))
            selected.extend(np.random.choice(remaining_pool, size=n_to_sample, replace=False, p=weights))
        
        return selected

    def _create_sample(
        self, cid, product, label, order_idx, order_date,
        past_feats, time_feats, cust_feats,
        popularity, category_map, cat_to_idx, archetype,
        is_first_purchase
    ):
        """Create a single training sample with discovery features."""
        
        # === HISTORY FEATURES ===
        hist_count = past_feats["past_product_counts"].get(product, 0)
        history_freq = hist_count / max(past_feats["n_orders"], 1)
        log_history = np.log1p(hist_count)
        in_hist = int(product in past_feats["past_products"])

        last_idx = past_feats["past_product_last_idx"].get(product, -1)
        orders_since = order_idx - last_idx - 1 if last_idx >= 0 else 999
        recent_flag = int(last_idx >= max(0, order_idx - 3))
        time_decay = np.exp(-0.3 * orders_since) if in_hist else 0.0
        freq_pct = hist_count / max(past_feats["max_product_count"], 1)

        # === CATEGORY/SIZE FEATURES ===
        category = category_map.get(product, "Unknown")
        cat_aff = past_feats["past_categories"].get(category, 0) / max(past_feats["total_items"], 1)
        is_pref_cat = int(category == past_feats["preferred_category"])

        size = self._extract_size(product)
        size_aff = past_feats["past_sizes"].get(size, 0) / max(past_feats["total_items"], 1)
        is_pref_size = int(size == past_feats["preferred_size"])

        base = self._get_base_product(product)
        has_variant = int(past_feats["past_base_products"].get(base, 0) > 0)

        # === NEW: CO-OCCURRENCE FEATURES ===
        cooccur_score, cooccur_max = self._compute_cooccurrence_score(
            product, past_feats["past_products"]
        )

        # === NEW: ARCHETYPE-PRODUCT MATCH ===
        archetype_match = self._compute_archetype_match(archetype, product)

        # === POPULARITY ===
        pop_scaled = popularity.get(product, 0) * 0.1
        adjusted_pop = pop_scaled * (0.5 if orders_since > 3 and not in_hist else 1)

        return {
            "customer_id": cid,
            "product": product,
            "label": label,
            "order_idx": order_idx,
            "order_date": order_date,

            # History
            "in_history": in_hist,
            "history_count": hist_count,
            "log_history_count": log_history,
            "history_freq": history_freq,
            "frequency_rank_pct": freq_pct,
            "recent_purchase_flag": recent_flag,
            "orders_since_last_purchase": min(orders_since, 100),
            "time_decay_score": time_decay,
            "never_purchased": int(not in_hist),

            # Category/Size
            "category_affinity": cat_aff,
            "is_preferred_category": is_pref_cat,
            "size_affinity": size_aff,
            "is_preferred_size": is_pref_size,
            "has_bought_variant": has_variant,

            # NEW: Discovery features
            "cooccur_with_history": cooccur_score,
            "cooccur_max": cooccur_max,
            "archetype_product_match": archetype_match,
            "category_exploration_rate": cust_feats["category_exploration_rate"],
            "product_exploration_rate": cust_feats["product_exploration_rate"],
            "is_first_purchase": int(is_first_purchase),

            # Time
            **time_feats,
            
            # Customer
            "history_length": cust_feats["history_length"],
            "avg_basket_size": cust_feats["avg_basket_size"],
            "avg_spend": cust_feats["avg_spend"],
            "segment_encoded": cust_feats["segment_encoded"],
            "archetype_encoded": cust_feats["archetype_encoded"],

            # Product
            "popularity_scaled": pop_scaled,
            "adjusted_popularity": adjusted_pop,
            "category_encoded": cat_to_idx.get(category, 0),
        }

    def _compute_cooccurrence_score(
        self, product: str, past_products: Set[str]
    ) -> tuple:
        """
        Compute co-occurrence score between product and customer's history.
        
        Returns:
        - cooccur_score: Average co-occurrence with all past products
        - cooccur_max: Max co-occurrence with any single past product
        
        This gives signal for items they've NEVER bought but co-occur
        with items they HAVE bought!
        """
        if not self.cooccurrence or not past_products:
            return 0.0, 0.0

        scores = []
        for past_p in past_products:
            if past_p in self.cooccurrence:
                score = self.cooccurrence[past_p].get(product, 0.0)
                scores.append(score)
            # Also check reverse direction
            if product in self.cooccurrence:
                score = self.cooccurrence[product].get(past_p, 0.0)
                scores.append(score)

        if not scores:
            return 0.0, 0.0

        return np.mean(scores), max(scores)

    def _compute_archetype_match(self, archetype: str, product: str) -> float:
        """
        Compute how well product matches customer archetype.
        
        Returns 1.0 for strong match, 0.5 for partial, 0.0 for no match.
        """
        keywords = self.ARCHETYPE_PRODUCTS.get(archetype, [])
        if not keywords:
            return 0.0

        product_lower = product.lower()
        
        # Strong match: keyword in product name
        for kw in keywords:
            if kw in product_lower:
                return 1.0
        
        return 0.0

    def _compute_past_features(self, past_orders, category_map):
        """Compute features from past orders."""
        past_products = set()
        product_counts = Counter()
        last_idx = {}
        cat_counts = Counter()
        size_counts = Counter()
        base_counts = Counter()

        total_items = 0
        total_spend = 0
        basket_sizes = []

        for i, order in enumerate(past_orders):
            for p in order["basket"]:
                if not isinstance(p, str):
                    continue

                past_products.add(p)
                product_counts[p] += 1
                last_idx[p] = i

                cat = category_map.get(p, "Unknown")
                cat_counts[cat] += 1

                size = self._extract_size(p)
                size_counts[size] += 1

                base = self._get_base_product(p)
                base_counts[base] += 1

                total_items += 1

            total_spend += order.get("order_total", 0)
            basket_sizes.append(order.get("basket_size", len(order["basket"])))

        max_count = max(product_counts.values()) if product_counts else 1

        return {
            "past_products": past_products,
            "past_product_counts": product_counts,
            "past_product_last_idx": last_idx,
            "past_categories": cat_counts,
            "past_sizes": size_counts,
            "past_base_products": base_counts,
            "total_items": total_items,
            "n_orders": len(past_orders),
            "max_product_count": max_count,
            "avg_basket_size": np.mean(basket_sizes) if basket_sizes else 0,
            "avg_spend": total_spend / len(past_orders) if past_orders else 0,
            "preferred_category": cat_counts.most_common(1)[0][0] if cat_counts else "Unknown",
            "preferred_size": size_counts.most_common(1)[0][0] if size_counts else "unknown",
        }

    def _compute_time_features(self, curr, past):
        """Compute time-based features."""
        hour = int(curr.get("order_time", "12:00:00").split(":")[0])
        dow = curr["order_date"].dayofweek

        if past:
            last_date = past[-1]["order_date"]
            days = (curr["order_date"] - last_date).days
            days = min(max(days, 0), 365)
        else:
            days = 0

        return {
            "hour_of_day": hour,
            "day_of_week": dow,
            "days_since_last_order": days,
        }

    def _extract_size(self, p):
        p = p.lower()
        if "extra large" in p: return "XL"
        if "large" in p: return "L"
        if "regular" in p: return "R"
        if "small" in p or "mini" in p: return "S"
        return "unknown"

    def _get_base_product(self, product):
        return product.split(" (")[0] if " (" in product else product