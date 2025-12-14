"""
FeatureMatrixBuilder V2 - Discovery Optimized
=============================================

Builds per-customer ML feature matrix that EXACTLY matches
TrainingDataBuilderV2 features, including discovery features.

New features aligned with training:
- cooccur_with_history
- cooccur_max  
- archetype_product_match
- category_exploration_rate
- product_exploration_rate
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Set, Optional


class FeatureMatrixBuilder:
    """
    Inference-time feature builder aligned with TrainingDataBuilderV2.
    
    Includes discovery features:
    - Co-occurrence with history
    - Archetype-product matching
    - Exploration rates
    """
    
    ARCHETYPE_PRODUCTS = {
        "coffee_purist": ["espresso", "long black", "americano", "ristretto", "black coffee", "pourover"],
        "latte_lover": ["latte", "cappuccino", "flat white", "mocha", "chai", "hot chocolate"],
        "health_conscious": ["smoothie", "juice", "acai", "salad", "healthy", "green", "protein"],
        "food_focused": ["sandwich", "toastie", "bacon", "egg", "burger", "wrap", "pie", "sausage"],
        "parent": ["kids", "babyccino", "small", "mini", "fluffy"],
        "casual": [],
    }

    def __init__(
        self, 
        prepared_data, 
        product_features, 
        customer_profiles, 
        category_map, 
        encoders,
        cooccurrence: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.prepared = prepared_data
        self.product_features = product_features
        self.customer_profiles = customer_profiles
        self.category_map = category_map or {}
        self.encoders = encoders
        
        # Get co-occurrence from product_features if not provided
        self.cooccurrence = cooccurrence
        if self.cooccurrence is None and hasattr(product_features, 'cooccurrence'):
            self.cooccurrence = product_features.cooccurrence or {}

    def extract_size(self, product):
        p = product.lower()
        if "extra large" in p: return "XL"
        if "large" in p: return "L"
        if "regular" in p: return "R"
        if "small" in p or "mini" in p: return "S"
        return "unknown"

    def get_base_product(self, product):
        return product.split(" (")[0] if " (" in product else product

    def _customer_features(self, customer_id, past_orders, order_idx):
        """Compute customer-level features."""
        profile = self.customer_profiles.get(customer_id)

        hist_len = order_idx
        avg_basket_size = (
            np.mean([o.get("basket_size", len(o["basket"])) for o in past_orders])
            if past_orders else 0
        )
        avg_spend = (
            np.mean([o.get("order_total", 0) for o in past_orders])
            if past_orders else 0
        )

        segment = past_orders[0].get("segment", "Regular") if past_orders else "Regular"
        segment_encoded = self.encoders["segment"].get(segment, 2)

        archetype = profile.archetype if profile else "casual"
        archetype_encoded = self.encoders["archetype"].get(archetype, 0)

        return {
            "history_length": hist_len,
            "avg_basket_size": avg_basket_size,
            "avg_spend": avg_spend,
            "segment_encoded": segment_encoded,
            "archetype_encoded": archetype_encoded,
            "archetype": archetype,  # Keep string for matching
        }

    def compute_past_features(self, past_orders):
        """Compute features from purchase history."""
        past_products = set()
        product_counts = Counter()
        last_idx = {}
        past_categories = Counter()
        past_sizes = Counter()
        past_base = Counter()

        total_items = 0

        for i, order in enumerate(past_orders):
            for p in order["basket"]:
                if not isinstance(p, str):
                    continue

                past_products.add(p)
                product_counts[p] += 1
                last_idx[p] = i

                cat = self.category_map.get(p, "Unknown")
                past_categories[cat] += 1

                size = self.extract_size(p)
                past_sizes[size] += 1

                base = self.get_base_product(p)
                past_base[base] += 1

                total_items += 1

        preferred_category = past_categories.most_common(1)[0][0] if past_categories else "Unknown"
        preferred_size = past_sizes.most_common(1)[0][0] if past_sizes else "unknown"
        max_count = max(product_counts.values()) if product_counts else 1

        return {
            "past_products": past_products,
            "past_product_counts": product_counts,
            "past_product_last_idx": last_idx,
            "past_categories": past_categories,
            "past_sizes": past_sizes,
            "past_base_products": past_base,
            "total_items": total_items,
            "n_orders": len(past_orders),
            "max_product_count": max_count,
            "preferred_category": preferred_category,
            "preferred_size": preferred_size,
        }

    def compute_exploration_rates(self, past_orders):
        """Compute how exploratory this customer is."""
        if not past_orders:
            return 0.0, 0.0
            
        seen_products = set()
        seen_categories = set()
        new_product_count = 0
        new_category_count = 0
        total_purchases = 0
        
        for order in past_orders:
            for p in order["basket"]:
                if not isinstance(p, str):
                    continue
                    
                total_purchases += 1
                if p not in seen_products:
                    new_product_count += 1
                    seen_products.add(p)
                    
                cat = self.category_map.get(p, "Unknown")
                if cat not in seen_categories:
                    new_category_count += 1
                    seen_categories.add(cat)
        
        product_exploration = new_product_count / max(total_purchases, 1)
        category_exploration = new_category_count / max(total_purchases, 1)
        
        return product_exploration, category_exploration

    def compute_cooccurrence_score(self, product: str, past_products: Set[str]) -> tuple:
        """
        Compute co-occurrence between product and customer's history.
        
        This is the KEY discovery feature - gives signal for items
        they've never bought but co-occur with their purchases.
        """
        if not self.cooccurrence or not past_products:
            return 0.0, 0.0

        scores = []
        for past_p in past_products:
            if past_p in self.cooccurrence:
                score = self.cooccurrence[past_p].get(product, 0.0)
                if score > 0:
                    scores.append(score)
            if product in self.cooccurrence:
                score = self.cooccurrence[product].get(past_p, 0.0)
                if score > 0:
                    scores.append(score)

        if not scores:
            return 0.0, 0.0

        return np.mean(scores), max(scores)

    def compute_archetype_match(self, archetype: str, product: str) -> float:
        """Check if product matches customer archetype."""
        keywords = self.ARCHETYPE_PRODUCTS.get(archetype, [])
        if not keywords:
            return 0.0

        product_lower = product.lower()
        for kw in keywords:
            if kw in product_lower:
                return 1.0
        return 0.0

    def compute_time_features(self, past_orders, now_order):
        """Compute time-based features."""
        if not past_orders:
            last_date = now_order["order_date"]
            days_since = 0
        else:
            last_date = past_orders[-1]["order_date"]
            days_since = min((now_order["order_date"] - last_date).days, 365)

        hour = int(now_order.get("order_time", "12:00:00").split(":")[0])

        return {
            "hour_of_day": hour,
            "day_of_week": now_order["order_date"].dayofweek,
            "days_since_last_order": max(days_since, 0),
        }

    def build(self, customer_id):
        """
        Build feature matrix for customer - all products scored.
        
        Aligned with TrainingDataBuilderV2 features.
        """
        history = self.prepared.customer_histories.get(customer_id, [])
        order_idx = len(history)
        past_orders = history

        pf = self.compute_past_features(past_orders)
        product_exp, category_exp = self.compute_exploration_rates(past_orders)

        now_order = {
            "order_date": past_orders[-1]["order_date"] if past_orders else pd.Timestamp("2024-01-01"),
            "order_time": past_orders[-1].get("order_time", "12:00:00") if past_orders else "12:00:00",
        }

        tf = self.compute_time_features(past_orders, now_order)
        custf = self._customer_features(customer_id, past_orders, order_idx)
        archetype = custf.pop("archetype")  # Extract for matching

        rows = []

        for product in self.product_features.popularity.keys():
            # === HISTORY FEATURES ===
            in_hist = product in pf["past_products"]
            hist_count = pf["past_product_counts"].get(product, 0)
            hist_freq = hist_count / max(pf["n_orders"], 1)
            log_history = np.log1p(hist_count)

            last_seen = pf["past_product_last_idx"].get(product, -1)
            orders_since = order_idx - last_seen - 1 if last_seen >= 0 else 999
            time_decay = np.exp(-0.3 * orders_since) if in_hist else 0
            recent_flag = int(last_seen >= max(0, order_idx - 3))
            freq_pct = hist_count / max(pf["max_product_count"], 1)

            # === CATEGORY + SIZE ===
            category = self.category_map.get(product, "Unknown")
            cat_aff = pf["past_categories"].get(category, 0) / max(pf["total_items"], 1)
            is_pref_cat = int(category == pf["preferred_category"])

            size = self.extract_size(product)
            size_aff = pf["past_sizes"].get(size, 0) / max(pf["total_items"], 1)
            is_pref_size = int(size == pf["preferred_size"])

            base = self.get_base_product(product)
            has_variant = int(pf["past_base_products"].get(base, 0) > 0)

            # === DISCOVERY FEATURES ===
            cooccur_avg, cooccur_max = self.compute_cooccurrence_score(
                product, pf["past_products"]
            )
            archetype_match = self.compute_archetype_match(archetype, product)

            # === POPULARITY ===
            pop = self.product_features.popularity.get(product, 0)
            pop_scaled = pop * 0.1
            adj_pop = pop_scaled * (0.5 if orders_since > 3 and not in_hist else 1)

            row = {
                "customer_id": customer_id,
                "product": product,

                # History features
                "in_history": int(in_hist),
                "history_count": hist_count,
                "log_history_count": log_history,
                "history_freq": hist_freq,
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

                # Discovery features (NEW)
                "cooccur_with_history": cooccur_avg,
                "cooccur_max": cooccur_max,
                "archetype_product_match": archetype_match,
                "category_exploration_rate": category_exp,
                "product_exploration_rate": product_exp,
                "is_first_purchase": 0,  # Unknown at inference time

                # Time features
                **tf,

                # Customer features
                **custf,

                # Product features
                "popularity_scaled": pop_scaled,
                "adjusted_popularity": adj_pop,
                "category_encoded": self.encoders["category"].get(category, 0),
            }

            rows.append(row)

        return pd.DataFrame(rows)