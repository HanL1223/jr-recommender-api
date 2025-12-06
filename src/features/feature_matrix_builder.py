"""
FeatureMatrixBuilder
====================

Builds per-customer ML feature matrix that EXACTLY matches
TrainingDataBuilder features.

This eliminates inference mismatch and ensures
LightGBM/XGBoost produce meaningful scores.
"""

import pandas as pd
import numpy as np
from collections import Counter


class FeatureMatrixBuilder:
    def __init__(self, prepared_data, product_features, customer_profiles, category_map, encoders):
        self.prepared = prepared_data
        self.product_features = product_features
        self.customer_profiles = customer_profiles
        self.category_map = category_map or {}
        self.encoders = encoders  # {"segment":{}, "archetype":{}, "category":{}}

    # ------------------------------------------------------------
    # SIZE + BASE PRODUCT FUNCTIONS
    # ------------------------------------------------------------
    def extract_size(self, product):
        p = product.lower()
        if "extra large" in p: return "XL"
        if "large" in p: return "L"
        if "regular" in p: return "R"
        if "small" in p or "mini" in p: return "S"
        return "unknown"

    def get_base_product(self, product):
        return product.split(" (")[0] if " (" in product else product

    # ------------------------------------------------------------
    # CUSTOMER FEATURES (aligned with training)
    # ------------------------------------------------------------
    def _customer_features(self, customer_id, past_orders, order_idx):
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
        }

    # ------------------------------------------------------------
    # PAST FEATURES — identical to TrainingDataBuilder
    # ------------------------------------------------------------
    def compute_past_features(self, past_orders):
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

        return {
            "past_products": past_products,
            "past_product_counts": product_counts,
            "past_product_last_idx": last_idx,
            "past_categories": past_categories,
            "past_sizes": past_sizes,
            "past_base_products": past_base,
            "total_items": total_items,
            "n_orders": len(past_orders),
            "preferred_category": preferred_category,
            "preferred_size": preferred_size,
        }

    # ------------------------------------------------------------
    # TIME FEATURES (aligned with training)
    # ------------------------------------------------------------
    def compute_time_features(self, past_orders, now_order):
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

    # ------------------------------------------------------------
    # MASTER INFERENCE FEATURE BUILDER — 1:1 MATCH WITH TRAINING
    # ------------------------------------------------------------
    def build(self, customer_id):
        """
        Builds one row per product for this customer.
        No labels, no order_idx — just feature matrix.
        """

        history = self.prepared.customer_histories.get(customer_id, [])
        order_idx = len(history)
        past_orders = history

        pf = self.compute_past_features(past_orders)

        now_order = {
            "order_date": past_orders[-1]["order_date"] if past_orders else pd.Timestamp("2024-01-01"),
            "order_time": past_orders[-1].get("order_time", "12:00:00") if past_orders else "12:00:00",
        }

        tf = self.compute_time_features(past_orders, now_order)
        custf = self._customer_features(customer_id, past_orders, order_idx)

        rows = []

        # Loop through ALL products the model knows
        for product in self.product_features.popularity.keys():

            # ---- HISTORY FEATURES ----
            in_hist = product in pf["past_products"]
            hist_count = pf["past_product_counts"].get(product, 0)
            hist_freq = hist_count / max(pf["n_orders"], 1)

            last_seen = pf["past_product_last_idx"].get(product, -1)
            orders_since = order_idx - last_seen - 1 if last_seen >= 0 else 999
            time_decay = np.exp(-0.1 * orders_since) if in_hist else 0

            # ---- CATEGORY + SIZE ----
            category = self.category_map.get(product, "Unknown")
            cat_aff = pf["past_categories"].get(category, 0) / max(pf["total_items"], 1)
            is_pref_cat = int(category == pf["preferred_category"])

            size = self.extract_size(product)
            size_aff = pf["past_sizes"].get(size, 0) / max(pf["total_items"], 1)
            is_pref_size = int(size == pf["preferred_size"])

            base = self.get_base_product(product)
            has_variant = int(pf["past_base_products"].get(base, 0) > 0)

            # ---- GLOBAL POPULARITY ----
            pop = self.product_features.popularity.get(product, 0)
            adj_pop = pop * (0.3 if orders_since > 3 and not in_hist else 1)

            row = {
                "customer_id": customer_id,
                "product": product,

                # HISTORY
                "in_history": int(in_hist),
                "history_count": hist_count,
                "history_freq": hist_freq,
                "orders_since_last_purchase": min(orders_since, 100),
                "time_decay_score": time_decay,
                "never_purchased": int(not in_hist),

                # CATEGORY + SIZE
                "category_affinity": cat_aff,
                "is_preferred_category": is_pref_cat,
                "size_affinity": size_aff,
                "is_preferred_size": is_pref_size,
                "has_bought_variant": has_variant,

                # TIME FEATURES
                **tf,

                # CUSTOMER FEATURES
                **custf,

                # GLOBAL PRODUCT FEATURES
                "popularity": pop,
                "adjusted_popularity": adj_pop,
                "category_encoded": self.encoders["category"].get(category, 0),
            }

            rows.append(row)

        return pd.DataFrame(rows)
