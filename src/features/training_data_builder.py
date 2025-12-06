"""
Training Data Builder
=====================

Builds pointwise training samples for ranking models from PreparedData.

New personalised features added:
 - log_history_count
 - recent_purchase_flag
 - frequency_rank_pct
 - stronger time_decay_score
 - popularity_scaled (reduced influence)
 - adjusted_popularity (milder penalty)

Fully aligned with inference-side FeatureMatrixBuilder.
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

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


# ----------------------------------------------------------------------
# Updated Feature Names
# ----------------------------------------------------------------------
class TrainingDataBuilder:

    FEATURE_NAMES = [
        # History
        "in_history", "history_count", "log_history_count", "history_freq",
        "frequency_rank_pct", "recent_purchase_flag",
        "orders_since_last_purchase",

        # Affinity
        "time_decay_score", "never_purchased",
        "category_affinity", "is_preferred_category",
        "size_affinity", "is_preferred_size",
        "has_bought_variant",

        # Time features
        "hour_of_day", "day_of_week", "days_since_last_order",

        # Customer-level
        "history_length", "avg_basket_size", "avg_spend",
        "segment_encoded", "archetype_encoded",

        # Product-level global
        "popularity_scaled", "adjusted_popularity", "category_encoded",
    ]

    def __init__(self, negative_ratio: int = 5, random_seed: int = 42):
        self.negative_ratio = negative_ratio
        np.random.seed(random_seed)
        logger.info("TrainingDataBuilder initialised (negative_ratio=%s)", negative_ratio)

    # ------------------------------------------------------------------
    def build(self, prepared_data, product_features, customer_profiles) -> TrainingData:
        logger.info("Building training samples...")

        samples = []
        product_set = set(prepared_data.product_list)
        category_map = prepared_data.category_map or {}
        popularity = product_features.popularity or {}

        # Encoders for inference
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

        # Loop customers
        for idx, (cid, orders) in enumerate(prepared_data.customer_histories.items()):
            if len(orders) < 2:
                continue

            if idx % 200 == 0:
                logger.info("Processing %s/%s customers", idx, len(prepared_data.customer_histories))

            profile = customer_profiles.get(cid)
            archetype = profile.archetype if profile else "casual"
            archetype_encoded = archetype_to_idx.get(archetype, 0)

            segment = orders[0].get("segment", "Regular")
            segment_encoded = segment_to_idx.get(segment, 2)

            samples.extend(
                self.build_customer_samples(
                    cid, orders, product_set,
                    popularity, category_map, cat_to_idx,
                    segment_encoded, archetype_encoded
                )
            )

        if not samples:
            raise ValueError("TrainingDataBuilder produced zero samples!")

        df = pd.DataFrame(samples).replace([np.inf, -np.inf], 0).fillna(0)

        # Required for training pipeline
        required = {"order_idx", "order_date", "label", "customer_id", "product"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"TrainingDataBuilder missing required columns: {missing}")

        df["order_date"] = pd.to_datetime(df["order_date"])

        n_pos = int(df["label"].sum())
        n_neg = len(df) - n_pos

        logger.info("Built %s samples (pos=%s neg=%s)", len(df), n_pos, n_neg)

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

    # ------------------------------------------------------------------
    def build_customer_samples(
        self, cid, orders, product_set, popularity,
        category_map, cat_to_idx, segment_encoded, archetype_encoded
    ):
        samples = []

        for order_idx in range(1, len(orders)):
            past = orders[:order_idx]
            curr = orders[order_idx]

            basket = {p for p in curr["basket"] if isinstance(p, str)}
            if not basket:
                continue

            past_feats = self.compute_past_features(past, category_map)
            time_feats = self.compute_time_features(curr, past)

            cust_feats = {
                "history_length": order_idx,
                "avg_basket_size": past_feats["avg_basket_size"],
                "avg_spend": past_feats["avg_spend"],
                "segment_encoded": segment_encoded,
                "archetype_encoded": archetype_encoded,
            }

            # Positive
            for p in basket:
                samples.append(
                    self.create_sample(
                        cid, p, 1, order_idx, curr["order_date"],
                        past_feats, time_feats, cust_feats,
                        popularity, category_map, cat_to_idx
                    )
                )

            # Negative
            neg_pool = list(product_set - basket)
            n_neg = min(len(basket) * self.negative_ratio, len(neg_pool))

            if n_neg > 0:
                weights = np.array([popularity.get(p, 0.0001) for p in neg_pool])
                weights /= weights.sum()
                negs = np.random.choice(neg_pool, size=n_neg, replace=False, p=weights)

                for p in negs:
                    samples.append(
                        self.create_sample(
                            cid, p, 0, order_idx, curr["order_date"],
                            past_feats, time_feats, cust_feats,
                            popularity, category_map, cat_to_idx
                        )
                    )
        return samples

    # ------------------------------------------------------------------
    def create_sample(
        self, cid, product, label, order_idx, order_date,
        past, time, cust, popularity, category_map, cat_to_idx
    ):
        # --- History ---
        hist_count = past["past_product_counts"].get(product, 0)
        history_freq = hist_count / max(past["n_orders"], 1)
        log_history = np.log1p(hist_count)

        in_hist = int(product in past["past_products"])

        last_idx = past["past_product_last_idx"].get(product, -1)
        if last_idx >= 0:
            orders_since = order_idx - last_idx - 1
        else:
            orders_since = 999

        recent_flag = int(last_idx >= max(0, order_idx - 3))

        # Stronger time decay
        time_decay = np.exp(-0.3 * orders_since) if in_hist else 0.0

        # Frequency pct (normalised by customer's max freq)
        freq_pct = hist_count / max(past["max_product_count"], 1)

        # --- Category & Size ---
        category = category_map.get(product, "Unknown")
        cat_aff = past["past_categories"].get(category, 0) / max(past["total_items"], 1)
        is_pref_cat = int(category == past["preferred_category"])

        size = self.extract_size(product)
        size_aff = past["past_sizes"].get(size, 0) / max(past["total_items"], 1)
        is_pref_size = int(size == past["preferred_size"])

        base = self.get_base_product(product)
        has_variant = int(past["past_base_products"].get(base, 0) > 0)

        # --- Popularity (reduced) ---
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

            # Affinity
            "time_decay_score": time_decay,
            "never_purchased": int(not in_hist),
            "category_affinity": cat_aff,
            "is_preferred_category": is_pref_cat,
            "size_affinity": size_aff,
            "is_preferred_size": is_pref_size,
            "has_bought_variant": has_variant,

            # Time
            **time,
            # Customer
            **cust,

            # Global
            "popularity_scaled": pop_scaled,
            "adjusted_popularity": adjusted_pop,
            "category_encoded": cat_to_idx.get(category, 0),
        }

    # ------------------------------------------------------------------
    def extract_size(self, p):
        p = p.lower()
        if "extra large" in p: return "XL"
        if "large" in p: return "L"
        if "regular" in p: return "R"
        if "small" in p or "mini" in p: return "S"
        return "unknown"

    def get_base_product(self, product):
        return product.split(" (")[0] if " (" in product else product

    # ------------------------------------------------------------------
    def compute_time_features(self, curr, past):
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

    # ------------------------------------------------------------------
    def compute_past_features(self, past_orders, category_map):
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

                size = self.extract_size(p)
                size_counts[size] += 1

                base = self.get_base_product(p)
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
