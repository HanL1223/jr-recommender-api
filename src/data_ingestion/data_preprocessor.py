"""
Data Preprocessor
=================
Transforms raw data into customer histories for modeling.

Refactored from: A_data_preparation.py
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PreparedData:
    """Container for preprocessed data ready for feature engineering."""
    customer_histories: Dict[int, List[dict]]  # customer_id -> list of orders
    customer_list: List[int]
    product_list: List[str]
    category_map: Dict[str, str]  # product -> category
    n_customers: int
    n_products: int
    n_orders: int




class BasePreprocessor(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame):
        pass
"""
PLACEHOLDER - Sampling
"""

class SequencePreprocessor(BasePreprocessor):
    def __init__(self,min_orders:int = 2):
        self.min_orders = min_orders
        logger.info(f"DataPreprocessor initialized (min_orders={min_orders})")

    def aggregate_to_orders(self,df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Aggregating transactions to order level")
        df = df.copy()

        if "product" not in df.columns:
            raise ValueError(f"Product not done at product + variance level")
        
        orders = df.groupby("order_id").agg({
            "customer_id": "first",
            "order_date": "first",
            "order_time": "first" if "order_time" in df.columns else lambda x: "12:00:00",
            "order_total_price": "first",
            "product": list,
            "product_category": (lambda x: list(x) if "product_category" in df.columns else []),
            "customer_segment": "first" if "customer_segment" in df.columns else lambda x: "Regular"
        }).reset_index()


        #Rename for readlibility
        orders.columns = [
            "order_id", "customer_id", "order_date", "order_time",
            "order_total", "basket", "categories", "segment"
        ]

        orders['basket_size'] = orders['basket'].apply(len)
        logger.info(f"Aggregated to {len(orders):,} orders")
        return orders
    
    def build_customer_histories(self,orders_df:pd.DataFrame) -> Dict[int, List[dict]]:
        logger.info("Building customer order histories...")

        orders_df = orders_df.sort_values(['customer_id','order_date'])
        #Each key will create a empty list
        #Then append value by key
        histories = defaultdict(list)

        for _, row in orders_df.iterrows():
            order = {
                "order_id": row["order_id"],
                "order_date": row["order_date"],
                "order_time": row["order_time"],
                "order_total": row["order_total"],
                "basket": row["basket"],
                "basket_size": row["basket_size"],
                "categories": row["categories"],
                "segment": row["segment"],
            }
            histories[row["customer_id"]].append(order)
        logger.info(f"Built histories for {len(histories):,} customers")
        return dict(histories)
    
    def extract_products(self,df:pd.DataFrame) -> List[str]:
        """Extract sorted unique product names """
        products = [p for p in df["product"].dropna().unique() if isinstance(p, str)]
        return sorted(products)
    
    def build_category_map(self,df:pd.DataFrame) -> Dict[str, str]:
        """Create Product:Category dict"""
        if "product_category" not in df.columns:
            return {}
        category_map = {}
        for _,row in df[['product','product_category']].drop_duplicates().iterrows():
            if isinstance(row['product'],str):
                #Error handler product name should be string only
                category_map[row['product']] = row['product_category']
        return category_map
    
    def transform(self,df:pd.DataFrame) ->PreparedData:
        logging.info(f"Staring preprocessing")

        #aggregate_to_orders
        orders_df = self.aggregate_to_orders(df)

        #Build customer-level sequential histories
        histories = self.build_customer_histories(orders_df)
        #Filter minimum # of order per customer id 
        filtered_histories = {
            id: h for id, h in histories.items()
            if len(h) >= self.min_orders
        }

        #Build product categories
        product_list = self.extract_products(df)
        category_map = self.build_category_map(df)

        return PreparedData(
            customer_histories=filtered_histories,
            customer_list=list(filtered_histories.keys()),
            product_list=product_list,
            category_map=category_map,
            n_customers=len(filtered_histories),
            n_products=len(product_list),
            n_orders=sum(len(h) for h in filtered_histories.values())
        )

class PreprocessingFactory:
    @staticmethod

    def create(method: str, **kwargs) ->BasePreprocessor:
        method = method.lower()

        if method == "sequence":
            min_orders = kwargs.pop("min_orders", 2)  
            return SequencePreprocessor(
                min_orders=kwargs.get("min_orders",2),
                **kwargs
            )
        raise ValueError(f"Unknown preprocessing method: {method}")
    

if __name__ == "__main__":
    pass
    """
    Use example

    df = raw.transactions

    preprocessor = PreprocessingFactory.create(
    method="sequence",
    min_orders=2
)
    prepared = preprocessor.transform(df)


    """


