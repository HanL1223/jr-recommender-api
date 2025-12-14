"""
Data Splitter
=============
Temporal train/valid/test split with customer overlap guarantees.

This module provides two splitting strategies:
1. Pure temporal split - chronological cutoff across all data
2. Stratified temporal split - per-customer chronological split

For personalized recommendations, stratified temporal is preferred as it
guarantees every customer in validation/test has purchase history in training.

Example:
    >>> splitter = TemporalDataSplitter(method="stratified_temporal")
    >>> split = splitter.split(training_data)
    >>> print(split.summary())
"""

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """
    Container for train/valid/test split results.
    
    Attributes:
        train_df: Training samples
        valid_df: Validation samples
        test_df: Test samples
        split_date_train: Cutoff date for training (or "per-customer")
        split_date_valid: Cutoff date for validation (or "per-customer")
        split_method: "temporal" or "stratified_temporal"
        feature_names: List of feature column names
        n_train_samples: Number of training samples
        n_valid_samples: Number of validation samples
        n_test_samples: Number of test samples
        n_train_customers: Unique customers in training
        n_valid_customers: Unique customers in validation
        n_test_customers: Unique customers in test
        overlap_train_valid: Customers appearing in both train and valid
        overlap_valid_test: Customers appearing in both valid and test
        overlap_train_test: Customers appearing in both train and test
        new_customers_valid: Customers in valid not seen in train
        new_customers_test: Customers in test not seen in train or valid
        customer_coverage_valid: Fraction of valid customers with training history
        customer_coverage_test: Fraction of test customers with training history
    """
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

    split_date_train: str
    split_date_valid: str
    split_method: str

    feature_names: List[str]

    n_train_samples: int
    n_valid_samples: int
    n_test_samples: int

    n_train_customers: int
    n_valid_customers: int
    n_test_customers: int

    overlap_train_valid: int
    overlap_valid_test: int
    overlap_train_test: int

    new_customers_valid: int
    new_customers_test: int

    customer_coverage_valid: float
    customer_coverage_test: float

    def summary(self) -> str:
        """
        Generate human-readable summary of the split.
        
        Returns:
            Formatted string with split statistics and diagnostics.
        """
        lines = [
            "=" * 60,
            f"SPLIT SUMMARY ({self.split_method})",
            "=" * 60,
            "",
            "SAMPLE COUNTS",
            "-" * 40,
            f"  Train: {self.n_train_samples:>10,} samples ({self.n_train_customers:,} customers)",
            f"  Valid: {self.n_valid_samples:>10,} samples ({self.n_valid_customers:,} customers)",
            f"  Test:  {self.n_test_samples:>10,} samples ({self.n_test_customers:,} customers)",
            "",
            "CUSTOMER OVERLAP",
            "-" * 40,
            f"  Train and Valid: {self.overlap_train_valid:,} customers",
            f"  Train and Test:  {self.overlap_train_test:,} customers",
            f"  Valid and Test:  {self.overlap_valid_test:,} customers",
            "",
            "COVERAGE (customers with history in train)",
            "-" * 40,
            f"  Valid coverage: {self.customer_coverage_valid:.1%}",
            f"  Test coverage:  {self.customer_coverage_test:.1%}",
            "",
            "COLD-START CUSTOMERS (no training history)",
            "-" * 40,
            f"  New in valid: {self.new_customers_valid:,}",
            f"  New in test:  {self.new_customers_test:,}",
        ]

        # Add warnings if coverage is low
        if self.customer_coverage_test < 0.8:
            lines.extend([
                "",
                "WARNING: Only {:.1%} of test customers have training history.".format(
                    self.customer_coverage_test
                ),
                "Consider using stratified_temporal split for better coverage.",
            ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert split metadata to dictionary for logging/serialization.
        
        Returns:
            Dictionary with all split statistics (excludes DataFrames).
        """
        return {
            "split_method": self.split_method,
            "split_date_train": self.split_date_train,
            "split_date_valid": self.split_date_valid,
            "n_train_samples": self.n_train_samples,
            "n_valid_samples": self.n_valid_samples,
            "n_test_samples": self.n_test_samples,
            "n_train_customers": self.n_train_customers,
            "n_valid_customers": self.n_valid_customers,
            "n_test_customers": self.n_test_customers,
            "overlap_train_valid": self.overlap_train_valid,
            "overlap_valid_test": self.overlap_valid_test,
            "overlap_train_test": self.overlap_train_test,
            "new_customers_valid": self.new_customers_valid,
            "new_customers_test": self.new_customers_test,
            "customer_coverage_valid": self.customer_coverage_valid,
            "customer_coverage_test": self.customer_coverage_test,
        }


class TemporalDataSplitter:
    """
    Chronological train/valid/test splitter with multiple strategies.
    
    Two splitting methods are available:
    
    1. "temporal" (default):
       Pure chronological split using global date cutoffs.
       Fast but may leave some customers with no training history.
       
    2. "stratified_temporal":
       Per-customer chronological split. For each customer, early orders
       go to train, later orders to valid/test. Guarantees every customer
       in valid/test has history in train.
    
    Example:
        >>> splitter = TemporalDataSplitter(
        ...     valid_ratio=0.1,
        ...     test_ratio=0.2,
        ...     method="stratified_temporal"
        ... )
        >>> split = splitter.split(training_data)
        >>> print(f"Test coverage: {split.customer_coverage_test:.1%}")
    
    Args:
        valid_ratio: Fraction of data for validation set (default: 0.1)
        test_ratio: Fraction of data for test set (default: 0.2)
        method: Splitting strategy - "temporal" or "stratified_temporal"
        min_orders_per_customer: Minimum orders required per customer
            for stratified split (default: 2)
    """

    VALID_METHODS = ("temporal", "stratified_temporal")

    def __init__(
        self,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.2,
        method: Literal["temporal", "stratified_temporal"] = "temporal",
        min_orders_per_customer: int = 2
    ):
        if valid_ratio < 0 or test_ratio < 0:
            raise ValueError("Ratios must be non-negative")
        if (valid_ratio + test_ratio) >= 1.0:
            raise ValueError("valid_ratio + test_ratio must be less than 1.0")
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        if min_orders_per_customer < 2:
            raise ValueError("min_orders_per_customer must be at least 2")

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.method = method
        self.min_orders_per_customer = min_orders_per_customer

        logger.info(
            "TemporalDataSplitter initialized: method=%s, valid_ratio=%.2f, "
            "test_ratio=%.2f, min_orders=%d",
            method, valid_ratio, test_ratio, min_orders_per_customer
        )

    def split(self, training_data, date_column: str = "order_date") -> SplitData:
        """
        Split training data into train/valid/test sets.
        
        Args:
            training_data: Object with samples_df and feature_names attributes
            date_column: Name of the date column for temporal ordering
            
        Returns:
            SplitData containing train/valid/test DataFrames and metadata
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If data is insufficient for splitting
        """
        if self.method == "stratified_temporal":
            return self._split_stratified_temporal(training_data, date_column)
        return self._split_temporal(training_data, date_column)

    def _split_temporal(self, training_data, date_column: str) -> SplitData:
        """
        Pure chronological split using global date cutoffs.
        
        All data before cutoff_1 goes to train, between cutoff_1 and cutoff_2
        goes to valid, and after cutoff_2 goes to test.
        """
        logger.info("Performing pure temporal split")

        df = training_data.samples_df.copy()
        self._validate_columns(df, date_column)

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        df = df.sort_values(date_column).reset_index(drop=True)

        # Compute split indices
        n = len(df)
        idx_train_end = int(n * (1 - self.valid_ratio - self.test_ratio))
        idx_valid_end = int(n * (1 - self.test_ratio))

        if idx_train_end < 1 or idx_valid_end <= idx_train_end:
            raise ValueError(
                f"Insufficient data for split: {n} samples with "
                f"valid_ratio={self.valid_ratio}, test_ratio={self.test_ratio}"
            )

        split_date_train = df[date_column].iloc[idx_train_end]
        split_date_valid = df[date_column].iloc[idx_valid_end]

        logger.info("Train cutoff: %s", split_date_train)
        logger.info("Valid cutoff: %s", split_date_valid)

        train_df = df.iloc[:idx_train_end].copy()
        valid_df = df.iloc[idx_train_end:idx_valid_end].copy()
        test_df = df.iloc[idx_valid_end:].copy()

        return self._build_split_data(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            split_date_train=split_date_train.strftime("%Y-%m-%d"),
            split_date_valid=split_date_valid.strftime("%Y-%m-%d"),
            feature_names=training_data.feature_names,
            method="temporal"
        )

    def _split_stratified_temporal(self, training_data, date_column: str) -> SplitData:
        """
        Per-customer chronological split.
        
        For each customer, orders are sorted by date and split such that
        early orders go to train, middle to valid, and recent to test.
        This guarantees every customer in valid/test has history in train.
        """
        logger.info("Performing stratified temporal split (per-customer)")

        df = training_data.samples_df.copy()
        self._validate_columns(df, date_column)

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Filter to customers with enough orders
        order_counts = df.groupby("customer_id")["order_idx"].nunique()
        eligible_customers = order_counts[
            order_counts >= self.min_orders_per_customer
        ].index

        n_excluded = len(order_counts) - len(eligible_customers)
        if n_excluded > 0:
            logger.warning(
                "Excluding %d customers with fewer than %d orders",
                n_excluded, self.min_orders_per_customer
            )

        if len(eligible_customers) == 0:
            raise ValueError(
                f"No customers have >= {self.min_orders_per_customer} orders"
            )

        df = df[df["customer_id"].isin(eligible_customers)].copy()
        logger.info(
            "Processing %d customers with >= %d orders",
            len(eligible_customers), self.min_orders_per_customer
        )

        train_rows, valid_rows, test_rows = [], [], []

        for cust_id, cust_df in df.groupby("customer_id"):
            cust_df = cust_df.sort_values(date_column)
            orders = cust_df["order_idx"].unique()
            n_orders = len(orders)

            # Calculate split points
            n_train = max(1, int(n_orders * (1 - self.valid_ratio - self.test_ratio)))
            n_valid = max(0, int(n_orders * self.valid_ratio))

            # Ensure at least 1 order in test if customer has enough orders
            if n_orders >= 3 and n_valid == 0:
                n_valid = 1

            train_orders = set(orders[:n_train])
            valid_orders = set(orders[n_train:n_train + n_valid])
            test_orders = set(orders[n_train + n_valid:])

            train_rows.append(cust_df[cust_df["order_idx"].isin(train_orders)])
            if valid_orders:
                valid_rows.append(cust_df[cust_df["order_idx"].isin(valid_orders)])
            if test_orders:
                test_rows.append(cust_df[cust_df["order_idx"].isin(test_orders)])

        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        valid_df = pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()

        return self._build_split_data(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            split_date_train="per-customer",
            split_date_valid="per-customer",
            feature_names=training_data.feature_names,
            method="stratified_temporal"
        )

    def _validate_columns(self, df: pd.DataFrame, date_column: str) -> None:
        """Validate that required columns are present."""
        required_cols = {"customer_id", "product", "label", "order_idx", date_column}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _build_split_data(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_date_train: str,
        split_date_valid: str,
        feature_names: List[str],
        method: str
    ) -> SplitData:
        """Compute statistics and build SplitData container."""
        train_customers = set(train_df["customer_id"].unique()) if len(train_df) > 0 else set()
        valid_customers = set(valid_df["customer_id"].unique()) if len(valid_df) > 0 else set()
        test_customers = set(test_df["customer_id"].unique()) if len(test_df) > 0 else set()

        overlap_train_valid = len(train_customers & valid_customers)
        overlap_valid_test = len(valid_customers & test_customers)
        overlap_train_test = len(train_customers & test_customers)

        new_customers_valid = len(valid_customers - train_customers)
        new_customers_test = len(test_customers - (train_customers | valid_customers))

        coverage_valid = (
            overlap_train_valid / len(valid_customers) 
            if valid_customers else 1.0
        )
        coverage_test = (
            overlap_train_test / len(test_customers) 
            if test_customers else 1.0
        )

        logger.info(
            "Split complete: train=%d, valid=%d, test=%d samples",
            len(train_df), len(valid_df), len(test_df)
        )
        logger.info(
            "Customer coverage: valid=%.1f%%, test=%.1f%%",
            coverage_valid * 100, coverage_test * 100
        )

        return SplitData(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            split_date_train=split_date_train,
            split_date_valid=split_date_valid,
            split_method=method,
            feature_names=feature_names,
            n_train_samples=len(train_df),
            n_valid_samples=len(valid_df),
            n_test_samples=len(test_df),
            n_train_customers=len(train_customers),
            n_valid_customers=len(valid_customers),
            n_test_customers=len(test_customers),
            overlap_train_valid=overlap_train_valid,
            overlap_valid_test=overlap_valid_test,
            overlap_train_test=overlap_train_test,
            new_customers_valid=new_customers_valid,
            new_customers_test=new_customers_test,
            customer_coverage_valid=coverage_valid,
            customer_coverage_test=coverage_test,
        )


def create_discovery_split(
    training_data,
    date_column: str = "order_date",
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    min_orders: int = 3
) -> SplitData:
    """
    Convenience function for discovery-focused evaluation.
    
    Uses stratified temporal split to ensure all test customers
    have purchase history in training, which is required for
    meaningful discovery (new item) evaluation.
    
    Args:
        training_data: Object with samples_df and feature_names attributes
        date_column: Name of date column
        valid_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        min_orders: Minimum orders per customer to include
        
    Returns:
        SplitData with guaranteed customer overlap
    """
    splitter = TemporalDataSplitter(
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        method="stratified_temporal",
        min_orders_per_customer=min_orders
    )
    return splitter.split(training_data, date_column)