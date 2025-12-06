import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
#                           SPLIT DATA CLASS
# =====================================================================
@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

    split_date_train: str
    split_date_valid: str

    feature_names: List[str]

    n_train_samples: int
    n_valid_samples: int
    n_test_samples: int

    n_train_customers: int
    n_valid_customers: int
    n_test_customers: int

    overlap_train_valid: int
    overlap_valid_test: int
    new_customers_valid: int
    new_customers_test: int


# =====================================================================
#                     TEMPORAL TRAIN/VAL/TEST SPLITTER
# =====================================================================
class TemporalDataSplitter:
    """
    Chronological (time-based) train/valid/test splitter.

    Ensures:
    - No rows are lost
    - Required ranking columns are preserved (customer_id, order_idx)
    - Non-overlapping time windows
    """

    def __init__(self, valid_ratio: float = 0.1, test_ratio: float = 0.2):
        if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1:
            raise ValueError("valid_ratio + test_ratio must be < 1")

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        logger.info(
            f"TemporalDataSplitter initialized (valid_ratio={valid_ratio}, "
            f"test_ratio={test_ratio})"
        )

    # ------------------------------------------------------------------
    def split(self, training_data, date_column: str = "order_date") -> SplitData:
        logger.info("Performing temporal train/valid/test split")

        df = training_data.samples_df.copy()

        # ----------------------------------------------
        # Validate required columns
        # ----------------------------------------------
        required_cols = {"customer_id", "product", "label", "order_idx", date_column}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required training columns: {missing}")

        # Ensure correct type
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Sort chronologically
        df = df.sort_values(date_column)

        # ----------------------------------------------
        # Compute split indices
        # ----------------------------------------------
        n = len(df)
        idx_train_end = int(n * (1 - self.valid_ratio - self.test_ratio))
        idx_valid_end = int(n * (1 - self.test_ratio))

        split_date_train = df[date_column].iloc[idx_train_end]
        split_date_valid = df[date_column].iloc[idx_valid_end]

        logger.info(f"Train split date: {split_date_train}")
        logger.info(f"Valid split date: {split_date_valid}")

        # ----------------------------------------------
        # Slice splits (non-overlapping)
        # ----------------------------------------------
        train_df = df.iloc[:idx_train_end].copy()
        valid_df = df.iloc[idx_train_end:idx_valid_end].copy()
        test_df = df.iloc[idx_valid_end:].copy()

        # ----------------------------------------------
        # Stats needed for SplitData
        # ----------------------------------------------
        train_customers = set(train_df.customer_id.unique())
        valid_customers = set(valid_df.customer_id.unique())
        test_customers = set(test_df.customer_id.unique())

        # ----------------------------------------------
        # Return packaged split
        # ----------------------------------------------
        return SplitData(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,

            split_date_train=split_date_train.strftime("%Y-%m-%d"),
            split_date_valid=split_date_valid.strftime("%Y-%m-%d"),

            feature_names=training_data.feature_names,

            n_train_samples=len(train_df),
            n_valid_samples=len(valid_df),
            n_test_samples=len(test_df),

            n_train_customers=len(train_customers),
            n_valid_customers=len(valid_customers),
            n_test_customers=len(test_customers),

            overlap_train_valid=len(train_customers & valid_customers),
            overlap_valid_test=len(valid_customers & test_customers),

            new_customers_valid=len(valid_customers - train_customers),
            new_customers_test=len(test_customers - (train_customers | valid_customers)),
        )
