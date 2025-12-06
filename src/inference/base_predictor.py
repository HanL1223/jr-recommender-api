from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd


class BasePredictor(ABC):
    """Interface for all predictor classes."""

    @abstractmethod
    def recommend(self, customer_id: int, top_k: int = 5) -> Any:
        """Produce recommendations for a customer."""
        pass

    @abstractmethod
    def format_prediction(self, prediction) -> str:
        """Pretty-print."""
        pass


class BaseColdStart(ABC):
    """Interface for cold-start recommendation strategies."""

    @abstractmethod
    def recommend(self, top_k: int = 5, **kwargs) -> List[Any]:
        """Produce recommendations for new customers."""
        pass
