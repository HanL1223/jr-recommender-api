"""
Hyperparameter Tuner using Optuna
================================
Supports:
- LightGBMRanker
- XGBoostRanker
- ...

Trainer calls:
    tuner.tune_lightgbm(split)
    tuner.tune_xgboost(split)
    split = output of data_splitter.py

"""
from abc import ABC,abstractmethod
from typing import Dict, Any
from src.models.lightgbm_ranker import LightGBMRanker
from src.models.xgboost_ranker import XGBoostRanker
#For any new model - import here

class ModelTuningStrategy(ABC):
    """
    Abstract strategy for model-specific tuning logic
    """

    @property    
    @abstractmethod
    def model_name(self)->str:
        pass

    @abstractmethod
    def suggest_params(self,trial) -> Dict[str,Any]:
        pass

    @abstractmethod
    def create_model(self,params:Dict[str,Any]):
        pass

    

#LGBM Strategy
class LightGBMTuningStrategy(ModelTuningStrategy):
    def model_name(self):
        return "LightGBM"
    def suggest_params(self, trial):
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
            "num_boost_round": 400,
            "early_stopping_rounds": 50,
        }
    def create_model(self, params):
        return LightGBMRanker.from_params(params)
    

#XGBoost Strategy
class XGBoostTuningStrategy(ModelTuningStrategy):
    def model_name(self):
        return "XGBoost"
    def suggest_params(self, trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 500,
            "early_stopping_rounds": 50,
        }
    
    def create_model(self, params):
        return XGBoostRanker.from_params(params)