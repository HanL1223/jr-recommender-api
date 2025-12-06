"""
Hyperparameter Tuner using Optuna
================================
Supports:
- LightGBMRanker
- XGBoostRanker
- Any future model via Strategy Pattern

Usage:
    tuner = HyperparameterTuner()
    result = tuner.tune(split, LightGBMTuningStrategy())
"""

import optuna
import logging
from dataclasses import dataclass
from typing import Dict, Any

from src.evaluation.metrics import RankingMetrics
from src.tuning.model_tuning_strategy import ModelTuningStrategy

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Result container for tuning output
# ----------------------------------------------------------
@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    model_name: str


# ----------------------------------------------------------
# Main Hyperparameter Tuner
# ----------------------------------------------------------
class HyperparameterTuner:

    def __init__(self, metric: str = "ndcg@3", direction: str = "maximize",
                 n_trials: int = 20, timeout: int = 600):
        """
        Hyperparameter tuning engine using Optuna.

        :param metric: Metric to optimize (e.g. 'ndcg@3')
        :param direction: 'maximize' or 'minimize'
        :param n_trials: Number of Optuna trials
        :param timeout: Max seconds to run tuning
        """
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout

        logger.info(
            f"HyperparameterTuner initialized (metric={metric}, "
            f"direction={direction}, trials={n_trials}, timeout={timeout}s)"
        )

    # ------------------------------------------------------
    # Generic tune() entry point (strategy-driven)
    # ------------------------------------------------------
    def tune(self, split, strategy: ModelTuningStrategy):
        """
        Tune hyperparameters for a given model strategy.

        :param split: SplitData (train/valid/test)
        :param strategy: A ModelTuningStrategy implementation
        :return: TuningResult
        """

        train_df = split.train_df
        valid_df = split.valid_df
        feature_names = split.feature_names

        evaluator = RankingMetrics(k_values=[int(self.metric.split('@')[1])])

        logger.info(f"Starting tuning for model: {strategy.model_name}")
        logger.info(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

        # --------------------------------------------------
        # Objective function for Optuna
        # --------------------------------------------------
        def objective(trial):
            # Step 1: Let strategy define search space
            params = strategy.suggest_params(trial)

            # Step 2: Build model using strategy
            model = strategy.create_model(params)

            # Step 3: Train model
            model.fit(train_df, feature_names, valid_df)

            # Step 4: Evaluate on validation set ONLY (no test leakage)
            metrics = evaluator.evaluate(model, valid_df, feature_names)

            score = metrics.metrics[self.metric]
            logger.debug(f"Trial {trial.number} — Score: {score:.5f}")

            return score

        # --------------------------------------------------
        # Run Optuna search
        # --------------------------------------------------
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        logger.info(f"Best params for {strategy.model_name}: {study.best_params}")
        logger.info(f"Best {self.metric}: {study.best_value:.5f}")

        # --------------------------------------------------
        # Return structured result
        # --------------------------------------------------
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            model_name=strategy.model_name,
        )
