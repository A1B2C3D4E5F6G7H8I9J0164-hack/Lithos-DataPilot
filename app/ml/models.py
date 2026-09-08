"""
Model candidate definitions, factory, and Optuna search spaces for classification and regression.
"""
from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional
import optuna
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

from .problem_detector import ProblemType


@dataclass
class ModelCandidate:
    """Metadata and constructor for a candidate ML algorithm."""
    model_id: str
    display_name: str
    problem_type: ProblemType
    estimator_factory: Callable[[int], BaseEstimator]
    search_space_fn: Callable[[optuna.Trial], Dict[str, Any]]


class ModelFactory:
    """Provides algorithm candidates and hyperparameter search spaces."""

    @classmethod
    def get_candidates(cls, problem_type: ProblemType, random_seed: int = 42) -> List[ModelCandidate]:
        """Returns candidate models tailored for the detected problem type."""
        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            return cls._get_classification_candidates(problem_type, random_seed)
        else:
            return cls._get_regression_candidates(random_seed)

    @classmethod
    def _get_classification_candidates(
        cls, problem_type: ProblemType, random_seed: int
    ) -> List[ModelCandidate]:
        return [
            ModelCandidate(
                model_id="logistic_regression",
                display_name="Logistic Regression",
                problem_type=problem_type,
                estimator_factory=lambda seed: LogisticRegression(
                    max_iter=1000, random_state=seed, class_weight="balanced"
                ),
                search_space_fn=lambda trial: {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                    "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                },
            ),
            ModelCandidate(
                model_id="random_forest",
                display_name="Random Forest Classifier",
                problem_type=problem_type,
                estimator_factory=lambda seed: RandomForestClassifier(
                    n_estimators=100, max_depth=12, random_state=seed, class_weight="balanced"
                ),
                search_space_fn=lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                    "max_depth": trial.suggest_int("max_depth", 4, 18),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                },
            ),
            ModelCandidate(
                model_id="hist_gradient_boosting",
                display_name="Gradient Boosting (Hist)",
                problem_type=problem_type,
                estimator_factory=lambda seed: HistGradientBoostingClassifier(
                    max_iter=100, random_state=seed
                ),
                search_space_fn=lambda trial: {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_iter": trial.suggest_int("max_iter", 50, 150, step=25),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
                },
            ),
            ModelCandidate(
                model_id="xgboost",
                display_name="XGBoost Classifier",
                problem_type=problem_type,
                estimator_factory=lambda seed: XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.08,
                    random_state=seed,
                    eval_metric="logloss",
                ),
                search_space_fn=lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                },
            ),
        ]

    @classmethod
    def _get_regression_candidates(cls, random_seed: int) -> List[ModelCandidate]:
        return [
            ModelCandidate(
                model_id="ridge",
                display_name="Ridge Regression",
                problem_type=ProblemType.REGRESSION,
                estimator_factory=lambda seed: Ridge(alpha=1.0, random_state=seed),
                search_space_fn=lambda trial: {
                    "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
                },
            ),
            ModelCandidate(
                model_id="random_forest_regressor",
                display_name="Random Forest Regressor",
                problem_type=ProblemType.REGRESSION,
                estimator_factory=lambda seed: RandomForestRegressor(
                    n_estimators=100, max_depth=12, random_state=seed
                ),
                search_space_fn=lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                    "max_depth": trial.suggest_int("max_depth", 4, 18),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                },
            ),
            ModelCandidate(
                model_id="hist_gradient_boosting_regressor",
                display_name="Gradient Boosting Regressor",
                problem_type=ProblemType.REGRESSION,
                estimator_factory=lambda seed: HistGradientBoostingRegressor(
                    max_iter=100, random_state=seed
                ),
                search_space_fn=lambda trial: {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_iter": trial.suggest_int("max_iter", 50, 150, step=25),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
                },
            ),
            ModelCandidate(
                model_id="xgboost_regressor",
                display_name="XGBoost Regressor",
                problem_type=ProblemType.REGRESSION,
                estimator_factory=lambda seed: XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.08,
                    random_state=seed,
                ),
                search_space_fn=lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                },
            ),
        ]
