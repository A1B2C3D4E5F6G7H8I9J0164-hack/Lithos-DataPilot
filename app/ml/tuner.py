"""
Optuna Bayesian hyperparameter optimization engine for candidate models.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .problem_detector import ProblemSpec
from .models import ModelCandidate
from .cross_validation import CrossValidationRunner
from app.data.preprocessing import LeakageSafePreprocessor

# Suppress verbose Optuna logging to stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrialRecord:
    """Detailed record of an individual Optuna trial."""
    trial_number: int
    score: float
    params: Dict[str, Any]
    duration_sec: float


@dataclass
class TuningResult:
    """Consolidated outcome of the hyperparameter optimization process."""
    model_id: str
    model_name: str
    metric_name: str
    direction: str  # 'maximize' or 'minimize'
    baseline_score: float
    best_score: float
    score_delta: float
    best_params: Dict[str, Any]
    trials_history: List[TrialRecord]
    total_trials: int
    duration_sec: float
    optimized_pipeline: Optional[Pipeline] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            "Metric": self.metric_name,
            "Baseline Score": round(self.baseline_score, 4),
            "Best Tuned Score": round(self.best_score, 4),
            "Improvement (Delta)": round(self.score_delta, 4),
            "Total Trials": self.total_trials,
            "Optimization Time (s)": round(self.duration_sec, 2),
            "Best Parameters": self.best_params,
        }


class HyperparameterTuner:
    """Executes Bayesian hyperparameter tuning using Optuna on top-ranked pipelines."""

    LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mse", "log_loss", "mape"}

    @classmethod
    def tune(
        cls,
        candidate: ModelCandidate,
        baseline_score: float,
        preprocessor_builder: LeakageSafePreprocessor,
        problem_spec: ProblemSpec,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 15,
        timeout: Optional[int] = 120,
        n_splits: int = 3,  # Faster 3-fold CV during inner HPO search
        random_seed: int = 42,
        trial_callback: Optional[Callable[[int, float, Dict[str, Any]], None]] = None,
    ) -> TuningResult:
        """
        Runs an Optuna study optimizing the candidate model's hyperparameters.
        """
        metric = problem_spec.recommended_primary_metric
        direction = "minimize" if metric in cls.LOWER_IS_BETTER_METRICS else "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_seed))

        trials_history: List[TrialRecord] = []
        t_start = time.time()

        def objective(trial: optuna.Trial) -> float:
            t0 = time.time()
            suggested_params = candidate.search_space_fn(trial)

            # Create candidate model with suggested parameters
            base_model = candidate.estimator_factory(random_seed)
            base_model.set_params(**suggested_params)

            # Evaluate with cross-validation
            # Create a temporary candidate wrapper
            temp_candidate = ModelCandidate(
                model_id=candidate.model_id,
                display_name=candidate.display_name,
                problem_type=candidate.problem_type,
                estimator_factory=lambda s: clone(base_model),
                search_space_fn=candidate.search_space_fn,
            )

            cv_res = CrossValidationRunner.run_cv(
                candidate=temp_candidate,
                preprocessor_builder=preprocessor_builder,
                problem_spec=problem_spec,
                X=X,
                y=y,
                n_splits=n_splits,
                random_seed=random_seed,
                primary_metric=metric,
            )

            score = cv_res.primary_mean
            duration = time.time() - t0

            record = TrialRecord(
                trial_number=trial.number + 1,
                score=score,
                params=suggested_params,
                duration_sec=duration,
            )
            trials_history.append(record)

            if trial_callback:
                trial_callback(record.trial_number, record.score, record.params)

            return score

        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        total_time = time.time() - t_start

        best_score = float(study.best_value)
        best_params = study.best_params
        delta = (best_score - baseline_score) if direction == "maximize" else (baseline_score - best_score)

        # Fit final pipeline with best hyperparameters on 100% of data
        best_estimator = candidate.estimator_factory(random_seed)
        best_estimator.set_params(**best_params)
        final_pipeline = Pipeline(steps=[
            ("preprocessor", clone(preprocessor_builder.transformer)),
            ("model", best_estimator),
        ])
        # Handle label encoding for final pipeline
        if problem_spec.is_classification:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_fit = le.fit_transform(y.values)
        else:
            y_fit = y.values
        final_pipeline.fit(X, y_fit)

        return TuningResult(
            model_id=candidate.model_id,
            model_name=candidate.display_name,
            metric_name=metric,
            direction=direction,
            baseline_score=baseline_score,
            best_score=best_score,
            score_delta=delta,
            best_params=best_params,
            trials_history=trials_history,
            total_trials=len(trials_history),
            duration_sec=total_time,
            optimized_pipeline=final_pipeline,
        )
