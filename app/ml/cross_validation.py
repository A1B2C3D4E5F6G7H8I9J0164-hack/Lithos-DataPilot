"""
Leakage-safe cross-validation runner evaluating full pipelines across data splits.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from .problem_detector import ProblemSpec, ProblemType
from .models import ModelCandidate
from .evaluator import ModelEvaluator, EvaluationMetrics
from app.data.preprocessing import LeakageSafePreprocessor


@dataclass
class CVFoldResult:
    """Individual fold performance record."""
    fold_index: int
    train_size: int
    val_size: int
    metrics: Dict[str, float]
    fit_time_sec: float


@dataclass
class ModelCVResult:
    """Consolidated cross-validation results for a model candidate."""
    model_id: str
    display_name: str
    fold_results: List[CVFoldResult]
    cv_mean: Dict[str, float]
    cv_std: Dict[str, float]
    primary_metric: str
    primary_mean: float
    primary_std: float
    total_time_sec: float
    fitted_pipeline: Optional[Pipeline] = None  # Full pipeline fitted on 100% of training data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "primary_metric": self.primary_metric,
            "primary_mean": round(self.primary_mean, 4),
            "primary_std": round(self.primary_std, 4),
            "total_time_sec": round(self.total_time_sec, 2),
            "cv_mean": {k: round(v, 4) for k, v in self.cv_mean.items()},
            "cv_std": {k: round(v, 4) for k, v in self.cv_std.items()},
        }


class CrossValidationRunner:
    """Orchestrates leakage-safe cross-validation for arbitrary candidate models."""

    @classmethod
    def run_cv(
        cls,
        candidate: ModelCandidate,
        preprocessor_builder: LeakageSafePreprocessor,
        problem_spec: ProblemSpec,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_seed: int = 42,
        primary_metric: Optional[str] = None,
    ) -> ModelCVResult:
        """
        Executes K-Fold or Stratified K-Fold cross-validation on full Pipelines.
        """
        t_start = time.time()
        metric_name = primary_metric or problem_spec.recommended_primary_metric

        # Target label encoding if classification
        y_vals = y.values
        label_encoder = None
        if problem_spec.is_classification:
            label_encoder = LabelEncoder()
            y_vals = label_encoder.fit_transform(y_vals)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        fold_records: List[CVFoldResult] = []
        all_metrics_keys: List[str] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_vals)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_vals[train_idx], y_vals[val_idx]

            fold_t0 = time.time()

            # Fresh instance of preprocessor transformer and estimator for this fold
            transformer = clone(preprocessor_builder.transformer)
            estimator = candidate.estimator_factory(random_seed + fold_idx)

            fold_pipeline = Pipeline(steps=[
                ("preprocessor", transformer),
                ("model", estimator),
            ])

            # FIT on training fold ONLY
            fold_pipeline.fit(X_tr, y_tr)
            fit_duration = time.time() - fold_t0

            # PREDICT on validation fold
            y_pred = fold_pipeline.predict(X_val)
            y_prob = None
            if problem_spec.is_classification and hasattr(fold_pipeline, "predict_proba"):
                try:
                    y_prob = fold_pipeline.predict_proba(X_val)
                except Exception:
                    y_prob = None

            eval_res = ModelEvaluator.evaluate(
                problem_type=problem_spec.problem_type,
                y_true=y_val,
                y_pred=y_pred,
                y_prob=y_prob,
                primary_metric=metric_name,
                classes=problem_spec.classes,
            )

            if not all_metrics_keys:
                all_metrics_keys = list(eval_res.metrics.keys())

            fold_records.append(CVFoldResult(
                fold_index=fold_idx + 1,
                train_size=len(train_idx),
                val_size=len(val_idx),
                metrics=eval_res.metrics,
                fit_time_sec=fit_duration,
            ))

        # Compute aggregate CV stats
        cv_mean = {}
        cv_std = {}
        for m_key in all_metrics_keys:
            vals = [f.metrics.get(m_key, 0.0) for f in fold_records]
            cv_mean[m_key] = float(np.mean(vals))
            cv_std[m_key] = float(np.std(vals))

        total_time = time.time() - t_start
        prim_mean = cv_mean.get(metric_name, 0.0)
        prim_std = cv_std.get(metric_name, 0.0)

        # Fit final full pipeline on 100% of data for serving/explanation
        final_transformer = clone(preprocessor_builder.transformer)
        final_estimator = candidate.estimator_factory(random_seed)
        final_pipeline = Pipeline(steps=[
            ("preprocessor", final_transformer),
            ("model", final_estimator),
        ])
        final_pipeline.fit(X, y_vals)

        return ModelCVResult(
            model_id=candidate.model_id,
            display_name=candidate.display_name,
            fold_results=fold_records,
            cv_mean=cv_mean,
            cv_std=cv_std,
            primary_metric=metric_name,
            primary_mean=prim_mean,
            primary_std=prim_std,
            total_time_sec=total_time,
            fitted_pipeline=final_pipeline,
        )
