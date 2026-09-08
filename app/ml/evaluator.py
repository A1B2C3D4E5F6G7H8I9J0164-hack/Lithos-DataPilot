"""
Rigorous evaluation metric calculator for classification and regression models.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)

from .problem_detector import ProblemType


@dataclass
class EvaluationMetrics:
    """Standardized metrics container with JSON serializable attributes."""
    problem_type: ProblemType
    primary_metric_name: str
    primary_metric_value: float
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    classes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type.value,
            "primary_metric": self.primary_metric_name,
            "primary_value": round(self.primary_metric_value, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "confusion_matrix": self.confusion_matrix,
            "classes": self.classes,
        }


class ModelEvaluator:
    """Computes transparent, un-fabricated ML evaluation metrics."""

    @classmethod
    def evaluate(
        cls,
        problem_type: ProblemType,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        primary_metric: Optional[str] = None,
        classes: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluates predictions against ground truth labels.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            return cls._evaluate_classification(
                problem_type, y_true, y_pred, y_prob, primary_metric, classes
            )
        else:
            return cls._evaluate_regression(
                y_true, y_pred, primary_metric
            )

    @classmethod
    def _evaluate_classification(
        cls,
        problem_type: ProblemType,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        primary_metric: Optional[str],
        classes: Optional[List[str]],
    ) -> EvaluationMetrics:
        is_binary = (problem_type == ProblemType.BINARY_CLASSIFICATION)
        average_mode = "binary" if is_binary else "weighted"

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average=average_mode, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=average_mode, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=average_mode, zero_division=0))

        metrics_dict: Dict[str, float] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        # ROC-AUC & Log-Loss calculation
        if y_prob is not None:
            try:
                if is_binary:
                    # y_prob can be 1D (prob of positive class) or 2D
                    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                    auc = float(roc_auc_score(y_true, prob_pos))
                    ll = float(log_loss(y_true, y_prob))
                else:
                    auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                    ll = float(log_loss(y_true, y_prob))
                metrics_dict["roc_auc"] = auc
                metrics_dict["log_loss"] = ll
            except Exception:
                # E.g. only 1 class in small split
                metrics_dict["roc_auc"] = 0.5
                metrics_dict["log_loss"] = 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_list = cm.tolist()

        if primary_metric is None or primary_metric not in metrics_dict:
            primary_metric = "f1" if is_binary else "f1"

        return EvaluationMetrics(
            problem_type=problem_type,
            primary_metric_name=primary_metric,
            primary_metric_value=metrics_dict.get(primary_metric, f1),
            metrics=metrics_dict,
            confusion_matrix=cm_list,
            classes=[str(c) for c in classes] if classes is not None else None,
        )

    @classmethod
    def _evaluate_regression(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        primary_metric: Optional[str],
    ) -> EvaluationMetrics:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        exp_var = float(explained_variance_score(y_true, y_pred))

        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mape = 0.0

        metrics_dict: Dict[str, float] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": exp_var,
            "mape": mape,
        }

        if primary_metric is None or primary_metric not in metrics_dict:
            primary_metric = "rmse"

        return EvaluationMetrics(
            problem_type=ProblemType.REGRESSION,
            primary_metric_name=primary_metric,
            primary_metric_value=metrics_dict[primary_metric],
            metrics=metrics_dict,
        )
