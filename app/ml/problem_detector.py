"""
Automated problem type detector determining task category, class imbalance, and CV strategy.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class ProblemSpec:
    """Rigorous specification of the detected ML problem and modeling strategy."""
    problem_type: ProblemType
    target_column: str
    target_dtype: str
    unique_values_count: int
    classes: Optional[List[Any]] = None
    class_distribution: Optional[Dict[str, float]] = None  # Class -> percentage
    is_imbalanced: bool = False
    imbalance_ratio: float = 1.0
    recommended_primary_metric: str = "f1"
    available_metrics: List[str] = field(default_factory=list)
    cv_strategy: str = "StratifiedKFold"
    reasoning: List[str] = field(default_factory=list)

    @property
    def is_classification(self) -> bool:
        return self.problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        )

    @property
    def is_regression(self) -> bool:
        return self.problem_type == ProblemType.REGRESSION


class ProblemDetector:
    """Infers machine learning problem taxonomy from target data characteristics."""

    MAX_CLASSIFICATION_UNIQUE_VALUES: int = 20
    IMBALANCE_RATIO_THRESHOLD: float = 2.0

    @classmethod
    def detect(cls, df: pd.DataFrame, target_column: str) -> ProblemSpec:
        """
        Analyzes target column properties to infer ML task type.

        Args:
            df: Input dataset.
            target_column: Target column name.

        Returns:
            ProblemSpec outlining task type, metric recommendations, and validation strategy.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

        series = df[target_column].dropna()
        n_samples = len(series)
        unique_count = int(series.nunique())
        unique_ratio = unique_count / n_samples if n_samples > 0 else 0.0
        dtype_str = str(series.dtype)

        reasoning: List[str] = []

        # Check for boolean or string/categorical types
        is_string_or_cat = (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        is_boolean = pd.api.types.is_bool_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_boolean or (is_numeric and unique_count == 2):
            problem_type = ProblemType.BINARY_CLASSIFICATION
            reasoning.append(f"Target contains exactly 2 unique values ({list(series.unique())[:2]}).")
        elif is_string_or_cat:
            if unique_count == 2:
                problem_type = ProblemType.BINARY_CLASSIFICATION
                reasoning.append("Target is non-numeric with exactly 2 distinct classes.")
            elif unique_count <= cls.MAX_CLASSIFICATION_UNIQUE_VALUES:
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
                reasoning.append(f"Target is non-numeric with {unique_count} distinct categorical levels.")
            else:
                # String with many unique values - potential high-cardinality ID or text
                raise ValueError(
                    f"Target column '{target_column}' is text with {unique_count} distinct values. "
                    "Cannot be directly modeled as tabular classification target without parsing."
                )
        elif is_numeric:
            # Numeric target: check if continuous vs discrete few levels
            if unique_count <= cls.MAX_CLASSIFICATION_UNIQUE_VALUES and unique_ratio < 0.10:
                if unique_count == 2:
                    problem_type = ProblemType.BINARY_CLASSIFICATION
                    reasoning.append(f"Discrete numeric target with 2 classes ({list(series.unique())}).")
                else:
                    problem_type = ProblemType.MULTICLASS_CLASSIFICATION
                    reasoning.append(f"Discrete integer-like target with {unique_count} classes.")
            else:
                problem_type = ProblemType.REGRESSION
                reasoning.append(
                    f"Continuous numeric target with {unique_count} unique values (spread across {n_samples} samples)."
                )
        else:
            problem_type = ProblemType.REGRESSION
            reasoning.append("Fallback to regression for continuous attributes.")

        # Class distribution & imbalance for classification
        classes = None
        class_dist = None
        is_imbalanced = False
        imbalance_ratio = 1.0

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            classes = [str(c) for c in series.unique()]
            val_counts = series.value_counts(normalize=True)
            class_dist = {str(k): round(float(v * 100), 2) for k, v in val_counts.items()}
            
            maj_prop = float(val_counts.iloc[0])
            min_prop = float(val_counts.iloc[-1])
            imbalance_ratio = round(maj_prop / max(1e-5, min_prop), 2)
            
            if imbalance_ratio >= cls.IMBALANCE_RATIO_THRESHOLD:
                is_imbalanced = True
                reasoning.append(
                    f"Detected class imbalance (Ratio {imbalance_ratio}:1). Prioritizing F1/ROC-AUC over raw accuracy."
                )
            
            cv_strategy = "StratifiedKFold"
            reasoning.append("Using Stratified K-Fold cross-validation to preserve class proportions across folds.")
            
            if problem_type == ProblemType.BINARY_CLASSIFICATION:
                available_metrics = ["f1", "roc_auc", "accuracy", "precision", "recall"]
                recommended_metric = "f1" if is_imbalanced else "roc_auc"
            else:
                available_metrics = ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted"]
                recommended_metric = "f1_weighted"
        else:
            cv_strategy = "KFold"
            available_metrics = ["rmse", "mae", "r2"]
            recommended_metric = "rmse"
            reasoning.append("Using standard K-Fold cross-validation for continuous target evaluation.")

        return ProblemSpec(
            problem_type=problem_type,
            target_column=target_column,
            target_dtype=dtype_str,
            unique_values_count=unique_count,
            classes=classes,
            class_distribution=class_dist,
            is_imbalanced=is_imbalanced,
            imbalance_ratio=imbalance_ratio,
            recommended_primary_metric=recommended_metric,
            available_metrics=available_metrics,
            cv_strategy=cv_strategy,
            reasoning=reasoning,
        )
