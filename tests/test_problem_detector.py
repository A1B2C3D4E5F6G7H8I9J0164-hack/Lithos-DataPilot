import pandas as pd
import numpy as np
from app.ml.problem_detector import ProblemDetector, ProblemType


def test_detect_binary_classification():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": ["churn", "stay"] * 50
    })
    spec = ProblemDetector.detect(df, "target")
    assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert spec.is_classification
    assert not spec.is_regression
    assert spec.cv_strategy == "StratifiedKFold"


def test_detect_imbalanced_classification():
    # 85% class 0, 15% class 1
    labels = [0] * 85 + [1] * 15
    df = pd.DataFrame({
        "feature1": range(100),
        "target": labels
    })
    spec = ProblemDetector.detect(df, "target")
    assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert spec.is_imbalanced
    assert spec.imbalance_ratio >= 2.0
    assert spec.recommended_primary_metric in ("f1", "roc_auc")


def test_detect_regression():
    # Continuous target with 100 distinct values
    df = pd.DataFrame({
        "feature1": range(100),
        "target": np.random.uniform(10.0, 100.0, size=100)
    })
    spec = ProblemDetector.detect(df, "target")
    assert spec.problem_type == ProblemType.REGRESSION
    assert spec.is_regression
    assert spec.recommended_primary_metric == "rmse"
    assert spec.cv_strategy == "KFold"
