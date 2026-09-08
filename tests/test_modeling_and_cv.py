import pandas as pd
import numpy as np
from app.data.profiler import DataProfiler
from app.data.preprocessing import LeakageSafePreprocessor
from app.ml.problem_detector import ProblemDetector, ProblemType
from app.ml.models import ModelFactory
from app.ml.cross_validation import CrossValidationRunner
from app.ml.trainer import ModelTrainer


def test_cross_validation_and_benchmarking_classification():
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "feat_num1": np.random.normal(0, 1, n),
        "feat_num2": np.random.exponential(1.5, n),
        "feat_cat": np.random.choice(["typeA", "typeB", "typeC"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })

    problem_spec = ProblemDetector.detect(df, "target")
    profile = DataProfiler.profile(df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="target")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Benchmark all candidates
    benchmark_res = ModelTrainer.benchmark(
        problem_spec=problem_spec,
        preprocessor_builder=preprocessor,
        X=X,
        y=y,
        n_splits=3,
        random_seed=42
    )

    assert benchmark_res.best_candidate_id in benchmark_res.results
    assert len(benchmark_res.leaderboard_df) >= 3
    assert not benchmark_res.leaderboard_df.empty
    assert benchmark_res.best_score > 0.0


def test_regression_cross_validation():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3500, n),
        "rooms": np.random.randint(1, 6, n),
        "price": np.random.uniform(100000, 800000, n)
    })

    problem_spec = ProblemDetector.detect(df, "price")
    assert problem_spec.is_regression

    profile = DataProfiler.profile(df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="price")

    X = df.drop(columns=["price"])
    y = df["price"]

    benchmark_res = ModelTrainer.benchmark(
        problem_spec=problem_spec,
        preprocessor_builder=preprocessor,
        X=X,
        y=y,
        n_splits=3,
        random_seed=42
    )

    assert benchmark_res.best_candidate_id in benchmark_res.results
    assert benchmark_res.primary_metric == "rmse"
