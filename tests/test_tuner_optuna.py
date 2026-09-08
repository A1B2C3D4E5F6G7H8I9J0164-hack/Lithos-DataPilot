import pandas as pd
import numpy as np
from app.data.profiler import DataProfiler
from app.data.preprocessing import LeakageSafePreprocessor
from app.ml.problem_detector import ProblemDetector
from app.ml.models import ModelFactory
from app.ml.tuner import HyperparameterTuner


def test_optuna_tuning_loop():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feat1": np.random.normal(0, 1, n),
        "feat2": np.random.uniform(0, 10, n),
        "target": np.random.choice([0, 1], n)
    })

    problem_spec = ProblemDetector.detect(df, "target")
    profile = DataProfiler.profile(df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="target")

    candidates = ModelFactory.get_candidates(problem_spec.problem_type, random_seed=42)
    cand = candidates[0]  # Logistic regression

    X = df.drop(columns=["target"])
    y = df["target"]

    tuning_res = HyperparameterTuner.tune(
        candidate=cand,
        baseline_score=0.5,
        preprocessor_builder=preprocessor,
        problem_spec=problem_spec,
        X=X,
        y=y,
        n_trials=4,
        n_splits=3,
        random_seed=42
    )

    assert tuning_res.total_trials == 4
    assert len(tuning_res.trials_history) == 4
    assert tuning_res.optimized_pipeline is not None
    assert isinstance(tuning_res.best_params, dict)
