import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from app.data.profiler import DataProfiler
from app.data.preprocessing import LeakageSafePreprocessor
from app.ml.problem_detector import ProblemDetector
from app.ml.explainability import ModelExplainer


def test_shap_explainability_generation():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat_num": np.random.normal(0, 1, n),
        "feat_cat": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n)
    })

    problem_spec = ProblemDetector.detect(df, "target")
    profile = DataProfiler.profile(df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="target")

    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor.transformer),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)

    exp_res = ModelExplainer.explain(
        pipeline=pipeline,
        problem_spec=problem_spec,
        X_sample=X.head(10),
        model_id="random_forest",
        n_background_samples=20,
        n_explain_samples=5
    )

    assert exp_res.global_importance is not None
    assert len(exp_res.global_importance) > 0
    assert len(exp_res.sample_explanations) > 0
    # Check that top features are ranked
    assert len(exp_res.top_features) > 0
