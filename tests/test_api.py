import os
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from app.api.main import app
from app.data.profiler import DataProfiler
from app.data.preprocessing import LeakageSafePreprocessor
from app.ml.problem_detector import ProblemDetector, ProblemType

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_api_predict_flow(tmp_path):
    # Train a minimal pipeline and persist to artifacts
    np.random.seed(42)
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "city": ["NY", "SF", "NY", "SF", "NY"],
        "target": [0, 1, 0, 1, 0]
    })
    problem_spec = ProblemDetector.detect(df, "target")
    profile = DataProfiler.profile(df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="target")

    X = df.drop(columns=["target"])
    y = df["target"]

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor.transformer),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipe.fit(X, y)

    os.makedirs("artifacts", exist_ok=True)
    bundle_path = "artifacts/model_bundle.joblib"
    bundle = {
        "pipeline": pipe,
        "problem_spec": problem_spec,
        "target_column": "target",
        "feature_schema": [
            {"name": "age", "dtype": "int64", "sample_value": 30, "is_numerical": True},
            {"name": "city", "dtype": "object", "sample_value": "NY", "is_numerical": False, "unique_values": ["NY", "SF"]}
        ],
        "numerical_features": ["age"],
        "categorical_features": ["city"],
        "drop_features": [],
        "classes": ["No", "Yes"],
        "best_model_name": "Random Forest Test",
        "best_model_id": "random_forest",
        "primary_metric": "f1",
        "best_score": 0.95,
        "baseline_score": 0.80,
        "best_params": {"n_estimators": 10},
        "created_at": "2026-09-07T00:00:00",
    }
    joblib.dump(bundle, bundle_path)

    # Test GET /model
    res_m = client.get("/model")
    assert res_m.status_code == 200
    assert res_m.json()["best_model_name"] == "Random Forest Test"

    # Test GET /metrics
    res_met = client.get("/metrics")
    assert res_met.status_code == 200
    assert res_met.json()["best_score"] == 0.95

    # Test POST /predict single
    res_p = client.post("/predict", json={"features": {"age": 35, "city": "NY"}})
    assert res_p.status_code == 200
    p_data = res_p.json()
    assert "prediction" in p_data
    assert p_data["status"] == "success"

    # Test POST /predict/batch
    res_b = client.post("/predict/batch", json={"records": [{"age": 22, "city": "SF"}, {"age": 45, "city": "NY"}]})
    assert res_b.status_code == 200
    b_data = res_b.json()
    assert b_data["count"] == 2
    assert len(b_data["predictions"]) == 2
