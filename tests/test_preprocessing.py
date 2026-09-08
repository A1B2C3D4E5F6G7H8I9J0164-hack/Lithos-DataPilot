import pandas as pd
import numpy as np
from app.data.profiler import DataProfiler
from app.data.preprocessing import LeakageSafePreprocessor


def test_leakage_safe_preprocessor_fitting():
    train_df = pd.DataFrame({
        "num1": [10.0, 20.0, np.nan, 40.0],
        "cat1": ["A", "B", "A", np.nan],
        "id_col": [f"ID_{i}" for i in range(4)],
        "target": [0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "num1": [np.nan, 25.0],
        "cat1": ["B", "Unknown_Cat"],
        "id_col": ["ID_4", "ID_5"],
        "target": [1, 0]
    })

    profile = DataProfiler.profile(train_df)
    preprocessor = LeakageSafePreprocessor.from_profile(profile, target_column="target")

    transformer = preprocessor.transformer
    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    # Fit transformer strictly on X_train
    transformer.fit(X_train)

    # Transform both
    X_train_trans = transformer.transform(X_train)
    X_test_trans = transformer.transform(X_test)

    # Asserts
    assert X_train_trans.shape[0] == 4
    assert X_test_trans.shape[0] == 2
    # Ensure ID column was excluded
    assert "id_col" in preprocessor.drop_features
    # Check transformed feature names
    names = preprocessor.get_feature_names_out()
    assert len(names) == X_train_trans.shape[1]
