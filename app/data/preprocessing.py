"""
Leakage-safe tabular preprocessor using scikit-learn ColumnTransformer and Pipeline.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .profiler import DatasetProfile


class LeakageSafePreprocessor:
    """
    Constructs scikit-learn ColumnTransformer architectures ensuring zero data leakage.
    Transformations are only fit on training splits/folds.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        drop_features: Optional[List[str]] = None,
        max_one_hot_cardinality: int = 25
    ):
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        self.drop_features = list(drop_features or [])
        self.max_one_hot_cardinality = max_one_hot_cardinality

        # Split categoricals into one-hot vs ordinal/high-cardinality
        self.ohe_features: List[str] = []
        self.high_card_features: List[str] = []
        
        # We will configure these during setup or build
        self.transformer: Optional[ColumnTransformer] = None
        self._feature_names_out: Optional[List[str]] = None

    @classmethod
    def from_profile(
        cls,
        profile: DatasetProfile,
        target_column: str,
        excluded_columns: Optional[List[str]] = None
    ) -> "LeakageSafePreprocessor":
        """
        Factory to instantiate a preprocessor directly from DatasetProfile findings.
        """
        excluded = set(excluded_columns or [])
        excluded.add(target_column)

        # Drop constant features and detected ID columns
        to_drop = list(profile.constant_columns)
        for col_name, col_prof in profile.column_profiles.items():
            if col_prof.inferred_type == "id" or col_prof.is_constant:
                if col_name not in to_drop:
                    to_drop.append(col_name)

        numerical = [c for c in profile.numerical_columns if c not in excluded and c not in to_drop]
        categorical = [
            c for c in (profile.categorical_columns + profile.boolean_columns)
            if c not in excluded and c not in to_drop
        ]

        preprocessor = cls(
            numerical_features=numerical,
            categorical_features=categorical,
            drop_features=to_drop,
        )
        preprocessor.build_transformer(profile)
        return preprocessor

    def build_transformer(self, profile: Optional[DatasetProfile] = None) -> ColumnTransformer:
        """
        Builds the scikit-learn ColumnTransformer instance.
        """
        transformers = []

        # Numerical pipeline: Median Imputation + Standard Scaling
        if self.numerical_features:
            num_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.numerical_features))

        # Categorical partitioning
        if profile is not None:
            self.ohe_features = [
                c for c in self.categorical_features
                if c in profile.column_profiles and profile.column_profiles[c].unique_count <= self.max_one_hot_cardinality
            ]
            self.high_card_features = [
                c for c in self.categorical_features if c not in self.ohe_features
            ]
        else:
            self.ohe_features = list(self.categorical_features)
            self.high_card_features = []

        if self.ohe_features:
            cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat_ohe", cat_pipe, self.ohe_features))

        if self.high_card_features:
            ord_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("cat_ord", ord_pipe, self.high_card_features))

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True
        )
        return self.transformer

    def get_feature_names_out(self) -> List[str]:
        """Returns readable transformed feature names."""
        if self.transformer is None:
            return []
        try:
            raw_names = self.transformer.get_feature_names_out()
            cleaned = []
            for name in raw_names:
                # e.g. "num__tenure" -> "tenure", "cat_ohe__internet_Fiber" -> "internet_Fiber"
                parts = name.split("__", 1)
                cleaned.append(parts[1] if len(parts) > 1 else name)
            return cleaned
        except Exception:
            # Fallback
            names = list(self.numerical_features)
            names.extend(self.ohe_features)
            names.extend(self.high_card_features)
            return names
