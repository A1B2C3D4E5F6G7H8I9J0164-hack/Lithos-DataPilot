"""
Automated tabular dataset profiler computing structural, statistical, and column-level metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Statistical summary for a single feature column."""
    name: str
    dtype: str
    inferred_type: str  # 'numerical', 'categorical', 'boolean', 'datetime', 'id'
    null_count: int
    null_percentage: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_high_cardinality: bool
    sample_values: List[Any]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Comprehensive dataset-level profiling report."""
    row_count: int
    column_count: int
    memory_usage_mb: float
    total_missing_cells: int
    missing_cells_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    numerical_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    datetime_columns: List[str]
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    column_profiles: Dict[str, ColumnProfile]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Rows": self.row_count,
            "Columns": self.column_count,
            "Memory (MB)": round(self.memory_usage_mb, 2),
            "Missing Cells": f"{self.total_missing_cells} ({self.missing_cells_percentage:.1f}%)",
            "Duplicate Rows": f"{self.duplicate_rows_count} ({self.duplicate_rows_percentage:.1f}%)",
            "Numerical Features": len(self.numerical_columns),
            "Categorical Features": len(self.categorical_columns),
            "Constant Features": len(self.constant_columns),
            "High Cardinality Features": len(self.high_cardinality_columns),
        }


class DataProfiler:
    """Generates rigorous statistical and structural profiles of tabular datasets."""

    HIGH_CARDINALITY_THRESHOLD: int = 50
    ID_UNIQUE_RATIO_THRESHOLD: float = 0.95

    @classmethod
    def profile(cls, df: pd.DataFrame) -> DatasetProfile:
        """
        Profiles a pandas DataFrame.

        Args:
            df: Input tabular DataFrame.

        Returns:
            DatasetProfile detailing dataset statistics.
        """
        n_rows, n_cols = df.shape
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = float(mem_bytes / (1024 * 1024))

        null_cells = int(df.isna().sum().sum())
        null_pct = float((null_cells / (n_rows * n_cols) * 100) if (n_rows * n_cols) > 0 else 0.0)

        dup_rows = int(df.duplicated().sum())
        dup_pct = float((dup_rows / n_rows * 100) if n_rows > 0 else 0.0)

        numerical_cols = []
        categorical_cols = []
        boolean_cols = []
        datetime_cols = []
        constant_cols = []
        high_card_cols = []
        col_profiles: Dict[str, ColumnProfile] = {}

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_ratio = float(null_count / n_rows) if n_rows > 0 else 0.0
            non_null_series = series.dropna()
            unique_count = int(non_null_series.nunique())
            unique_ratio = float(unique_count / n_rows) if n_rows > 0 else 0.0

            is_constant = (unique_count <= 1)
            if is_constant:
                constant_cols.append(col)

            # Inferred type
            if pd.api.types.is_bool_dtype(series):
                inferred = "boolean"
                boolean_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred = "datetime"
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                # Check if integer column acts as boolean
                if unique_count == 2 and set(non_null_series.unique()).issubset({0, 1}):
                    inferred = "boolean"
                    boolean_cols.append(col)
                else:
                    inferred = "numerical"
                    numerical_cols.append(col)
            else:
                # String / Object / Categorical
                # Check if this looks like an ID column
                col_lower = str(col).lower()
                is_id_name = (
                    col_lower == "id"
                    or col_lower.endswith("_id")
                    or col_lower.startswith("id_")
                    or "identifier" in col_lower
                )
                if (unique_ratio >= 0.95 and is_id_name) or (unique_ratio > cls.ID_UNIQUE_RATIO_THRESHOLD and n_rows > 30):
                    inferred = "id"
                else:
                    inferred = "categorical"
                categorical_cols.append(col)

            is_high_card = (
                inferred in ("categorical", "id")
                and unique_count > cls.HIGH_CARDINALITY_THRESHOLD
            )
            if is_high_card:
                high_card_cols.append(col)

            # Column stats
            stats: Dict[str, Any] = {}
            if inferred == "numerical" and len(non_null_series) > 0:
                stats["min"] = float(non_null_series.min())
                stats["max"] = float(non_null_series.max())
                stats["mean"] = float(non_null_series.mean())
                stats["std"] = float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
                stats["median"] = float(non_null_series.median())
                stats["q25"] = float(non_null_series.quantile(0.25))
                stats["q75"] = float(non_null_series.quantile(0.75))
                stats["skewness"] = float(non_null_series.skew()) if len(non_null_series) > 2 else 0.0
            elif inferred in ("categorical", "boolean", "id") and len(non_null_series) > 0:
                value_counts = non_null_series.value_counts()
                stats["top_category"] = str(value_counts.index[0])
                stats["top_frequency"] = int(value_counts.iloc[0])
                stats["top_percentage"] = float((value_counts.iloc[0] / len(non_null_series)) * 100)

            # Sample values for preview
            sample_vals = [
                str(v) if not pd.isna(v) else "NaN"
                for v in series.head(5).tolist()
            ]

            col_profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred,
                null_count=null_count,
                null_percentage=round(null_ratio * 100, 2),
                unique_count=unique_count,
                unique_ratio=round(unique_ratio, 4),
                is_constant=is_constant,
                is_high_cardinality=is_high_card,
                sample_values=sample_vals,
                stats=stats,
            )

        return DatasetProfile(
            row_count=n_rows,
            column_count=n_cols,
            memory_usage_mb=mem_mb,
            total_missing_cells=null_cells,
            missing_cells_percentage=null_pct,
            duplicate_rows_count=dup_rows,
            duplicate_rows_percentage=dup_pct,
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            boolean_columns=boolean_cols,
            datetime_columns=datetime_cols,
            constant_columns=constant_cols,
            high_cardinality_columns=high_card_cols,
            column_profiles=col_profiles,
        )
