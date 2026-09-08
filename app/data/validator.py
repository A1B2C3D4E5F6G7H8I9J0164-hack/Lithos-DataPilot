"""
Dataset schema and feasibility validator.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


@dataclass
class ValidationResult:
    """Represents data validation status and messages."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    target_column: Optional[str] = None
    target_unique_count: Optional[int] = None
    target_null_count: Optional[int] = None


class DataValidator:
    """Validates structural correctness of tabular datasets prior to ML workflows."""

    MIN_ROWS: int = 25
    MIN_COLUMNS: int = 2

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate dataset dimensions, target presence, and viability.

        Args:
            df: Input dataset DataFrame.
            target_column: Optional target feature name.

        Returns:
            ValidationResult with validation decisions.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or not isinstance(df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                errors=["Input is not a valid pandas DataFrame."]
            )

        n_rows, n_cols = df.shape

        if n_rows == 0:
            errors.append("Dataset has 0 rows.")
        elif n_rows < cls.MIN_ROWS:
            errors.append(
                f"Dataset contains only {n_rows} rows. Minimum {cls.MIN_ROWS} rows are required for cross-validation."
            )

        if n_cols < cls.MIN_COLUMNS:
            errors.append(
                f"Dataset contains {n_cols} columns. At least {cls.MIN_COLUMNS} columns (1 feature + 1 target) are required."
            )

        # Check duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicated_names = df.columns[df.columns.duplicated()].tolist()
            errors.append(f"Duplicate column names detected: {duplicated_names}")

        # Check all columns completely null
        all_null_cols = [c for c in df.columns if df[c].isna().all()]
        if all_null_cols:
            warnings.append(f"Columns with 100% missing values detected: {all_null_cols}")

        target_unique_count = None
        target_null_count = None

        if target_column is not None:
            if target_column not in df.columns:
                errors.append(f"Specified target column '{target_column}' is not found in dataset columns.")
            else:
                target_series = df[target_column]
                target_null_count = int(target_series.isna().sum())
                target_unique_count = int(target_series.dropna().nunique())

                if target_null_count == len(target_series):
                    errors.append(f"Target column '{target_column}' is entirely empty / NaN.")
                elif target_null_count > 0:
                    warnings.append(
                        f"Target column '{target_column}' has {target_null_count} missing values ({target_null_count / len(target_series):.1%}). Rows with missing targets will be dropped."
                    )

                if target_unique_count < 2:
                    errors.append(
                        f"Target column '{target_column}' must have at least 2 distinct values, but found {target_unique_count}."
                    )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=n_rows,
            column_count=n_cols,
            target_column=target_column,
            target_unique_count=target_unique_count,
            target_null_count=target_null_count,
        )
