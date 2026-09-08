"""
Automated Exploratory Data Analysis (EDA) computations and factual statistical insight generation.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from .profiler import DatasetProfile


@dataclass
class StatisticalInsight:
    """Factual, data-driven observation from the dataset."""
    category: str  # 'Target', 'Correlation', 'Distribution', 'Missingness'
    title: str
    description: str
    importance: str  # 'high', 'medium', 'info'


class EDAEngine:
    """Computes distributions, correlation matrices, and factual analytical insights."""

    @classmethod
    def get_correlation_matrix(
        cls,
        df: pd.DataFrame,
        numerical_columns: List[str]
    ) -> pd.DataFrame:
        """Computes Pearson correlation for numerical columns."""
        if not numerical_columns or len(numerical_columns) < 2:
            return pd.DataFrame()
        valid_cols = [c for c in numerical_columns if c in df.columns]
        return df[valid_cols].corr().round(3)

    @classmethod
    def get_top_correlations(
        cls,
        corr_matrix: pd.DataFrame,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Returns top pairs of correlated features."""
        if corr_matrix.empty:
            return []
        
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((c1, c2, float(val)))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:top_k]

    @classmethod
    def generate_factual_insights(
        cls,
        df: pd.DataFrame,
        profile: DatasetProfile,
        target_column: Optional[str] = None
    ) -> List[StatisticalInsight]:
        """Generates strictly verified factual insights from empirical data."""
        insights: List[StatisticalInsight] = []

        # 1. Target column insights
        if target_column and target_column in df.columns:
            target_series = df[target_column].dropna()
            u_count = target_series.nunique()
            
            if u_count == 2:
                vc = target_series.value_counts(normalize=True)
                majority_pct = float(vc.iloc[0] * 100)
                minority_pct = float(vc.iloc[1] * 100)
                ratio = majority_pct / max(0.1, minority_pct)
                imbalance_str = "imbalanced" if ratio >= 2.0 else "well-balanced"
                importance = "high" if ratio >= 2.0 else "info"
                
                insights.append(StatisticalInsight(
                    category="Target",
                    title="Binary Target Distribution",
                    description=(
                        f"Target '{target_column}' is {imbalance_str} with {majority_pct:.1f}% "
                        f"class '{vc.index[0]}' vs {minority_pct:.1f}% class '{vc.index[1]}' (Ratio: {ratio:.1f}:1)."
                    ),
                    importance=importance
                ))
            elif pd.api.types.is_numeric_dtype(target_series) and u_count > 10:
                skew = float(target_series.skew())
                skew_desc = "right-skewed" if skew > 1.0 else ("left-skewed" if skew < -1.0 else "symmetric")
                insights.append(StatisticalInsight(
                    category="Target",
                    title="Continuous Target Profile",
                    description=(
                        f"Target '{target_column}' has a mean of {target_series.mean():,.2f} "
                        f"(median: {target_series.median():,.2f}, std: {target_series.std():,.2f}) with {skew_desc} distribution (skew={skew:.2f})."
                    ),
                    importance="high"
                ))

        # 2. Strong Correlations
        num_cols = [c for c in profile.numerical_columns if c in df.columns]
        if len(num_cols) >= 2:
            corr_mat = cls.get_correlation_matrix(df, num_cols)
            top_pairs = cls.get_top_correlations(corr_mat, top_k=3)
            for c1, c2, corr_val in top_pairs:
                if abs(corr_val) >= 0.70:
                    insights.append(StatisticalInsight(
                        category="Correlation",
                        title="Strong Multicollinearity",
                        description=f"Strong linear relationship detected between '{c1}' and '{c2}' (Pearson r = {corr_val:+.2f}).",
                        importance="medium"
                    ))

        # 3. Skewed Features
        skewed_cols = []
        for c in num_cols:
            if c != target_column:
                s = df[c].dropna()
                if len(s) > 10:
                    sk = float(s.skew())
                    if abs(sk) > 2.0:
                        skewed_cols.append((c, sk))

        if skewed_cols:
            top_skew = sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)[:2]
            names_str = ", ".join([f"'{c}' (skew={sk:.2f})" for c, sk in top_skew])
            insights.append(StatisticalInsight(
                category="Distribution",
                title="Substantial Feature Skewness",
                description=f"Highly skewed distributions observed in {names_str}. Robust scaling or log transformation recommended.",
                importance="medium"
            ))

        # 4. Missingness
        if profile.total_missing_cells > 0:
            top_missing_cols = [
                (col, prof.null_percentage)
                for col, prof in profile.column_profiles.items()
                if prof.null_percentage > 0
            ]
            top_missing_cols.sort(key=lambda x: x[1], reverse=True)
            if top_missing_cols:
                top_m_str = ", ".join([f"'{c}' ({pct:.1f}%)" for c, pct in top_missing_cols[:3]])
                insights.append(StatisticalInsight(
                    category="Missingness",
                    title="Missing Values Present",
                    description=f"Missing data concentrated in {top_m_str}. Handled via fold-safe median/mode imputation.",
                    importance="info"
                ))

        return insights
