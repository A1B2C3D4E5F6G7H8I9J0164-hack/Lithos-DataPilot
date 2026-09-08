"""
Data quality engine detecting anomalies, leakage, cardinality, missingness, and generating audits.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .profiler import DatasetProfile


class QualitySeverity(str, Enum):
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"


@dataclass
class QualityIssue:
    """Individual data quality finding with audit trail."""
    rule_id: str
    feature: Optional[str]
    severity: QualitySeverity
    detected_metric: str
    detected_value: Any
    description: str
    recommended_action: str
    applied_action: Optional[str] = None  # None until preprocessor applies it


@dataclass
class QualityAuditReport:
    """Consolidated audit report across all dataset features."""
    overall_score: int  # 0 to 100
    healthy_checks_count: int
    warning_count: int
    critical_count: int
    issues: List[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "healthy_count": self.healthy_checks_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "issues": [
                {
                    "rule_id": issue.rule_id,
                    "feature": issue.feature,
                    "severity": issue.severity.value,
                    "metric": issue.detected_metric,
                    "value": issue.detected_value,
                    "description": issue.description,
                    "recommended": issue.recommended_action,
                    "applied": issue.applied_action,
                }
                for issue in self.issues
            ]
        }


class DataQualityAuditor:
    """Audits tabular datasets against strict quality rules."""

    @classmethod
    def audit(
        cls,
        df: pd.DataFrame,
        profile: DatasetProfile,
        target_column: Optional[str] = None
    ) -> QualityAuditReport:
        """
        Runs comprehensive data quality verification.

        Args:
            df: Raw input DataFrame.
            profile: Pre-calculated DatasetProfile.
            target_column: Optional target feature.

        Returns:
            QualityAuditReport with categorized findings.
        """
        issues: List[QualityIssue] = []
        n_rows = profile.row_count

        # 1. Check Missing Values per Column
        for col_name, col_prof in profile.column_profiles.items():
            if col_prof.null_percentage > 40.0:
                issues.append(QualityIssue(
                    rule_id="HIGH_MISSINGNESS",
                    feature=col_name,
                    severity=QualitySeverity.CRITICAL,
                    detected_metric="Missingness %",
                    detected_value=f"{col_prof.null_percentage:.1f}%",
                    description=f"Column '{col_name}' is missing {col_prof.null_percentage:.1f}% of values.",
                    recommended_action="Drop column or add missingness indicator flag with median imputation.",
                    applied_action="Median imputation with missing indicator enabled in preprocessing pipeline."
                ))
            elif col_prof.null_percentage > 5.0:
                issues.append(QualityIssue(
                    rule_id="MODERATE_MISSINGNESS",
                    feature=col_name,
                    severity=QualitySeverity.WARNING,
                    detected_metric="Missingness %",
                    detected_value=f"{col_prof.null_percentage:.1f}%",
                    description=f"Column '{col_name}' is missing {col_prof.null_percentage:.1f}% of values.",
                    recommended_action="Impute with median (numerical) or most frequent (categorical).",
                    applied_action="Imputed in preprocessing pipeline using fold-specific statistics."
                ))

        # 2. Check Duplicate Rows
        if profile.duplicate_rows_count > 0:
            severity = QualitySeverity.CRITICAL if profile.duplicate_rows_percentage > 10.0 else QualitySeverity.WARNING
            issues.append(QualityIssue(
                rule_id="DUPLICATE_ROWS",
                feature=None,
                severity=severity,
                detected_metric="Duplicate Rows",
                detected_value=f"{profile.duplicate_rows_count} ({profile.duplicate_rows_percentage:.1f}%)",
                description=f"Dataset contains {profile.duplicate_rows_count} duplicate row(s).",
                recommended_action="Deduplicate rows to avoid training bias and data leakage.",
                applied_action="Deduplication recommended prior to train/test splitting."
            ))

        # 3. Check Constant / Zero Variance Columns
        for col in profile.constant_columns:
            issues.append(QualityIssue(
                rule_id="ZERO_VARIANCE",
                feature=col,
                severity=QualitySeverity.CRITICAL,
                detected_metric="Unique Values",
                detected_value=profile.column_profiles[col].unique_count,
                description=f"Column '{col}' carries no variance (single unique value).",
                recommended_action="Drop column to eliminate dead weights and singular matrices.",
                applied_action="Automatically excluded from candidate feature set."
            ))

        # 4. Check High Cardinality Categoricals
        for col in profile.high_cardinality_columns:
            u_count = profile.column_profiles[col].unique_count
            inferred = profile.column_profiles[col].inferred_type
            if inferred == "id":
                issues.append(QualityIssue(
                    rule_id="IDENTIFIER_COLUMN",
                    feature=col,
                    severity=QualitySeverity.WARNING,
                    detected_metric="Unique Ratio",
                    detected_value=f"{profile.column_profiles[col].unique_ratio:.2%}",
                    description=f"Column '{col}' exhibits near-unique row values ({u_count} unique IDs).",
                    recommended_action="Drop identifier columns to prevent high-dimensional memorization and overfitting.",
                    applied_action="Excluded identifier column from feature matrix."
                ))
            else:
                issues.append(QualityIssue(
                    rule_id="HIGH_CARDINALITY",
                    feature=col,
                    severity=QualitySeverity.WARNING,
                    detected_metric="Unique Categories",
                    detected_value=u_count,
                    description=f"Categorical column '{col}' has {u_count} distinct categories.",
                    recommended_action="Use frequency/target encoding or group rare categories into 'Other'.",
                    applied_action="Target/Ordinal or frequency encoding applied in ColumnTransformer."
                ))

        # 5. Check Numerical Outliers (IQR heuristic)
        for col in profile.numerical_columns:
            if col == target_column:
                continue
            series = df[col].dropna()
            if len(series) > 20:
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                if iqr > 0:
                    outliers = series[(series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)]
                    outlier_pct = (len(outliers) / len(series)) * 100
                    if outlier_pct > 5.0:
                        issues.append(QualityIssue(
                            rule_id="POTENTIAL_OUTLIERS",
                            feature=col,
                            severity=QualitySeverity.WARNING,
                            detected_metric="Outlier % (IQR)",
                            detected_value=f"{outlier_pct:.1f}% ({len(outliers)} rows)",
                            description=f"Column '{col}' has {outlier_pct:.1f}% outliers outside 1.5 * IQR.",
                            recommended_action="Use RobustScaler or tree-based models resilient to monotonic extremes.",
                            applied_action="Standard/Robust scaling and tree ensembles (XGBoost/RandomForest) utilized."
                        ))

        # 6. Check Target Leakage (High Correlation with Target)
        if target_column and target_column in df.columns:
            if profile.column_profiles[target_column].inferred_type == "numerical":
                target_series = df[target_column].dropna()
                for col in profile.numerical_columns:
                    if col == target_column:
                        continue
                    valid_idx = df[[col, target_column]].dropna().index
                    if len(valid_idx) > 20:
                        corr = float(np.abs(df.loc[valid_idx, col].corr(df.loc[valid_idx, target_column])))
                        if corr >= 0.98:
                            issues.append(QualityIssue(
                                rule_id="TARGET_LEAKAGE",
                                feature=col,
                                severity=QualitySeverity.CRITICAL,
                                detected_metric="Target Correlation",
                                detected_value=f"{corr:.3f}",
                                description=f"Feature '{col}' has suspiciously high correlation ({corr:.3f}) with target '{target_column}'.",
                                recommended_action="Inspect domain logic. Highly probable target proxy/leakage. Drop if derived from target.",
                                applied_action="Flagged for investigator review; pipeline retains with warning."
                            ))

        # Calculate score (deduct for critical and warning)
        crit_count = sum(1 for i in issues if i.severity == QualitySeverity.CRITICAL)
        warn_count = sum(1 for i in issues if i.severity == QualitySeverity.WARNING)
        healthy_checks = max(0, len(profile.column_profiles) + 4 - (crit_count + warn_count))

        score = max(0, 100 - (crit_count * 20 + warn_count * 5))

        return QualityAuditReport(
            overall_score=score,
            healthy_checks_count=healthy_checks,
            warning_count=warn_count,
            critical_count=crit_count,
            issues=issues,
        )
