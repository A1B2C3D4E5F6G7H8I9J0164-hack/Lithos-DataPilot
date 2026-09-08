"""
Model benchmarking and transparent multi-candidate ranking engine.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import pandas as pd

from .problem_detector import ProblemSpec
from .models import ModelFactory, ModelCandidate
from .cross_validation import CrossValidationRunner, ModelCVResult
from app.data.preprocessing import LeakageSafePreprocessor


@dataclass
class BenchmarkResult:
    """Complete benchmarking run with leaderboard and selected best candidate."""
    primary_metric: str
    is_higher_better: bool
    leaderboard_df: pd.DataFrame
    best_candidate_id: str
    best_candidate_name: str
    best_score: float
    results: Dict[str, ModelCVResult]
    decision_reasoning: str


class ModelTrainer:
    """Coordinates multi-model cross-validation benchmarking and candidate selection."""

    LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mse", "log_loss", "mape"}

    @classmethod
    def benchmark(
        cls,
        problem_spec: ProblemSpec,
        preprocessor_builder: LeakageSafePreprocessor,
        X: pd.DataFrame,
        y: pd.Series,
        primary_metric: Optional[str] = None,
        n_splits: int = 5,
        random_seed: int = 42,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> BenchmarkResult:
        """
        Executes cross-validation benchmark over all candidates for problem type.
        """
        metric = primary_metric or problem_spec.recommended_primary_metric
        is_higher_better = metric not in cls.LOWER_IS_BETTER_METRICS

        candidates = ModelFactory.get_candidates(problem_spec.problem_type, random_seed=random_seed)
        results: Dict[str, ModelCVResult] = {}

        total = len(candidates)
        for idx, cand in enumerate(candidates):
            if progress_callback:
                progress_callback(cand.display_name, idx + 1, total)

            cv_res = CrossValidationRunner.run_cv(
                candidate=cand,
                preprocessor_builder=preprocessor_builder,
                problem_spec=problem_spec,
                X=X,
                y=y,
                n_splits=n_splits,
                random_seed=random_seed,
                primary_metric=metric,
            )
            results[cand.model_id] = cv_res

        # Build comparison leaderboard
        rows = []
        for m_id, res in results.items():
            row: Dict[str, Any] = {
                "Model ID": m_id,
                "Model": res.display_name,
                "CV Mean": res.primary_mean,
                "CV Std": res.primary_std,
                "Train Time (s)": res.total_time_sec,
            }
            # Add all individual metric means
            for k, v in res.cv_mean.items():
                col_name = k.upper() if len(k) <= 4 else k.replace("_", " ").title()
                row[col_name] = v
            rows.append(row)

        df_leaderboard = pd.DataFrame(rows)
        # Sort by primary metric mean
        df_leaderboard = df_leaderboard.sort_values(
            by="CV Mean",
            ascending=(not is_higher_better)
        ).reset_index(drop=True)

        best_row = df_leaderboard.iloc[0]
        best_id = str(best_row["Model ID"])
        best_name = str(best_row["Model"])
        best_score = float(best_row["CV Mean"])

        reasoning = (
            f"Selected '{best_name}' as top candidate with CV mean {metric.upper()} = {best_score:.4f} "
            f"(±{results[best_id].primary_std:.4f}) across {n_splits}-fold cross-validation."
        )

        return BenchmarkResult(
            primary_metric=metric,
            is_higher_better=is_higher_better,
            leaderboard_df=df_leaderboard,
            best_candidate_id=best_id,
            best_candidate_name=best_name,
            best_score=best_score,
            results=results,
            decision_reasoning=reasoning,
        )
