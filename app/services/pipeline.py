"""
End-to-end autonomous data science workflow orchestrator.
Manages step transitions, real-time logging, decision records, artifact serialization, and MLflow logging.
"""
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.data.loader import DataLoader
from app.data.validator import DataValidator, ValidationResult
from app.data.profiler import DataProfiler, DatasetProfile
from app.data.quality import DataQualityAuditor, QualityAuditReport
from app.data.preprocessing import LeakageSafePreprocessor
from app.data.eda import EDAEngine, StatisticalInsight
from app.ml.problem_detector import ProblemDetector, ProblemSpec, ProblemType
from app.ml.models import ModelFactory, ModelCandidate
from app.ml.cross_validation import CrossValidationRunner
from app.ml.trainer import ModelTrainer, BenchmarkResult
from app.ml.tuner import HyperparameterTuner, TuningResult
from app.ml.explainability import ModelExplainer, ExplanationResult
from app.ml.evaluator import ModelEvaluator
from app.tracking.mlflow_tracker import MLflowTracker


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState:
    """Encapsulates current execution state, step progress, and generated artifacts."""

    PIPELINE_STEPS = [
        ("validation", "Dataset Validation"),
        ("profiling", "Data Profiling"),
        ("quality", "Quality & Leakage Audit"),
        ("problem_detection", "Problem Type Detection"),
        ("preprocessing", "Leakage-Safe Preprocessing"),
        ("benchmarking", "Candidate Model Benchmarking"),
        ("tuning", "Optuna Hyperparameter Tuning"),
        ("explainability", "SHAP Feature Explainability"),
        ("registration", "Model & Artifact Registration"),
    ]

    def __init__(self):
        self.step_statuses: Dict[str, StepStatus] = {
            step_id: StepStatus.PENDING for step_id, _ in self.PIPELINE_STEPS
        }
        self.logs: List[str] = []
        self.decisions: List[str] = []
        self.error_message: Optional[str] = None
        self.is_running: bool = False
        self.is_completed: bool = False

        # Phase artifacts
        self.validation_result: Optional[ValidationResult] = None
        self.profile: Optional[DatasetProfile] = None
        self.quality_report: Optional[QualityAuditReport] = None
        self.insights: List[StatisticalInsight] = []
        self.problem_spec: Optional[ProblemSpec] = None
        self.preprocessor: Optional[LeakageSafePreprocessor] = None
        self.benchmark_result: Optional[BenchmarkResult] = None
        self.tuning_result: Optional[TuningResult] = None
        self.explanation_result: Optional[ExplanationResult] = None
        self.mlflow_run_id: Optional[str] = None
        self.model_bundle_path: Optional[str] = None

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.logs.append(formatted)

    def add_decision(self, decision: str) -> None:
        self.decisions.append(decision)
        self.log(f"DECISION: {decision}")

    def update_step(self, step_id: str, status: StepStatus) -> None:
        self.step_statuses[step_id] = status


class AutonomousPipelineService:
    """Executes the complete autonomous machine learning lifecycle."""

    ARTIFACTS_DIR = "artifacts"
    BUNDLE_FILENAME = "model_bundle.joblib"

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracker = MLflowTracker(tracking_uri=tracking_uri)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

    def execute(
        self,
        df: pd.DataFrame,
        target_column: str,
        state: Optional[PipelineState] = None,
        optuna_trials: int = 15,
        cv_splits: int = 5,
        random_seed: int = 42,
        step_callback: Optional[Callable[[str, StepStatus], None]] = None,
    ) -> PipelineState:
        """
        Executes the end-to-end data science pipeline.
        """
        if state is None:
            state = PipelineState()

        state.is_running = True
        state.is_completed = False
        state.error_message = None

        def notify(step_id: str, status: StepStatus):
            state.update_step(step_id, status)
            if step_callback:
                step_callback(step_id, status)

        try:
            # 1. Dataset Validation
            notify("validation", StepStatus.RUNNING)
            state.log(f"Validating dataset structure and target column '{target_column}'...")
            val_res = DataValidator.validate(df, target_column=target_column)
            state.validation_result = val_res

            if not val_res.is_valid:
                err_str = "; ".join(val_res.errors)
                state.log(f"Validation failed: {err_str}")
                notify("validation", StepStatus.FAILED)
                raise ValueError(err_str)

            state.log(f"Validation passed: {val_res.row_count} rows, {val_res.column_count} columns.")
            notify("validation", StepStatus.COMPLETED)

            # 2. Data Profiling & Automated EDA
            notify("profiling", StepStatus.RUNNING)
            state.log("Profiling column distributions, null ratios, and data types...")
            profile = DataProfiler.profile(df)
            state.profile = profile
            state.insights = EDAEngine.generate_factual_insights(df, profile, target_column)
            state.log(
                f"Profiling complete: {len(profile.numerical_columns)} numerical, "
                f"{len(profile.categorical_columns)} categorical features detected."
            )
            notify("profiling", StepStatus.COMPLETED)

            # 3. Data Quality & Leakage Audit
            notify("quality", StepStatus.RUNNING)
            state.log("Auditing dataset for duplicates, zero-variance columns, outliers, and leakage...")
            quality_report = DataQualityAuditor.audit(df, profile, target_column=target_column)
            state.quality_report = quality_report
            state.log(
                f"Data quality audit finished with score {quality_report.overall_score}/100 "
                f"({quality_report.critical_count} critical, {quality_report.warning_count} warnings)."
            )
            notify("quality", StepStatus.COMPLETED)

            # 4. Problem Type Detection
            notify("problem_detection", StepStatus.RUNNING)
            state.log(f"Inferring ML task taxonomy for target '{target_column}'...")
            problem_spec = ProblemDetector.detect(df, target_column)
            state.problem_spec = problem_spec
            for reason in problem_spec.reasoning:
                state.add_decision(reason)
            notify("problem_detection", StepStatus.COMPLETED)

            # 5. Leakage-Safe Preprocessing Pipeline
            notify("preprocessing", StepStatus.RUNNING)
            state.log("Building leakage-safe ColumnTransformer architecture...")
            
            # Clean dataset of rows where target is NaN
            clean_df = df.dropna(subset=[target_column]).copy()
            X = clean_df.drop(columns=[target_column])
            y = clean_df[target_column]

            preprocessor = LeakageSafePreprocessor.from_profile(
                profile=profile,
                target_column=target_column,
            )
            state.preprocessor = preprocessor
            state.add_decision(
                f"Configured preprocessor: {len(preprocessor.numerical_features)} numerical features (median imputer + scaler), "
                f"{len(preprocessor.categorical_features)} categorical features (mode imputer + OHE/ordinal), "
                f"excluded {len(preprocessor.drop_features)} redundant/ID columns: {preprocessor.drop_features}"
            )
            notify("preprocessing", StepStatus.COMPLETED)

            # 6. Model Benchmarking & CV Leaderboard
            notify("benchmarking", StepStatus.RUNNING)
            state.log(
                f"Benchmarking candidate models using {cv_splits}-fold {problem_spec.cv_strategy} "
                f"optimizing for {problem_spec.recommended_primary_metric.upper()}..."
            )

            benchmark_res = ModelTrainer.benchmark(
                problem_spec=problem_spec,
                preprocessor_builder=preprocessor,
                X=X,
                y=y,
                primary_metric=problem_spec.recommended_primary_metric,
                n_splits=cv_splits,
                random_seed=random_seed,
                progress_callback=lambda name, idx, total: state.log(
                    f"Evaluating model [{idx}/{total}]: {name}..."
                ),
            )
            state.benchmark_result = benchmark_res
            state.add_decision(benchmark_res.decision_reasoning)
            notify("benchmarking", StepStatus.COMPLETED)

            # 7. Hyperparameter Optimization via Optuna
            notify("tuning", StepStatus.RUNNING)
            state.log(
                f"Launching Optuna Bayesian optimization ({optuna_trials} trials) "
                f"for top candidate: {benchmark_res.best_candidate_name}..."
            )

            best_candidate_obj = next(
                c for c in ModelFactory.get_candidates(problem_spec.problem_type, random_seed)
                if c.model_id == benchmark_res.best_candidate_id
            )

            tuning_res = HyperparameterTuner.tune(
                candidate=best_candidate_obj,
                baseline_score=benchmark_res.best_score,
                preprocessor_builder=preprocessor,
                problem_spec=problem_spec,
                X=X,
                y=y,
                n_trials=optuna_trials,
                timeout=180,
                random_seed=random_seed,
                trial_callback=lambda t_num, score, params: state.log(
                    f"Trial {t_num:02d}: Score = {score:.4f}"
                ),
            )
            state.tuning_result = tuning_res
            state.add_decision(
                f"Hyperparameter tuning completed: {tuning_res.metric_name.upper()} improved from "
                f"{tuning_res.baseline_score:.4f} to {tuning_res.best_score:.4f} (Delta: {tuning_res.score_delta:+.4f})."
            )
            notify("tuning", StepStatus.COMPLETED)

            # 8. Explainability via SHAP
            notify("explainability", StepStatus.RUNNING)
            state.log("Generating global and local instance-level SHAP attributions...")
            active_pipeline = tuning_res.optimized_pipeline or benchmark_res.results[benchmark_res.best_candidate_id].fitted_pipeline
            
            explanation_res = ModelExplainer.explain(
                pipeline=active_pipeline,
                problem_spec=problem_spec,
                X_sample=X.head(20),
                model_id=benchmark_res.best_candidate_id,
            )
            state.explanation_result = explanation_res
            state.log(
                f"SHAP explanations computed via {explanation_res.explainer_type}. "
                f"Top driver: '{explanation_res.top_features[0] if explanation_res.top_features else 'N/A'}'."
            )
            notify("explainability", StepStatus.COMPLETED)

            # 9. Model Registration, MLflow Tracking & Bundle Persistence
            notify("registration", StepStatus.RUNNING)
            state.log("Serializing production model bundle and logging to MLflow...")

            # Extract schema for FastAPI and prediction UI
            feature_schema = []
            for col in X.columns:
                if col not in preprocessor.drop_features:
                    feature_schema.append({
                        "name": col,
                        "dtype": str(X[col].dtype),
                        "sample_value": X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else 0,
                        "is_numerical": col in preprocessor.numerical_features,
                        "unique_values": X[col].dropna().unique()[:20].tolist() if col in preprocessor.categorical_features else None,
                    })

            # Save full model bundle with preprocessor embedded
            bundle_path = os.path.join(self.ARTIFACTS_DIR, self.BUNDLE_FILENAME)
            model_bundle = {
                "pipeline": active_pipeline,
                "problem_spec": problem_spec,
                "target_column": target_column,
                "feature_schema": feature_schema,
                "numerical_features": preprocessor.numerical_features,
                "categorical_features": preprocessor.categorical_features,
                "drop_features": preprocessor.drop_features,
                "classes": problem_spec.classes,
                "best_model_name": benchmark_res.best_candidate_name,
                "best_model_id": benchmark_res.best_candidate_id,
                "primary_metric": problem_spec.recommended_primary_metric,
                "best_score": tuning_res.best_score,
                "baseline_score": tuning_res.baseline_score,
                "best_params": tuning_res.best_params,
                "created_at": datetime.now().isoformat(),
            }
            joblib.dump(model_bundle, bundle_path)
            state.model_bundle_path = bundle_path

            # Log into MLflow
            cv_best = benchmark_res.results[benchmark_res.best_candidate_id]
            run_id = self.tracker.log_run(
                run_name=f"{benchmark_res.best_candidate_name}_{problem_spec.problem_type.value}",
                params={
                    "model_id": benchmark_res.best_candidate_id,
                    "problem_type": problem_spec.problem_type.value,
                    "target": target_column,
                    "n_samples": len(X),
                    "n_features": len(feature_schema),
                    **tuning_res.best_params,
                },
                metrics={
                    f"cv_{k}_mean": v for k, v in cv_best.cv_mean.items()
                } | {
                    "tuned_best_score": tuning_res.best_score,
                    "baseline_score": tuning_res.baseline_score,
                },
                tags={
                    "target_column": target_column,
                    "best_model": benchmark_res.best_candidate_name,
                },
                artifacts={
                    "quality_report": quality_report.to_dict(),
                    "top_features": explanation_res.top_features,
                }
            )
            state.mlflow_run_id = run_id
            state.log(f"Model saved to '{bundle_path}'. MLflow run recorded (Run ID: {run_id[:8]}).")
            notify("registration", StepStatus.COMPLETED)

            state.is_running = False
            state.is_completed = True
            state.log("Autonomous Data Science pipeline completed successfully!")

        except Exception as e:
            state.is_running = False
            state.error_message = str(e)
            state.log(f"Pipeline execution halted due to error: {str(e)}")
            raise e

        return state
