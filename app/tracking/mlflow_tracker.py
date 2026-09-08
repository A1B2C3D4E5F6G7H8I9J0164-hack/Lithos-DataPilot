"""
MLflow experiment tracking integration with local SQLite storage backend.
"""
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import mlflow
from mlflow.entities import ViewType


class MLflowTracker:
    """Manages local experiment tracking, run logging, and run queries."""

    DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
    DEFAULT_EXPERIMENT_NAME = "Autonomous_Data_Scientist"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", self.DEFAULT_TRACKING_URI)
        self.experiment_name = experiment_name or self.DEFAULT_EXPERIMENT_NAME
        
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            # Fallback for file store
            self.experiment_id = "0"

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Logs an experiment run with parameters, metrics, and JSON artifacts.

        Returns:
            run_id: str
        """
        all_tags = {"platform": "Autonomous Data Scientist", "version": "1.0.0"}
        if tags:
            all_tags.update(tags)

        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id

            # Log parameters (sanitize complex objects to strings)
            sanitized_params = {}
            for k, v in params.items():
                if isinstance(v, (dict, list)):
                    sanitized_params[k] = json.dumps(v)[:250]
                else:
                    sanitized_params[k] = str(v)[:250]
            mlflow.log_params(sanitized_params)

            # Log metrics
            clean_metrics = {}
            for k, v in metrics.items():
                try:
                    clean_metrics[k] = float(v)
                except (ValueError, TypeError):
                    continue
            mlflow.log_metrics(clean_metrics)

            # Log tags
            mlflow.set_tags(all_tags)

            # Log artifacts if provided
            if artifacts:
                os.makedirs("artifacts/temp", exist_ok=True)
                for art_name, art_data in artifacts.items():
                    temp_path = os.path.join("artifacts/temp", f"{art_name}.json")
                    with open(temp_path, "w") as f:
                        json.dump(art_data, f, indent=2, default=str)
                    mlflow.log_artifact(temp_path)
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            return run_id

    def get_experiment_runs(self, max_runs: int = 50) -> pd.DataFrame:
        """
        Retrieves past experiment runs formatted for table display.
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_runs,
                order_by=["start_time DESC"]
            )
            if runs.empty:
                return pd.DataFrame()

            # Clean and project columns
            cols_to_keep = ["run_id", "status", "start_time"]
            rename_map = {
                "run_id": "Run ID",
                "status": "Status",
                "start_time": "Timestamp",
            }

            # Extract param and metric columns
            param_cols = [c for c in runs.columns if c.startswith("params.")]
            metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
            tag_cols = [c for c in runs.columns if c.startswith("tags.")]

            # Specific high-value columns
            if "tags.mlflow.runName" in runs.columns:
                cols_to_keep.append("tags.mlflow.runName")
                rename_map["tags.mlflow.runName"] = "Run Name"

            for c in param_cols:
                cols_to_keep.append(c)
                rename_map[c] = c.replace("params.", "param_")

            for c in metric_cols:
                cols_to_keep.append(c)
                rename_map[c] = c.replace("metrics.", "metric_")

            sub_df = runs[cols_to_keep].rename(columns=rename_map)
            return sub_df
        except Exception as e:
            return pd.DataFrame()
