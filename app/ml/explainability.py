"""
Model explainability engine computing genuine global and local SHAP explanations.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import shap

from .problem_detector import ProblemType, ProblemSpec


@dataclass
class LocalFeatureContribution:
    """Individual feature contribution to a specific prediction."""
    feature_name: str
    feature_value: Any
    shap_value: float
    impact_direction: str  # 'positive' (increases target) or 'negative' (decreases target)
    magnitude: float
    plain_english: str


@dataclass
class LocalExplanation:
    """Complete local explanation for a specific prediction instance."""
    sample_index: int
    predicted_value: Any
    predicted_probability: Optional[float]
    base_value: float  # Expected model output
    contributions: List[LocalFeatureContribution]


@dataclass
class ExplanationResult:
    """Global and local explainability artifact."""
    model_id: str
    problem_type: ProblemType
    feature_names: List[str]
    global_importance: Dict[str, float]  # Feature -> mean |SHAP|
    top_features: List[str]
    sample_explanations: Dict[int, LocalExplanation] = field(default_factory=dict)
    explainer_type: str = "TreeExplainer"


class ModelExplainer:
    """Computes global feature importance and instance-level SHAP attributions."""

    @classmethod
    def explain(
        cls,
        pipeline: Pipeline,
        problem_spec: ProblemSpec,
        X_sample: pd.DataFrame,
        model_id: str,
        n_background_samples: int = 50,
        n_explain_samples: int = 20,
    ) -> ExplanationResult:
        """
        Computes SHAP explanations on transformed features.
        """
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        # Transform features
        X_transformed = preprocessor.transform(X_sample)
        
        # Get readable feature names
        try:
            raw_names = preprocessor.get_feature_names_out()
            feature_names = [n.split("__", 1)[-1] for n in raw_names]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        if isinstance(X_transformed, pd.DataFrame):
            X_trans_arr = X_transformed.values
        elif hasattr(X_transformed, "toarray"):
            X_trans_arr = X_transformed.toarray()
        else:
            X_trans_arr = np.asarray(X_transformed)

        # Select background data for SHAP
        bg_size = min(len(X_trans_arr), n_background_samples)
        X_bg = X_trans_arr[:bg_size]

        explainer = None
        explainer_name = "TreeExplainer"
        shap_values = None

        # 1. Try TreeExplainer (fast and exact for XGBoost / RandomForest / HistGB)
        try:
            if "xgb" in model_id or "forest" in model_id:
                explainer = shap.TreeExplainer(model)
                explainer_name = "TreeExplainer"
                shap_values = explainer.shap_values(X_trans_arr[:n_explain_samples])
            elif "hist" in model_id:
                # HistGradientBoosting often uses Permutation or Kernel/Tree
                explainer = shap.TreeExplainer(model)
                explainer_name = "TreeExplainer"
                shap_values = explainer.shap_values(X_trans_arr[:n_explain_samples])
            else:
                # Linear / Ridge / Logistic
                explainer = shap.LinearExplainer(model, X_bg)
                explainer_name = "LinearExplainer"
                shap_values = explainer.shap_values(X_trans_arr[:n_explain_samples])
        except Exception:
            # Fallback to KernelExplainer on background summary
            try:
                bg_summary = shap.kmeans(X_bg, min(10, len(X_bg)))
                if hasattr(model, "predict_proba"):
                    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], bg_summary)
                else:
                    explainer = shap.KernelExplainer(model.predict, bg_summary)
                explainer_name = "KernelExplainer"
                shap_values = explainer.shap_values(X_trans_arr[:min(10, n_explain_samples)])
            except Exception:
                # Fallback to Gini / Feature Importances
                explainer_name = "FeatureImportanceFallback"

        # Global feature importance calculation
        global_importance: Dict[str, float] = {}
        
        if shap_values is not None:
            # Handle multi-class or list of arrays
            if isinstance(shap_values, list):
                # Binary classification often gives list [shap_class_0, shap_class_1]
                sv_arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Shape (samples, features, classes)
                sv_arr = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            else:
                sv_arr = np.asarray(shap_values)

            mean_abs_shap = np.mean(np.abs(sv_arr), axis=0)
            for idx, feat in enumerate(feature_names):
                val = float(mean_abs_shap[idx]) if idx < len(mean_abs_shap) else 0.0
                global_importance[feat] = round(val, 5)
        else:
            # Gini / Coefficient fallback
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                for idx, feat in enumerate(feature_names):
                    val = float(fi[idx]) if idx < len(fi) else 0.0
                    global_importance[feat] = round(val, 5)
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_).flatten()
                for idx, feat in enumerate(feature_names):
                    val = float(coefs[idx]) if idx < len(coefs) else 0.0
                    global_importance[feat] = round(val, 5)
            else:
                for idx, feat in enumerate(feature_names):
                    global_importance[feat] = 1.0 / max(1, len(feature_names))

        # Sort global importance
        sorted_importance = dict(
            sorted(global_importance.items(), key=lambda item: item[1], reverse=True)
        )
        top_features = list(sorted_importance.keys())[:10]

        # Generate Local Explanations for first few samples
        sample_explanations: Dict[int, LocalExplanation] = {}
        n_samples_to_explain = min(len(X_sample), 10)
        
        # Predictions on original DataFrame
        preds = pipeline.predict(X_sample.iloc[:n_samples_to_explain])
        probs = None
        if problem_spec.is_classification and hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(X_sample.iloc[:n_samples_to_explain])
            except Exception:
                probs = None

        base_val = 0.0
        if explainer is not None and hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_val = float(ev[1] if len(ev) > 1 else ev[0])
            else:
                base_val = float(ev)

        for s_idx in range(n_samples_to_explain):
            pred_val = preds[s_idx]
            pred_prob = None
            if probs is not None:
                pred_prob = float(probs[s_idx, 1]) if probs.shape[1] > 1 else float(probs[s_idx, 0])

            contributions: List[LocalFeatureContribution] = []
            
            if shap_values is not None and s_idx < len(sv_arr):
                row_shap = sv_arr[s_idx]
                for f_idx, f_name in enumerate(feature_names):
                    if f_idx < len(row_shap):
                        s_val = float(row_shap[f_idx])
                        direction = "positive" if s_val >= 0 else "negative"
                        mag = abs(s_val)
                        
                        # Plain English interpretation
                        if problem_spec.is_classification:
                            effect = "increases target likelihood" if s_val >= 0 else "decreases target likelihood"
                        else:
                            effect = f"shifts target value by {'+' if s_val >= 0 else ''}{s_val:.2f}"

                        # Original feature value if available
                        orig_val = X_sample.iloc[s_idx].get(f_name, "Encoded")

                        contributions.append(LocalFeatureContribution(
                            feature_name=f_name,
                            feature_value=orig_val,
                            shap_value=round(s_val, 4),
                            impact_direction=direction,
                            magnitude=round(mag, 4),
                            plain_english=f"{f_name} {effect} ({'+' if s_val >= 0 else ''}{s_val:.3f})",
                        ))

                # Sort by impact magnitude
                contributions.sort(key=lambda x: x.magnitude, reverse=True)

            sample_explanations[s_idx] = LocalExplanation(
                sample_index=s_idx,
                predicted_value=str(pred_val),
                predicted_probability=pred_prob,
                base_value=base_val,
                contributions=contributions[:10],
            )

        return ExplanationResult(
            model_id=model_id,
            problem_type=problem_spec.problem_type,
            feature_names=feature_names,
            global_importance=sorted_importance,
            top_features=top_features,
            sample_explanations=sample_explanations,
            explainer_type=explainer_name,
        )
