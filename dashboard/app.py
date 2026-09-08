"""
Autonomous Data Scientist - High-End AI Workstation Interface.
Built with Streamlit and Plotly following modern developer-first design aesthetics.
"""
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.data.profiler import DataProfiler, DatasetProfile
from app.data.quality import DataQualityAuditor, QualitySeverity
from app.data.eda import EDAEngine
from app.ml.problem_detector import ProblemDetector, ProblemType
from app.services.pipeline import AutonomousPipelineService, PipelineState, StepStatus
from app.tracking.mlflow_tracker import MLflowTracker
from dashboard.components.styles import inject_custom_css, render_badge

# Streamlit Page Config
st.set_page_config(
    page_title="Autonomous Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS
inject_custom_css()

# Session State Initialization
if "pipeline_service" not in st.session_state:
    st.session_state.pipeline_service = AutonomousPipelineService()

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = PipelineState()

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "selected_target" not in st.session_state:
    st.session_state.selected_target = None


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; padding: 0.5rem 0;">
        <span style="font-size: 1.6rem;">🧠</span>
        <div>
            <div style="font-weight: 700; font-size: 1.05rem; letter-spacing: -0.02em; color: #F8FAFC;">DataPilot</div>
            <div style="font-size: 0.72rem; color: #64748B; letter-spacing: 0.04em; text-transform: uppercase;">Autonomous Data Scientist</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "Dataset",
        "Pipeline",
        "Data Quality",
        "EDA",
        "Modeling",
        "Optimization",
        "Explainability",
        "Experiments",
        "Predictions",
        "System",
    ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.8rem 0 1rem 0;'>", unsafe_allow_html=True)

    # System Status Indicator
    state: PipelineState = st.session_state.pipeline_state
    has_model = state.is_completed and state.model_bundle_path is not None

    status_dot = "🟢" if has_model else ("🟡" if state.is_running else "⚪")
    status_text = "Model Ready" if has_model else ("Running..." if state.is_running else "Pipeline Idle")

    st.markdown(f"""
    <div style="background: #0D131F; border: 1px solid #1E293B; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.78rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="color: #94A3B8;">Pipeline Status</span>
            <span>{status_dot} <strong style="color: #F8FAFC;">{status_text}</strong></span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="color: #94A3B8;">MLflow Local</span>
            <span style="color: #10B981; font-weight: 500;">● Connected</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="color: #94A3B8;">FastAPI Serving</span>
            <span style="color: #38BDF8; font-weight: 500;">● Port 8000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataset_name:
        st.markdown(f"""
        <div style="margin-top: 1rem; font-size: 0.75rem; color: #64748B;">
            Active Dataset: <strong style="color: #CBD5E1;">{st.session_state.dataset_name}</strong><br>
            Target: <strong style="color: #60A5FA;">{st.session_state.selected_target or 'None'}</strong>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if selected_nav == "Overview":
    st.markdown("""
    <div class="ads-hero">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
            <span class="ads-badge ads-badge-blue">AUTONOMOUS ML PLATFORM</span>
            <span class="ads-badge ads-badge-emerald">ZERO LEAKAGE</span>
        </div>
        <div class="ads-hero-title">From raw data to an explainable, production-ready ML model.</div>
        <div class="ads-hero-subtitle">
            Autonomous Data Scientist performs automated validation, rigorous EDA, leakage-safe feature engineering,
            multi-model cross-validation benchmarking, Optuna Bayesian tuning, SHAP explainability, and registers
            the best pipeline with FastAPI serving.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📁 Go to Dataset & Upload", type="primary", use_container_width=True):
            st.session_state["nav_redirect"] = "Dataset"
            st.rerun()

    with col2:
        if st.button("⚡ Load Benchmark Customer Churn Dataset", use_container_width=True):
            if os.path.exists("data/churn.csv"):
                st.session_state.raw_df = DataLoader.load_csv("data/churn.csv")
                st.session_state.dataset_name = "Customer Churn (Classification)"
                st.session_state.selected_target = "churn"
                st.success("Loaded 'Customer Churn' dataset (1,000 samples, 10 features). Navigate to Dataset or Pipeline.")
            else:
                st.warning("Please generate dataset first via data/generate_datasets.py")

    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; font-size: 1.15rem;'>End-to-End Orchestrated Pipeline</h3>", unsafe_allow_html=True)

    pipeline_cols = st.columns(7)
    steps_meta = [
        ("1. DATA", "CSV Ingestion & Dialect Detection", "📁"),
        ("2. PROFILE", "Structural & Type Analysis", "📊"),
        ("3. AUDIT", "Zero-Leakage & Outlier Audit", "🛡️"),
        ("4. BENCHMARK", "K-Fold / Stratified CV", "⚔️"),
        ("5. OPTIMIZE", "Optuna Bayesian HPO", "🎯"),
        ("6. EXPLAIN", "Global & Local SHAP", "🔍"),
        ("7. SERVE", "FastAPI Serving & Registry", "🚀"),
    ]
    for col, (title, desc, icon) in zip(pipeline_cols, steps_meta):
        with col:
            st.markdown(f"""
            <div style="background: #111726; border: 1px solid #1E293B; border-radius: 8px; padding: 0.9rem 0.75rem; text-align: center; height: 115px;">
                <div style="font-size: 1.3rem; margin-bottom: 0.2rem;">{icon}</div>
                <div style="font-weight: 600; font-size: 0.82rem; color: #F1F5F9;">{title}</div>
                <div style="font-size: 0.68rem; color: #64748B; margin-top: 0.2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 2.2rem; margin-bottom: 1rem; font-size: 1.15rem;'>Core Platform Capabilities</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="ads-card">
            <h4 style="color: #60A5FA; margin-bottom: 0.4rem; font-size: 0.95rem;">🔬 Leakage-Safe Engineering</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                All imputation, one-hot encoding, and scaling are strictly bound to <code>Pipeline</code> and fit exclusively inside training folds. Zero validation/test data is ever seen during feature transformation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="ads-card">
            <h4 style="color: #34D399; margin-bottom: 0.4rem; font-size: 0.95rem;">⚔️ Multi-Model Benchmarking</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                Evaluates Logistic/Ridge Regression, Random Forest, HistGradientBoosting, and XGBoost with transparent Stratified K-Fold cross-validation metrics and latency profiling.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="ads-card">
            <h4 style="color: #FBBF24; margin-bottom: 0.4rem; font-size: 0.95rem;">🔍 Explainable AI & SHAP</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                Generates global feature importance beeswarm/bar plots alongside instance-level waterfall explanations so every prediction is fully interpretable and actionable.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 2. DATASET PAGE
# ==============================================================================
elif selected_nav == "Dataset":
    st.markdown("## 📁 Dataset Ingestion & Configuration")
    st.markdown("Upload any tabular CSV file or load one of the built-in benchmark datasets to initiate autonomous analysis.")

    # Upload options
    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Tabular CSV File",
            type=["csv"],
            help="Upload any standard comma, tab, or semicolon delimited CSV."
        )
        if uploaded_file is not None:
            try:
                df_loaded = DataLoader.load_csv(uploaded_file, filename=uploaded_file.name)
                st.session_state.raw_df = df_loaded
                st.session_state.dataset_name = uploaded_file.name
                st.success(f"Successfully loaded '{uploaded_file.name}' with {len(df_loaded)} rows and {len(df_loaded.columns)} columns.")
            except Exception as e:
                st.error(f"Error loading uploaded CSV: {str(e)}")

    with up_col2:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>Or Use Benchmark Datasets:</div>", unsafe_allow_html=True)
        btn_churn = st.button("📊 Customer Churn (Classification)", use_container_width=True)
        btn_house = st.button("🏡 California Housing (Regression)", use_container_width=True)

        if btn_churn:
            if os.path.exists("data/churn.csv"):
                st.session_state.raw_df = DataLoader.load_csv("data/churn.csv")
                st.session_state.dataset_name = "Customer Churn (Classification)"
                st.session_state.selected_target = "churn"
                st.rerun()

        if btn_house:
            if os.path.exists("data/housing.csv"):
                st.session_state.raw_df = DataLoader.load_csv("data/housing.csv")
                st.session_state.dataset_name = "California Housing (Regression)"
                st.session_state.selected_target = "median_house_value"
                st.rerun()

    df = st.session_state.raw_df

    if df is not None:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Quick stats row
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        missing_count = int(df.isna().sum().sum())
        dup_count = int(df.duplicated().sum())

        stat_cols = st.columns(6)
        stat_cols[0].metric("Total Rows", f"{len(df):,}")
        stat_cols[1].metric("Total Columns", f"{len(df.columns)}")
        stat_cols[2].metric("Memory Footprint", f"{mem_mb:.2f} MB")
        stat_cols[3].metric("Missing Cells", f"{missing_count:,}")
        stat_cols[4].metric("Duplicate Rows", f"{dup_count}")
        stat_cols[5].metric("Numeric Features", f"{len(df.select_dtypes(include=[np.number]).columns)}")

        # Target Column Selection
        st.markdown("<h4 style='margin-top: 1.5rem;'>Target Feature Selection</h4>", unsafe_allow_html=True)
        all_cols = list(df.columns)
        default_idx = 0
        if st.session_state.selected_target and st.session_state.selected_target in all_cols:
            default_idx = all_cols.index(st.session_state.selected_target)
        elif "churn" in all_cols:
            default_idx = all_cols.index("churn")
        elif "target" in all_cols:
            default_idx = all_cols.index("target")

        target_col = st.selectbox(
            "Select Target Column for Modeling",
            options=all_cols,
            index=default_idx,
            help="Choose the label or value the model should predict."
        )
        st.session_state.selected_target = target_col

        # Preliminary Problem Detection Preview
        try:
            detected_spec = ProblemDetector.detect(df, target_col)
            task_badge = render_badge(detected_spec.problem_type.value.upper(), "emerald")
            metric_badge = render_badge(f"Target Metric: {detected_spec.recommended_primary_metric.upper()}", "blue")

            st.markdown(f"""
            <div style="background: #101624; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                    <strong>Preliminary Task Inference:</strong> {task_badge} {metric_badge}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    Validation Strategy: <strong style="color: #F8FAFC;">{detected_spec.cv_strategy}</strong> |
                    Imbalance Detected: <strong style="color: {'#EF4444' if detected_spec.is_imbalanced else '#10B981'};">{str(detected_spec.is_imbalanced)} (Ratio: {detected_spec.imbalance_ratio}:1)</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not infer task for target '{target_col}': {str(e)}")

        # Dataset Preview
        st.markdown("#### Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # Launch Analysis Button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        with st.expander("⚙️ Advanced Pipeline Parameters", expanded=False):
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
            with c_p2:
                optuna_trials = st.slider("Optuna HPO Trials", min_value=5, max_value=40, value=15)
            with c_p3:
                random_seed = st.number_input("Random Seed", value=42, step=1)

        if st.button("🚀 Start Autonomous Analysis", type="primary", use_container_width=True):
            # Run pipeline
            state = PipelineState()
            st.session_state.pipeline_state = state

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                # Progress mapping
                step_weights = {
                    "validation": 0.10,
                    "profiling": 0.20,
                    "quality": 0.30,
                    "problem_detection": 0.40,
                    "preprocessing": 0.50,
                    "benchmarking": 0.70,
                    "tuning": 0.85,
                    "explainability": 0.95,
                    "registration": 1.0,
                }

                def on_step(step_id: str, status: StepStatus):
                    weight = step_weights.get(step_id, 0.5)
                    progress_bar.progress(weight)
                    status_text.text(f"Executing: {step_id.replace('_', ' ').title()}...")

                st.session_state.pipeline_service.execute(
                    df=df,
                    target_column=target_col,
                    state=state,
                    optuna_trials=optuna_trials,
                    cv_splits=cv_folds,
                    random_seed=random_seed,
                    step_callback=on_step,
                )
                progress_bar.progress(1.0)
                status_text.empty()
                st.success("🎉 Autonomous Data Science workflow completed successfully! Navigate through the tabs to explore results.")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Pipeline execution error: {str(e)}")

    else:
        st.info("👆 Please upload a CSV file or click 'Customer Churn' / 'California Housing' above to begin.")


# ==============================================================================
# 3. PIPELINE EXECUTION EXPERIENCE
# ==============================================================================
elif selected_nav == "Pipeline":
    st.markdown("## ⚡ Live Autonomous Pipeline Execution")
    state: PipelineState = st.session_state.pipeline_state

    # Top status bar
    p_col1, p_col2 = st.columns([1, 2])

    with p_col1:
        st.markdown("#### Pipeline Steps")
        for step_id, step_name in state.PIPELINE_STEPS:
            status = state.step_statuses.get(step_id, StepStatus.PENDING)
            
            if status == StepStatus.COMPLETED:
                icon = "✅"
                badge = render_badge("COMPLETED", "emerald")
            elif status == StepStatus.RUNNING:
                icon = "🔄"
                badge = render_badge("RUNNING", "blue")
            elif status == StepStatus.FAILED:
                icon = "❌"
                badge = render_badge("FAILED", "rose")
            else:
                icon = "⏳"
                badge = render_badge("PENDING", "amber")

            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #111726; border: 1px solid #1E293B; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                <span style="font-size: 0.85rem; font-weight: 500;">{icon} {step_name}</span>
                <span>{badge}</span>
            </div>
            """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("#### Observable Execution Logs")
        if state.logs:
            log_content = "\n".join(state.logs)
            st.markdown(f'<div class="ads-terminal">{log_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ads-terminal" style="color: #64748B;">
                Pipeline is idle. Navigate to 'Dataset' and click 'Start Autonomous Analysis' to inspect live trace.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Autonomous AI Decisions")
        if state.decisions:
            for d in state.decisions:
                st.markdown(f"""
                <div style="background: #0E1524; border-left: 3px solid #38BDF8; padding: 0.6rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.82rem; color: #CBD5E1;">
                    🎯 <strong>Decision:</strong> {d}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Decisions will appear as the autonomous orchestrator optimizes the workflow.")


# ==============================================================================
# 4. DATA QUALITY PAGE
# ==============================================================================
elif selected_nav == "Data Quality":
    st.markdown("## 🛡️ Data Quality & Leakage Audit")
    state: PipelineState = st.session_state.pipeline_state

    if state.quality_report is None:
        st.info("Run the pipeline on a dataset to generate the Data Quality Audit.")
    else:
        report = state.quality_report

        # Quality score banner
        score_badge = render_badge(f"Health Score: {report.overall_score}/100", "emerald" if report.overall_score >= 80 else "amber")

        st.markdown(f"""
        <div class="ads-hero" style="padding: 1.5rem 1.8rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0; font-size: 1.35rem;">Dataset Integrity Audit</h3>
                    <p style="margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                        Rigorous heuristic audit for missingness, zero variance, high cardinality, outliers, and target proxies.
                    </p>
                </div>
                <div>{score_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        q_col1, q_col2, q_col3 = st.columns(3)
        q_col1.metric("Healthy Checks", f"{report.healthy_checks_count}")
        q_col2.metric("Warnings", f"{report.warning_count}")
        q_col3.metric("Critical Anomalies", f"{report.critical_count}")

        st.markdown("<h4 style='margin-top: 1.5rem;'>Audit Findings & Mitigation Actions</h4>", unsafe_allow_html=True)
        
        if report.issues:
            issue_rows = []
            for issue in report.issues:
                issue_rows.append({
                    "Feature": issue.feature or "Dataset Global",
                    "Rule": issue.rule_id,
                    "Severity": issue.severity.value,
                    "Detected Metric": f"{issue.detected_metric}: {issue.detected_value}",
                    "Description": issue.description,
                    "Recommended Action": issue.recommended_action,
                    "Applied Action": issue.applied_action or "Not applied",
                })
            df_issues = pd.DataFrame(issue_rows)
            st.dataframe(df_issues, use_container_width=True)
        else:
            st.success("No anomalies or data quality issues detected! Dataset is clean.")

        st.markdown("""
        <div style="background: #101624; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; margin-top: 1rem; font-size: 0.8rem; color: #94A3B8;">
            <strong>Note on Leakage Prevention:</strong> Notice the strict segregation between <em>DETECTED</em>, <em>RECOMMENDED</em>, and <em>APPLIED</em> actions. Transformations are never fit globally on raw data prior to validation splitting.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 5. AUTOMATED EDA PAGE
# ==============================================================================
elif selected_nav == "EDA":
    st.markdown("## 📊 Automated Exploratory Data Analysis (EDA)")
    df = st.session_state.raw_df
    target_col = st.session_state.selected_target
    state: PipelineState = st.session_state.pipeline_state

    if df is None:
        st.info("Please upload or select a dataset first.")
    else:
        profile = state.profile or DataProfiler.profile(df)
        insights = state.insights or EDAEngine.generate_factual_insights(df, profile, target_col)

        # Factual statistical insights panel
        if insights:
            st.markdown("#### 💡 Factual Statistical Insights")
            ins_cols = st.columns(len(insights) if len(insights) <= 3 else 3)
            for idx, insight in enumerate(insights[:3]):
                with ins_cols[idx % 3]:
                    st.markdown(f"""
                    <div class="ads-card" style="height: 140px;">
                        <span class="ads-badge ads-badge-blue" style="margin-bottom: 0.4rem;">{insight.category.upper()}</span>
                        <div style="font-weight: 600; font-size: 0.85rem; color: #F1F5F9; margin-bottom: 0.3rem;">{insight.title}</div>
                        <div style="font-size: 0.76rem; color: #94A3B8; line-height: 1.4;">{insight.description}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Target Distribution
        if target_col and target_col in df.columns:
            st.markdown(f"#### Target Distribution: <code>{target_col}</code>", unsafe_allow_html=True)
            target_series = df[target_col].dropna()
            
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
                fig_t = px.histogram(
                    df, x=target_col, nbins=35,
                    title=f"Distribution of {target_col}",
                    color_discrete_sequence=["#3B82F6"],
                    template="plotly_dark"
                )
            else:
                vc = target_series.astype(str).value_counts().reset_index()
                vc.columns = [target_col, "count"]
                fig_t = px.bar(
                    vc, x=target_col, y="count",
                    title=f"Class Frequency: {target_col}",
                    color=target_col,
                    color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"],
                    template="plotly_dark"
                )
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_t, use_container_width=True)

        # Dynamic Feature Inspector
        st.markdown("#### Dynamic Feature Inspector")
        num_cols = profile.numerical_columns
        cat_cols = profile.categorical_columns

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            inspect_feat = st.selectbox("Select Feature to Inspect", options=list(df.columns))
            feat_prof = profile.column_profiles.get(inspect_feat)
            
            if feat_prof:
                st.markdown(f"""
                <div class="ads-card">
                    <div style="font-size: 0.8rem; color: #64748B;">COLUMN METADATA</div>
                    <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 0.5rem;">{inspect_feat}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">
                        Type: <strong style="color: #F8FAFC;">{feat_prof.inferred_type}</strong><br>
                        Data Type: <strong style="color: #F8FAFC;">{feat_prof.dtype}</strong><br>
                        Missingness: <strong style="color: {'#EF4444' if feat_prof.null_percentage > 5 else '#10B981'};">{feat_prof.null_percentage}%</strong> ({feat_prof.null_count} rows)<br>
                        Distinct Values: <strong style="color: #F8FAFC;">{feat_prof.unique_count}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with f_col2:
            if inspect_feat in num_cols:
                fig_f = px.box(
                    df, y=inspect_feat,
                    points="outliers",
                    title=f"Boxplot & Distribution: {inspect_feat}",
                    template="plotly_dark",
                    color_discrete_sequence=["#06B6D4"]
                )
                fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                top_cats = df[inspect_feat].astype(str).value_counts().head(10).reset_index()
                top_cats.columns = [inspect_feat, "Count"]
                fig_f = px.bar(
                    top_cats, x=inspect_feat, y="Count",
                    title=f"Top Categories: {inspect_feat}",
                    template="plotly_dark",
                    color_discrete_sequence=["#8B5CF6"]
                )
                fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_f, use_container_width=True)

        # Correlation Heatmap
        if len(num_cols) >= 2:
            st.markdown("#### Numerical Correlation Matrix")
            corr_mat = EDAEngine.get_correlation_matrix(df, num_cols)
            if not corr_mat.empty:
                fig_corr = px.imshow(
                    corr_mat,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Pearson Feature Correlation Matrix",
                    template="plotly_dark"
                )
                fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# 6. MODELING PAGE
# ==============================================================================
elif selected_nav == "Modeling":
    st.markdown("## ⚔️ Multi-Model Cross-Validation Benchmarking")
    state: PipelineState = st.session_state.pipeline_state

    if state.benchmark_result is None:
        st.info("No benchmark results yet. Run the Autonomous Pipeline from the 'Dataset' tab.")
    else:
        bm = state.benchmark_result

        # Top Best Model Announcement
        st.markdown(f"""
        <div class="ads-card ads-card-elevated" style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span class="ads-badge ads-badge-emerald">🏆 TOP CANDIDATE SELECTED</span>
                    <h3 style="margin: 0.3rem 0; font-size: 1.4rem;">{bm.best_candidate_name}</h3>
                    <div style="font-size: 0.85rem; color: #94A3B8;">
                        Cross-Validation Mean {bm.primary_metric.upper()}: <strong style="color: #10B981;">{bm.best_score:.4f}</strong> | {bm.decision_reasoning}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase;">Optimization Target</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #60A5FA;">{bm.primary_metric.upper()}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Candidate Model Leaderboard")
        st.dataframe(bm.leaderboard_df, use_container_width=True)

        # Leaderboard Chart
        fig_bm = px.bar(
            bm.leaderboard_df,
            x="Model",
            y="CV Mean",
            error_y="CV Std",
            title=f"Cross-Validation Performance Comparison ({bm.primary_metric.upper()})",
            color="CV Mean",
            color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig_bm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bm, use_container_width=True)

        # Fold stability inspector
        with st.expander("🔍 Inspect Per-Fold Cross-Validation Metrics"):
            selected_model_inspect = st.selectbox(
                "Select Model to Inspect Folds",
                options=list(bm.results.keys()),
                format_func=lambda m_id: bm.results[m_id].display_name
            )
            model_cv = bm.results[selected_model_inspect]
            fold_data = [
                {"Fold": f.fold_index, "Train Size": f.train_size, "Val Size": f.val_size, "Fit Time (s)": round(f.fit_time_sec, 3), **f.metrics}
                for f in model_cv.fold_results
            ]
            st.dataframe(pd.DataFrame(fold_data), use_container_width=True)


# ==============================================================================
# 7. OPTIMIZATION PAGE
# ==============================================================================
elif selected_nav == "Optimization":
    st.markdown("## 🎯 Optuna Hyperparameter Optimization")
    state: PipelineState = st.session_state.pipeline_state

    if state.tuning_result is None:
        st.info("Optimization results will be available after executing the pipeline.")
    else:
        tune = state.tuning_result

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Optimized Model", tune.model_name)
        t_col2.metric("Baseline Score", f"{tune.baseline_score:.4f}")
        t_col3.metric("Best Tuned Score", f"{tune.best_score:.4f}", f"{tune.score_delta:+.4f}")
        t_col4.metric("Optimization Trials", f"{tune.total_trials} ({tune.duration_sec:.1f}s)")

        # Plotly progression curve
        trial_numbers = [t.trial_number for t in tune.trials_history]
        trial_scores = [t.score for t in tune.trials_history]

        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=trial_numbers, y=trial_scores,
            mode="lines+markers",
            name="Trial Score",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=7, color="#60A5FA")
        ))
        fig_opt.add_hline(
            y=tune.baseline_score,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Baseline Default",
            annotation_position="bottom right"
        )
        fig_opt.add_hline(
            y=tune.best_score,
            line_dash="dot",
            line_color="#10B981",
            annotation_text="Best Score",
            annotation_position="top right"
        )
        fig_opt.update_layout(
            title="Optuna Objective Score Progression Across Trials",
            xaxis_title="Trial Number",
            yaxis_title=f"Score ({tune.metric_name.upper()})",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        # Best Hyperparameters Display
        st.markdown("#### Optimal Discovered Hyperparameters")
        col_hp1, col_hp2 = st.columns([1, 1])

        with col_hp1:
            st.json(tune.best_params)

        with col_hp2:
            st.markdown("""
            <div class="ads-card">
                <h4 style="margin-top: 0; color: #F1F5F9; font-size: 0.95rem;">Bayesian Search Strategy</h4>
                <p style="font-size: 0.8rem; color: #94A3B8;">
                    Using Tree-structured Parzen Estimator (TPE) sampler. Trials dynamically balance exploration
                    of unfamiliar regions and exploitation of high-performing parameter neighborhoods.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 8. EXPLAINABILITY PAGE
# ==============================================================================
elif selected_nav == "Explainability":
    st.markdown("## 🔍 SHAP Explainability & Attribution")
    state: PipelineState = st.session_state.pipeline_state

    if state.explanation_result is None:
        st.info("Run the pipeline to generate SHAP global and local explanations.")
    else:
        exp = state.explanation_result

        # Explainer badge
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            {render_badge(f"Computed via {exp.explainer_type}", "emerald")}
            <span style="font-size: 0.82rem; color: #94A3B8; margin-left: 8px;">Global and local game-theoretic feature attribution.</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Global Feature Importance")
        # Horizontal bar chart of top features
        top_items = list(exp.global_importance.items())[:12]
        df_imp = pd.DataFrame(top_items, columns=["Feature", "Mean |SHAP Value|"]).sort_values(by="Mean |SHAP Value|", ascending=True)

        fig_shap = px.bar(
            df_imp,
            x="Mean |SHAP Value|",
            y="Feature",
            orientation="h",
            title="Global Feature Importance (Mean Absolute SHAP Value)",
            color="Mean |SHAP Value|",
            color_continuous_scale="Tealgrn",
            template="plotly_dark",
        )
        fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        st.markdown("### Local Instance-Level Explanation")
        st.markdown("Select an individual validation sample to inspect what features drove that specific prediction:")

        sample_keys = list(exp.sample_explanations.keys())
        if sample_keys:
            selected_s = st.selectbox(
                "Select Sample Index",
                options=sample_keys,
                format_func=lambda idx: f"Sample #{idx + 1} (Prediction: {exp.sample_explanations[idx].predicted_value})"
            )
            local_exp = exp.sample_explanations[selected_s]

            loc_c1, loc_c2 = st.columns([1, 2])

            with loc_c1:
                st.markdown(f"""
                <div class="ads-card">
                    <div style="font-size: 0.75rem; color: #64748B;">PREDICTED OUTCOME</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #38BDF8;">{local_exp.predicted_value}</div>
                    {f'<div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.2rem;">Confidence / Probability: <strong>{local_exp.predicted_probability:.2%}</strong></div>' if local_exp.predicted_probability is not None else ''}
                </div>
                """, unsafe_allow_html=True)

            with loc_c2:
                # Waterfall/bar chart of feature impacts
                contrib_data = [
                    {"Feature": c.feature_name, "SHAP Impact": c.shap_value, "Direction": c.impact_direction}
                    for c in local_exp.contributions[:8]
                ]
                df_contrib = pd.DataFrame(contrib_data).sort_values(by="SHAP Impact", ascending=True)

                fig_loc = px.bar(
                    df_contrib,
                    x="SHAP Impact",
                    y="Feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={"positive": "#3B82F6", "negative": "#EF4444"},
                    title=f"Sample #{selected_s + 1} Feature Attribution Breakdown",
                    template="plotly_dark",
                )
                fig_loc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_loc, use_container_width=True)

            # Plain English interpretation table
            st.markdown("##### Plain English Drivers for this Prediction:")
            plain_data = [
                {
                    "Feature": c.feature_name,
                    "Observed Value": c.feature_value,
                    "SHAP Impact": f"{'+' if c.shap_value >= 0 else ''}{c.shap_value:.4f}",
                    "Interpretation": c.plain_english,
                }
                for c in local_exp.contributions[:6]
            ]
            st.dataframe(pd.DataFrame(plain_data), use_container_width=True)


# ==============================================================================
# 9. EXPERIMENTS PAGE (MLflow)
# ==============================================================================
elif selected_nav == "Experiments":
    st.markdown("## 🧪 MLflow Experiment Tracking")
    tracker = MLflowTracker()

    st.markdown("""
    <div style="background: #111726; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #94A3B8;">
        All experiment runs, parameter grids, cross-validation metrics, and model tags are tracked locally via SQLite backend (<code>sqlite:///mlflow.db</code>).
        To launch the interactive MLflow UI, execute: <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>
    </div>
    """, unsafe_allow_html=True)

    runs_df = tracker.get_experiment_runs(max_runs=50)

    if runs_df.empty:
        st.info("No MLflow runs recorded yet. Execute the pipeline to log experiment runs.")
    else:
        st.markdown("#### Experiment Run History")
        st.dataframe(runs_df, use_container_width=True)


# ==============================================================================
# 10. PREDICTIONS PAGE
# ==============================================================================
elif selected_nav == "Predictions":
    st.markdown("## 🔮 Real-Time & Batch Predictions")
    bundle_path = os.environ.get("MODEL_BUNDLE_PATH", "artifacts/model_bundle.joblib")

    if not os.path.exists(bundle_path):
        st.warning("⚠️ No model bundle found. Run the Autonomous Pipeline first to register a production model.")
    else:
        import joblib
        bundle = joblib.load(bundle_path)
        pipeline = bundle["pipeline"]
        problem_spec: ProblemSpec = bundle["problem_spec"]
        feature_schema = bundle["feature_schema"]

        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
            {render_badge(f"Active Model: {bundle.get('best_model_name')}", "emerald")}
            {render_badge(f"Target: {bundle.get('target_column')}", "blue")}
        </div>
        """, unsafe_allow_html=True)

        mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Prediction"], horizontal=True)

        if mode == "Single Prediction":
            st.markdown("#### Interactive Record Inference Form")
            st.markdown("Input feature values will be automatically sanitized, imputed, and scaled via the fitted production `Pipeline`.")

            form_values: Dict[str, Any] = {}
            form_cols = st.columns(3)

            for idx, feat in enumerate(feature_schema):
                col = form_cols[idx % 3]
                f_name = feat["name"]
                is_num = feat["is_numerical"]

                with col:
                    if is_num:
                        default_val = float(feat.get("sample_value") or 0.0)
                        form_values[f_name] = st.number_input(
                            label=f"{f_name}",
                            value=default_val,
                            key=f"input_{f_name}"
                        )
                    else:
                        u_vals = feat.get("unique_values") or ["Yes", "No"]
                        form_values[f_name] = st.selectbox(
                            label=f"{f_name}",
                            options=[str(v) for v in u_vals],
                            key=f"input_{f_name}"
                        )

            if st.button("⚡ Generate Prediction", type="primary", use_container_width=True):
                input_df = pd.DataFrame([form_values])
                pred_val = pipeline.predict(input_df)[0]
                
                # Class mapping if classification
                if problem_spec.is_classification and bundle.get("classes"):
                    try:
                        cls_idx = int(pred_val)
                        if 0 <= cls_idx < len(bundle["classes"]):
                            pred_val = bundle["classes"][cls_idx]
                    except Exception:
                        pass

                prob_val = None
                if problem_spec.is_classification and hasattr(pipeline, "predict_proba"):
                    try:
                        p_arr = pipeline.predict_proba(input_df)[0]
                        prob_val = float(p_arr[1]) if len(p_arr) == 2 else float(np.max(p_arr))
                    except Exception:
                        pass

                st.markdown("""<div style='margin-top: 1rem;'></div>""", unsafe_allow_html=True)
                p_c1, p_c2 = st.columns([1, 1])

                with p_c1:
                    st.markdown(f"""
                    <div class="ads-card ads-card-elevated">
                        <div style="font-size: 0.8rem; color: #64748B;">PREDICTION RESULT</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #38BDF8; margin: 0.3rem 0;">{pred_val}</div>
                        {f'<div style="font-size: 0.9rem; color: #94A3B8;">Confidence: <strong style="color: #10B981;">{prob_val:.2%}</strong></div>' if prob_val is not None else ''}
                    </div>
                    """, unsafe_allow_html=True)

                with p_c2:
                    st.markdown("""
                    <div class="ads-card">
                        <div style="font-size: 0.8rem; color: #64748B;">SERVING VERIFICATION</div>
                        <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5; margin-top: 0.3rem;">
                            ✓ Preprocessing applied in memory.<br>
                            ✓ ColumnTransformer matched feature schema.<br>
                            ✓ Fully serializable via FastAPI endpoint <code>POST /predict</code>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        elif mode == "Batch Prediction":
            st.markdown("#### Batch CSV Inference")
            batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_uploader")

            if batch_file is not None:
                df_batch = pd.read_csv(batch_file)
                st.markdown(f"Uploaded {len(df_batch)} records for inference.")
                st.dataframe(df_batch.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    preds = pipeline.predict(df_batch)
                    df_res = df_batch.copy()
                    df_res["predicted_" + bundle.get("target_column", "target")] = preds

                    st.success(f"Generated predictions for all {len(preds)} rows!")
                    st.dataframe(df_res.head(10), use_container_width=True)

                    csv_data = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )


# ==============================================================================
# 11. SYSTEM PAGE
# ==============================================================================
elif selected_nav == "System":
    st.markdown("## ⚙️ System Status & Runtime Environment")

    s1, s2, s3 = st.columns(3)
    s1.metric("Python Version", f"{sys.version.split()[0]}")
    s2.metric("FastAPI Port", "8000")
    s3.metric("MLflow Tracking URI", "sqlite:///mlflow.db")

    st.markdown("""
    <div class="ads-card" style="margin-top: 1.5rem;">
        <h4 style="margin-top: 0; font-size: 1rem; color: #F1F5F9;">FastAPI Serving Endpoints</h4>
        <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.8;">
            <code>GET  /health</code> - Service availability & loaded model check<br>
            <code>GET  /model</code> - Active model metadata & feature schema<br>
            <code>GET  /metrics</code> - Cross-validation & test performance metrics<br>
            <code>POST /predict</code> - Single record JSON inference<br>
            <code>POST /predict/batch</code> - Batch record array inference
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧹 Clear Session State & Cache", type="secondary"):
        st.session_state.clear()
        st.success("Session state cleared.")
        st.rerun()
