from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "Automated Data Science & Analytics Studio"


st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 18% 16%, rgba(62, 118, 255, 0.18), transparent 24%),
            radial-gradient(circle at 82% 12%, rgba(255, 128, 66, 0.16), transparent 22%),
            linear-gradient(180deg, #07111f 0%, #0c1728 35%, #0e1724 100%);
        color: #eff4ff;
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }
    .hero-shell, .panel-shell, .metric-shell {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(10, 16, 28, 0.78);
        border-radius: 28px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.28);
        backdrop-filter: blur(12px);
    }
    .hero-shell { padding: 1.8rem 1.8rem 1.4rem 1.8rem; }
    .panel-shell { padding: 1rem 1.2rem; }
    .metric-shell { padding: 1rem 1.2rem; min-height: 120px; }
    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        background: rgba(78, 132, 255, 0.16);
        color: #b7ceff;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 3.3rem;
        line-height: 1.02;
        color: #f8fbff;
        font-weight: 800;
        max-width: 760px;
        margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        color: #b7c3d9;
        font-size: 1.06rem;
        line-height: 1.65;
        max-width: 760px;
        margin-bottom: 0;
    }
    .section-title {
        color: #ffffff;
        font-weight: 750;
        font-size: 1.2rem;
        margin-bottom: 0.2rem;
    }
    .section-copy {
        color: #98a8c2;
        font-size: 0.94rem;
        margin-bottom: 0;
    }
    .metric-kicker {
        color: #88a2c8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #f8fbff;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.35rem 0 0.2rem 0;
    }
    .metric-note {
        color: #9eb1cf;
        font-size: 0.92rem;
    }
    .stDataFrame, .stTable, div[data-testid="stJson"] {
        border-radius: 18px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <span class="eyebrow">AutoML</span>
            <span class="eyebrow">Insights</span>
            <span class="eyebrow">Knowledge Assistant</span>
            <div class="hero-title">{APP_NAME}</div>
            <p class="hero-subtitle">
                Upload a dataset, validate the right target, run a production-grade ML workflow,
                inspect explainability and insight outputs, then ask follow-up questions against a retrieval-backed knowledge layer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="panel-shell">
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: Any, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
            <div class="metric-kicker">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_error(exc: requests.RequestException) -> str:
    if exc.response is not None:
        try:
            payload = exc.response.json()
            return payload.get("detail", str(exc))
        except Exception:
            return f"{exc.response.status_code}: {exc.response.text}"
    return str(exc)


def backend_request(method: str, path: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=900, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(extract_error(exc))
        return None


def backend_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def upload_dataset(uploaded_file) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/csv")}
    payload = backend_request("POST", "/upload", files=files)
    if payload:
        st.session_state["upload_response"] = payload
        st.session_state.pop("results_response", None)
        st.success("Dataset uploaded and profiled successfully.")


def train_model(target_column: str) -> None:
    payload = backend_request("POST", "/train", json={"target_column": target_column})
    if payload:
        st.session_state["results_response"] = payload["results"]
        st.success("AI analysis completed successfully.")


def ask_assistant(question: str) -> None:
    payload = backend_request("POST", "/ask", json={"query": question})
    if payload:
        st.session_state["assistant_response"] = payload


def submit_prediction(record: Dict[str, Any]) -> None:
    payload = backend_request("POST", "/predict", json={"records": [record]})
    if payload:
        st.session_state["prediction_response"] = payload


def build_prediction_record(schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    cols = st.columns(2)
    for index, feature in enumerate(schema):
        name = feature["name"]
        dtype = str(feature.get("dtype", "object")).lower()
        sample_values = [value for value in feature.get("sample_values", []) if value is not None]
        example = feature.get("example")
        container = cols[index % 2]

        with container:
            if "int" in dtype or "float" in dtype:
                default_value = float(example) if example is not None else 0.0
                record[name] = st.number_input(name, value=default_value, key=f"pred_{name}")
            elif sample_values and len(sample_values) <= 8:
                options = [str(value) for value in sample_values]
                record[name] = st.selectbox(name, options=options, key=f"pred_{name}")
            else:
                record[name] = st.text_input(name, value="" if example is None else str(example), key=f"pred_{name}")
    return record


render_hero()

status_col, notes_col = st.columns([1, 2])
with status_col:
    if backend_health():
        st.success("Backend connection healthy")
    else:
        st.error("Backend unavailable. Start FastAPI on http://127.0.0.1:8000 first.")

with notes_col:
    st.info("This studio now includes target validation, AutoML, explainability, cached artifacts, and a retrieval-backed assistant.")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    render_section("1. Upload Dataset", "Send the dataset to the backend. The system will profile columns and recommend only sensible target choices.")
    if st.button("Upload Dataset", type="primary", use_container_width=True, disabled=not backend_health()):
        with st.spinner("Uploading and profiling dataset..."):
            upload_dataset(uploaded_file)

upload_response = st.session_state.get("upload_response")
results = st.session_state.get("results_response")

if upload_response:
    preview_df = pd.DataFrame(upload_response.get("preview", []))
    recommended_targets = upload_response.get("recommended_targets", [])
    blocked_targets = upload_response.get("blocked_targets", [])

    left, right = st.columns([1.8, 1])
    with left:
        render_section("Dataset Preview", "Inspect a sample of the uploaded dataset before training.")
        st.dataframe(preview_df, use_container_width=True)

    with right:
        render_section("Dataset Profile", "Quick metadata extracted at upload time.")
        shape = upload_response.get("shape", {})
        render_metric("Rows", shape.get("rows", 0), "Observed rows in uploaded dataset")
        render_metric("Columns", shape.get("columns", 0), "Observed feature columns before training")

    config_col, warning_col = st.columns([1.1, 0.9])
    with config_col:
        render_section("2. Target Selection", "Only recommended targets are selectable. Identifier-like and unstable columns are blocked.")
        candidate_names = [item["column"] for item in recommended_targets]
        if candidate_names:
            target_column = st.selectbox("Choose target column", options=candidate_names)
            if st.button("Run AI Analysis", use_container_width=True, disabled=not backend_health()):
                with st.spinner("Running intelligent data science pipeline..."):
                    train_model(target_column)
        else:
            st.warning("No valid target candidates were found. Review the blocked target list.")

    with warning_col:
        render_section("Blocked Targets", "These columns look like IDs, indexes, monotonic keys, or otherwise weak supervised targets.")
        if blocked_targets:
            st.dataframe(pd.DataFrame(blocked_targets), use_container_width=True)
        else:
            st.success("No blocked targets detected.")

if results:
    summary_cols = st.columns(4)
    metrics = results.get("metrics", {})
    primary_metric_label = "Accuracy" if "accuracy" in metrics else "R2" if "r2" in metrics else "Metric"
    primary_metric_value = metrics.get("accuracy", metrics.get("r2", "N/A"))
    with summary_cols[0]:
        render_metric("Problem Type", results.get("problem_type", "N/A"), "Automatically detected from the selected target")
    with summary_cols[1]:
        render_metric("Best Model", results.get("best_model", "N/A"), "Highest-ranked model after CV and tuning")
    with summary_cols[2]:
        render_metric(primary_metric_label, primary_metric_value, "Primary holdout metric")
    with summary_cols[3]:
        render_metric("Insights", len(results.get("insights", [])), "Structured findings generated from the run")

    workspace_tab, insights_tab, assistant_tab = st.tabs(["Workspace", "Insights", "Assistant"])

    with workspace_tab:
        render_section("Leaderboard", "Candidate models ranked by cross-validation performance.")
        st.dataframe(pd.DataFrame(results.get("leaderboard", [])), use_container_width=True)

        left, right = st.columns(2)
        with left:
            render_section("Metrics", "Detailed evaluation metrics from the trained winner.")
            st.json(metrics)
        with right:
            render_section("Feature Importance", "Top predictive signals from the explainability layer.")
            st.dataframe(pd.DataFrame(results.get("feature_importance", [])), use_container_width=True)

        render_section("Visualization Wall", "EDA and evaluation visuals generated by the pipeline.")
        chart_paths = results.get("artifacts", {}).get("charts", [])
        visual_paths = list(chart_paths)
        for extra in ("confusion_matrix", "actual_vs_predicted"):
            path = results.get("artifacts", {}).get(extra)
            if path:
                visual_paths.append(path)
        shap_path = results.get("artifacts", {}).get("shap_summary")
        if shap_path:
            visual_paths.append(shap_path)

        if visual_paths:
            columns = st.columns(2)
            for index, chart_path in enumerate(visual_paths[:10]):
                with columns[index % 2]:
                    if Path(chart_path).exists():
                        st.image(chart_path, use_container_width=True)
        else:
            st.info("No visuals available yet.")

    with insights_tab:
        render_section("Generated Insights", "Structured findings distilled from data quality checks, model behavior, and target characteristics.")
        for insight in results.get("insights", []):
            st.markdown(f"**{insight['title']}**")
            st.write(insight["detail"])

        render_section("Knowledge Base", "This artifact powers the retrieval-backed assistant.")
        knowledge_path = results.get("artifacts", {}).get("knowledge_base")
        if knowledge_path and Path(knowledge_path).exists():
            st.code(Path(knowledge_path).read_text(encoding="utf-8"), language="text")

    with assistant_tab:
        render_section("Ask The Studio", "Question the trained system about the dataset, model, features, insights, or evaluation results.")
        question = st.text_input("Ask a question", placeholder="Why did the model choose these features?")
        if st.button("Ask Assistant", use_container_width=True):
            with st.spinner("Searching the knowledge base and composing an answer..."):
                ask_assistant(question)

        assistant_response = st.session_state.get("assistant_response")
        if assistant_response:
            st.success(assistant_response.get("answer", ""))
            sources = assistant_response.get("sources", [])
            if sources:
                st.dataframe(pd.DataFrame(sources), use_container_width=True)

        render_section("Prediction Console", "Test a fresh row against the saved model using the persisted pipeline.")
        prediction_schema = results.get("prediction_schema", [])
        if prediction_schema:
            with st.form("prediction_form"):
                prediction_record = build_prediction_record(prediction_schema)
                submitted = st.form_submit_button("Run Prediction")
            if submitted:
                with st.spinner("Scoring row with the saved model..."):
                    submit_prediction(prediction_record)
            if st.session_state.get("prediction_response"):
                st.json(st.session_state["prediction_response"])
        else:
            st.info("Prediction schema will appear after a successful training run.")
