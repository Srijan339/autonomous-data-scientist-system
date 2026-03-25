from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from src.data_processing.data_loader import DataLoader
from src.ml_pipeline.model_trainer import AutoMLSystem


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

st.set_page_config(page_title="Autonomous AutoML System", layout="wide")
st.title("Autonomous Data Scientist AutoML System")
st.write("Upload a tabular dataset, select a target column, and run the production AutoML pipeline.")

uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "json", "jsonl", "xlsx", "xls", "parquet", "feather"],
)

if uploaded_file is not None:
    try:
        df = DataLoader().load_data(source=uploaded_file)
        st.subheader("Dataset preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select target column", options=list(df.columns))
        output_dir = st.text_input("Artifacts output directory", value="reports")

        if st.button("Run AutoML Pipeline", type="primary"):
            with st.spinner("Running validation, EDA, feature engineering, training, tuning, and explainability..."):
                automl = AutoMLSystem(output_dir=output_dir)
                results = automl.run(df=df, target_column=target_column)

            st.success("AutoML pipeline completed.")
            st.subheader("Results")
            st.json(
                {
                    "problem_type": results["problem_type"],
                    "best_model": results["best_model"],
                    "metrics": results["metrics"],
                }
            )

            st.subheader("Leaderboard")
            st.dataframe(results["leaderboard"])

            st.subheader("Top feature importance")
            st.dataframe(results["feature_importance"])

            shap_path = Path(results["artifacts"]["metrics"])
            metrics_payload = json.loads(shap_path.read_text(encoding="utf-8"))
            shap_info = metrics_payload.get("shap", {})
            if shap_info.get("enabled") and shap_info.get("summary_plot"):
                st.subheader("SHAP summary plot")
                st.image(shap_info["summary_plot"])

            st.subheader("Saved artifacts")
            st.json(results["artifacts"])

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
