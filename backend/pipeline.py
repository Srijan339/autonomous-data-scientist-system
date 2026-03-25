from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_validator import DataValidator
from src.ml_pipeline.model_trainer import AutoMLSystem
from src.rag.qa_engine import QAAssistant
from src.rag.retriever import Retriever
from src.rag.vector_store import LocalVectorStore


LOGGER = logging.getLogger(__name__)


class BackendPipelineService:
    """Service layer that manages uploads, cached training, and knowledge-backed QA."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.cwd()
        self.storage_dir = self.base_dir / "backend" / "storage"
        self.models_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_metadata_path = self.storage_dir / "dataset_metadata.json"
        self.train_state_path = self.storage_dir / "train_state.json"
        self.vector_store_path = self.models_dir / "knowledge_vector_store.joblib"

    def save_uploaded_dataset(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        suffix = Path(file_name).suffix.lower()
        active_path = self.storage_dir / f"active_dataset{suffix}"
        active_path.write_bytes(file_bytes)

        df = DataLoader(active_path).load_data()
        if df.empty:
            raise ValueError("Uploaded dataset is empty.")

        validator = DataValidator(df)
        target_assessment = validator.assess_target_candidates()
        preview = df.head(10).replace({pd.NA: None}).to_dict(orient="records")
        dataset_hash = hashlib.sha256(file_bytes).hexdigest()

        metadata = {
            "file_name": file_name,
            "stored_path": str(active_path.resolve()),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "preview": preview,
            "dataset_hash": dataset_hash,
            **target_assessment,
        }
        self.dataset_metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        LOGGER.info("Uploaded dataset stored at %s", active_path)
        return metadata

    def train_from_uploaded_dataset(self, target_column: str) -> Dict[str, Any]:
        upload_metadata = self._read_json(self.dataset_metadata_path)
        blocked_targets = {item["column"]: item["reason"] for item in upload_metadata.get("blocked_targets", [])}
        if target_column in blocked_targets:
            raise ValueError(f"'{target_column}' is not a valid target: {blocked_targets[target_column]}")

        if self._can_reuse_cached_results(upload_metadata.get("dataset_hash"), target_column):
            LOGGER.info("Reusing cached training artifacts for target '%s'.", target_column)
            return self.get_results()

        dataset_path = self._get_active_dataset_path()
        df = DataLoader(dataset_path).load_data()
        automl = AutoMLSystem(output_dir=str(self.reports_dir))
        results = automl.run(df=df, target_column=target_column)
        self._sync_model_artifacts(results["artifacts"])
        self._persist_train_state(upload_metadata.get("dataset_hash"), target_column)
        self._build_knowledge_store()
        return self.get_results()

    def ask_question(self, query: str) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Question cannot be empty.")
        if not self.vector_store_path.exists():
            raise FileNotFoundError("No trained knowledge base found. Train the system before asking questions.")

        store = LocalVectorStore.load(self.vector_store_path)
        retrieved = Retriever(store).retrieve(query=query, top_k=4)
        return QAAssistant().answer(query=query, retrieved_chunks=retrieved, results_payload=self.get_results())

    def get_results(self) -> Dict[str, Any]:
        metrics_payload = self._read_json(self.reports_dir / "metrics.json")
        leaderboard_payload = self._read_json(self.reports_dir / "leaderboard.json")
        feature_importance_payload = self._read_json(self.reports_dir / "feature_importance.json")
        profile_payload = self._read_json(self.reports_dir / "data_profile.json")
        metadata_payload = self._read_json(self.reports_dir / "model_metadata.json")
        insights_payload = self._read_json(self.reports_dir / "insights.json")

        return {
            "problem_type": metadata_payload.get("problem_type"),
            "best_model": metadata_payload.get("best_model"),
            "target_column": metadata_payload.get("target_column"),
            "metrics": metrics_payload.get("metrics", {}),
            "shap": metrics_payload.get("shap", {}),
            "leaderboard": leaderboard_payload,
            "feature_importance": feature_importance_payload,
            "eda_summary": profile_payload.get("eda_summary", {}),
            "data_profile": profile_payload.get("profile", {}),
            "prediction_schema": metadata_payload.get("prediction_schema", []),
            "insights": insights_payload,
            "artifacts": self._artifact_manifest(),
        }

    def _build_knowledge_store(self) -> None:
        results = self.get_results()
        knowledge_path = self.reports_dir / "knowledge.txt"
        knowledge_text = knowledge_path.read_text(encoding="utf-8") if knowledge_path.exists() else ""
        insights = results.get("insights", [])

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for insight in insights:
            text = f"{insight.get('title', 'Insight')}: {insight.get('detail', '')}"
            texts.append(text)
            metadatas.append({"section": "insight"})

        if knowledge_text.strip():
            for chunk in self._chunk_text(knowledge_text):
                texts.append(chunk)
                metadatas.append({"section": "knowledge"})

        metrics_line = ", ".join(f"{key}={value}" for key, value in results.get("metrics", {}).items())
        texts.append(
            f"Problem type: {results.get('problem_type')}. Best model: {results.get('best_model')}. Metrics: {metrics_line}."
        )
        metadatas.append({"section": "summary"})

        store = LocalVectorStore.from_texts(texts=texts, metadatas=metadatas)
        store.save(self.vector_store_path)

    def _sync_model_artifacts(self, artifact_paths: Dict[str, str]) -> None:
        mapping = {
            "model": self.models_dir / "model.pkl",
            "preprocessing_pipeline": self.models_dir / "pipeline.pkl",
            "model_metadata": self.models_dir / "model_metadata.json",
        }
        for key, destination in mapping.items():
            source = Path(artifact_paths[key])
            shutil.copy2(source, destination)
            LOGGER.info("Copied %s to %s", source, destination)

    def _artifact_manifest(self) -> Dict[str, Any]:
        chart_paths = []
        eda_summary = self._read_json(self.reports_dir / "data_profile.json").get("eda_summary", {})
        for path in eda_summary.get("chart_paths", []):
            chart_paths.append(str(Path(path).resolve()))

        manifest = {
            "model": str((self.models_dir / "model.pkl").resolve()),
            "pipeline": str((self.models_dir / "pipeline.pkl").resolve()),
            "metadata": str((self.models_dir / "model_metadata.json").resolve()),
            "metrics": str((self.reports_dir / "metrics.json").resolve()),
            "leaderboard": str((self.reports_dir / "leaderboard.json").resolve()),
            "feature_importance": str((self.reports_dir / "feature_importance.json").resolve()),
            "eda_summary": str((self.reports_dir / "eda_summary.json").resolve()),
            "insights": str((self.reports_dir / "insights.json").resolve()),
            "knowledge_base": str((self.reports_dir / "knowledge.txt").resolve()),
            "vector_store": str(self.vector_store_path.resolve()),
            "shap_summary": str((self.reports_dir / "shap_summary.png").resolve()),
            "charts": chart_paths,
        }

        for optional_name in ("confusion_matrix.png", "actual_vs_predicted.png"):
            path = self.reports_dir / optional_name
            if path.exists():
                manifest[path.stem] = str(path.resolve())

        return manifest

    def _persist_train_state(self, dataset_hash: str | None, target_column: str) -> None:
        payload = {"dataset_hash": dataset_hash, "target_column": target_column}
        self.train_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _can_reuse_cached_results(self, dataset_hash: str | None, target_column: str) -> bool:
        if not dataset_hash or not self.train_state_path.exists():
            return False
        if not (self.reports_dir / "metrics.json").exists():
            return False
        state = self._read_json(self.train_state_path)
        return state.get("dataset_hash") == dataset_hash and state.get("target_column") == target_column

    def _get_active_dataset_path(self) -> Path:
        metadata = self._read_json(self.dataset_metadata_path)
        stored_path = metadata.get("stored_path")
        if not stored_path:
            raise FileNotFoundError("No uploaded dataset found. Upload a dataset first.")
        path = Path(stored_path)
        if not path.exists():
            raise FileNotFoundError("Stored uploaded dataset could not be found.")
        return path

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any] | List[Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
