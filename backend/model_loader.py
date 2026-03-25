from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


class ModelLoader:
    """Load the trained model and perform predictions."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "model.pkl"
        self.metadata_path = self.models_dir / "model_metadata.json"

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        model = self._load_model()
        metadata = self._load_metadata()

        if not records:
            raise ValueError("Prediction input is empty.")

        frame = pd.DataFrame(records)
        expected_columns = metadata.get("prediction_columns", [])
        missing_columns = [column for column in expected_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing prediction columns: {missing_columns}")

        frame = frame.reindex(columns=expected_columns).replace({"": None})
        predictions = model.predict(frame)
        return {
            "predictions": predictions.tolist(),
            "problem_type": metadata.get("problem_type"),
            "best_model": metadata.get("best_model"),
        }

    def get_prediction_schema(self) -> List[Dict[str, Any]]:
        metadata = self._load_metadata()
        return metadata.get("prediction_schema", [])

    def get_metadata(self) -> Dict[str, Any]:
        return self._load_metadata()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError("Trained model not found. Train the model first.")
        return joblib.load(self.model_path)

    def _load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError("Model metadata not found. Train the model first.")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))
