from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model_loader import ModelLoader
from backend.pipeline import BackendPipelineService


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "backend" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Automated Data Science & Analytics Studio API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_service = BackendPipelineService(base_dir=BASE_DIR)
model_loader = ModelLoader(models_dir=str(BASE_DIR / "models"))


class TrainRequest(BaseModel):
    target_column: str = Field(..., min_length=1)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Automated Data Science & Analytics Studio backend is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported for the /upload endpoint.")

    try:
        content = await file.read()
        result = pipeline_service.save_uploaded_dataset(file.filename, content)
        return {"message": "Dataset uploaded successfully.", **result}
    except Exception as exc:
        LOGGER.exception("Dataset upload failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train_model(request: TrainRequest) -> Dict[str, Any]:
    try:
        results = pipeline_service.train_from_uploaded_dataset(request.target_column)
        primary_metric = _select_primary_metric(results.get("metrics", {}))
        return {
            "message": "Training completed successfully.",
            "problem_type": results.get("problem_type"),
            "best_model": results.get("best_model"),
            "accuracy": primary_metric,
            "results": results,
        }
    except Exception as exc:
        LOGGER.exception("Training failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    try:
        return model_loader.predict(request.records)
    except Exception as exc:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask_question(request: AskRequest) -> Dict[str, Any]:
    try:
        return pipeline_service.ask_question(request.query)
    except Exception as exc:
        LOGGER.exception("Question answering failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/results")
def get_results() -> Dict[str, Any]:
    try:
        return pipeline_service.get_results()
    except Exception as exc:
        LOGGER.exception("Fetching results failed.")
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _select_primary_metric(metrics: Dict[str, Any]) -> float | None:
    for key in ("accuracy", "f1", "r2", "rmse"):
        value = metrics.get(key)
        if value is not None:
            return value
    return None
