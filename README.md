# Automated Data Science & Analytics Studio

An end-to-end AI data scientist web app built with FastAPI, Streamlit, and a production-style AutoML pipeline.

## Overview

This project turns a tabular dataset into a working analysis workspace:

- Upload a CSV dataset
- Validate and recommend target columns
- Run cleaning, EDA, feature engineering, model selection, tuning, and evaluation
- Generate explainability outputs and structured insights
- Save reusable models and artifacts
- Ask questions against a retrieval-backed knowledge layer
- Run live predictions from the trained model

## Product Architecture

The current system includes:

- `backend/`
  - FastAPI application
  - upload, train, predict, results, and ask endpoints
  - artifact caching and persistence
- `frontend/`
  - Streamlit product UI
  - upload flow, target selection, metrics, charts, assistant, predictions
- `src/data_processing/`
  - loading, validation, cleaning, profiling
- `src/analytics/`
  - EDA engine
  - insight generation
- `src/ml_pipeline/`
  - problem detection
  - feature engineering
  - model training and evaluation
- `src/rag/`
  - local vector store
  - retriever
  - question-answering layer

## Features

- Automatic problem detection for classification vs regression
- Target validation with blocked identifier/index-like columns
- Leakage-safe preprocessing with sklearn pipelines
- Missing value imputation, encoding, and scaling
- Domain and interaction feature creation
- Model selection across classification and regression algorithms
- Cross-validation and RandomizedSearchCV tuning
- Explainability with feature importance and SHAP summary plots
- Structured insight generation and knowledge-base creation
- Retrieval-backed assistant over stored analysis artifacts
- Saved model, pipeline, metrics, leaderboard, insights, and knowledge files
- Product-style Streamlit interface with backend health handling

## Project Structure

```text
.
├── backend/
│   ├── main.py
│   ├── model_loader.py
│   ├── pipeline.py
│   ├── logs/
│   └── storage/
├── frontend/
│   ├── app.py
│   └── logs/
├── models/
├── reports/
├── src/
│   ├── analytics/
│   ├── data_processing/
│   ├── ml_pipeline/
│   └── rag/
├── data/
├── requirements.txt
└── README.md
```

## Backend API

Base URL:

```text
http://127.0.0.1:8000
```

Endpoints:

- `GET /health`
  - backend health check
- `POST /upload`
  - upload CSV dataset
  - returns preview, shape, recommended targets, blocked targets
- `POST /train`
  - runs the full AutoML workflow for the selected target
- `POST /predict`
  - predicts on JSON records using the saved model
- `GET /results`
  - returns latest metrics, leaderboard, insights, artifacts, and schema
- `POST /ask`
  - answers dataset/model questions using the knowledge layer

## Installation

```bash
git clone https://github.com/Srijan339/autonomous-data-scientist-system.git
cd autonomous-data-scientist-system
pip install -r requirements.txt
```

## Run Locally

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload
```

Start the Streamlit frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

Open:

- Frontend: `http://127.0.0.1:8501`
- Backend: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

## Typical User Flow

1. Open the Streamlit frontend
2. Upload a CSV dataset
3. Pick a recommended target column
4. Click `Run AI Analysis`
5. Review:
   - dataset preview
   - leaderboard
   - metrics
   - EDA charts
   - SHAP/explainability outputs
   - structured insights
6. Ask questions in the assistant tab
7. Test predictions in the prediction console

## Saved Artifacts

After training, the app saves artifacts under `models/` and `reports/`, including:

- trained model
- preprocessing pipeline
- metrics JSON
- leaderboard JSON
- feature importance JSON
- EDA summary
- insights JSON
- knowledge text file
- vector store
- SHAP summary plot
- confusion matrix or regression evaluation chart

## Example Datasets

Bundled sample datasets are available in `data/`, including:

- Titanic classification dataset
- housing/regression datasets
- additional tabular CSV files

## Notes

- The current assistant uses a local retrieval layer for reliability in offline/local environments.
- Only CSV upload is enabled in the web upload endpoint right now.
- The app is designed for local development first and can be deployed after repository cleanup and hosting configuration.

## Author

Srijan K
