from __future__ import annotations

import logging

from src.data_processing.data_loader import DataLoader
from src.ml_pipeline.model_trainer import AutoMLSystem


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    df = DataLoader("data/Titanic-Dataset.csv").load_data()
    automl = AutoMLSystem(output_dir="reports")
    automl.run(df=df, target_column="Survived")


if __name__ == "__main__":
    main()
