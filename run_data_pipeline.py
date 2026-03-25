from __future__ import annotations

import argparse
import logging

from src.data_processing.data_loader import DataLoader
from src.ml_pipeline.model_trainer import AutoMLSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production AutoML pipeline on a dataset.")
    parser.add_argument("--data", default="data/2019.csv", help="Path to the input dataset.")
    parser.add_argument("--target", default="Score", help="Name of the target column.")
    parser.add_argument("--output-dir", default="reports", help="Directory used for generated artifacts.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    df = DataLoader(args.data).load_data()
    automl = AutoMLSystem(output_dir=args.output_dir)
    automl.run(df=df, target_column=args.target)


if __name__ == "__main__":
    main()
