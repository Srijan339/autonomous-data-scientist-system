from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger(__name__)


class EDAEngine:
    """Generate compact EDA plots and summaries for arbitrary tabular datasets."""

    def __init__(self, df: pd.DataFrame, output_folder: str = "reports", target_column: Optional[str] = None):
        self.df = df.copy()
        self.output_folder = Path(output_folder)
        self.target_column = target_column
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run_full_eda(self) -> Dict[str, object]:
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        summary = self.generate_summary_report()
        chart_paths: List[str] = []

        if numeric_columns:
            correlation_path = self._save_correlation_heatmap(numeric_columns)
            if correlation_path:
                chart_paths.append(str(correlation_path.resolve()))
            chart_paths.extend(self._save_numeric_distributions(numeric_columns[: min(6, len(numeric_columns))]))

        summary["chart_paths"] = chart_paths

        return summary

    def generate_summary_report(self) -> Dict[str, object]:
        summary = {
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "columns": list(self.df.columns),
            "missing_values": self.df.isna().sum().to_dict(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "numeric_summary": self.df.describe(include=["number"]).fillna(0).to_dict() if not self.df.empty else {},
            "categorical_summary": self._categorical_summary(),
        }

        if self.target_column and self.target_column in self.df.columns:
            summary["target_summary"] = {
                "nunique": int(self.df[self.target_column].nunique(dropna=True)),
                "missing": int(self.df[self.target_column].isna().sum()),
            }

        summary_path = self.output_folder / "eda_summary.json"
        with summary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, default=str)

        text_path = self.output_folder / "EDA_SUMMARY.txt"
        with text_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(self._build_text_summary(summary))

        LOGGER.info("EDA summary saved to %s", summary_path)
        return summary

    def _categorical_summary(self) -> Dict[str, Dict[str, object]]:
        categorical_columns = self.df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        summary: Dict[str, Dict[str, object]] = {}

        for column in categorical_columns[:20]:
            mode_series = self.df[column].mode(dropna=True)
            summary[column] = {
                "unique": int(self.df[column].nunique(dropna=True)),
                "top": None if mode_series.empty else str(mode_series.iloc[0]),
                "missing": int(self.df[column].isna().sum()),
            }

        return summary

    def _save_correlation_heatmap(self, numeric_columns: list[str]) -> Optional[Path]:
        if len(numeric_columns) < 2:
            return None

        plt.figure(figsize=(10, 8))
        corr = self.df[numeric_columns].corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        output_path = self.output_folder / "correlation_heatmap.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    def _save_numeric_distributions(self, numeric_columns: list[str]) -> List[str]:
        saved_paths: List[str] = []
        for column in numeric_columns:
            series = self.df[column].dropna()
            if series.empty:
                continue

            plt.figure(figsize=(8, 4))
            sns.histplot(series, kde=True)
            plt.title(f"Distribution of {column}")
            plt.tight_layout()
            safe_name = self._safe_file_name(column)
            hist_path = self.output_folder / f"hist_{safe_name}.png"
            plt.savefig(hist_path)
            plt.close()
            saved_paths.append(str(hist_path.resolve()))

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=series)
            plt.title(f"Boxplot of {column}")
            plt.tight_layout()
            boxplot_path = self.output_folder / f"boxplot_{safe_name}.png"
            plt.savefig(boxplot_path)
            plt.close()
            saved_paths.append(str(boxplot_path.resolve()))
        return saved_paths

    @staticmethod
    def _safe_file_name(value: str) -> str:
        return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)

    @staticmethod
    def _build_text_summary(summary: Dict[str, object]) -> str:
        lines = [
            "EDA SUMMARY REPORT",
            "=" * 60,
            f"Rows: {summary['shape']['rows']}",
            f"Columns: {summary['shape']['columns']}",
            "",
            "Missing values:",
        ]

        for column, count in summary["missing_values"].items():
            lines.append(f"  - {column}: {count}")

        return "\n".join(lines)
