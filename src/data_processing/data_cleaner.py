from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    fully_empty_columns_removed: List[str]
    normalized_missing_placeholders: int


class DataCleaner:
    """Dataset-level cleaning that is safe to run before train/test split."""

    MISSING_PLACEHOLDERS = {"", " ", "na", "n/a", "none", "null", "nan", "unknown", "missing"}

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_dataset(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            raise ValueError("Cannot clean an empty dataset.")

        LOGGER.info("Starting dataset cleaning.")
        self._normalize_column_names()
        normalized_count = self._normalize_missing_placeholders()
        duplicates_removed = self._remove_duplicates()
        removed_columns = self._drop_fully_empty_columns()
        self._strip_text_columns()
        self._convert_datetime_columns()

        self.cleaning_report = CleaningReport(
            rows_before=int(self.original_row_count),
            rows_after=int(len(self.df)),
            duplicates_removed=int(duplicates_removed),
            fully_empty_columns_removed=removed_columns,
            normalized_missing_placeholders=int(normalized_count),
        )

        LOGGER.info("Cleaning finished. Shape after cleaning: %s", self.df.shape)
        return self.df

    def get_cleaning_report(self) -> Dict[str, object]:
        if not hasattr(self, "cleaning_report"):
            raise ValueError("clean_dataset must be called before requesting the report.")
        return self.cleaning_report.__dict__

    def print_cleaning_report(self) -> None:
        report = self.get_cleaning_report()
        print("\nDATA CLEANING REPORT")
        print("-" * 60)
        print(f"Rows before: {report['rows_before']}")
        print(f"Rows after: {report['rows_after']}")
        print(f"Duplicates removed: {report['duplicates_removed']}")
        print(f"Fully empty columns removed: {report['fully_empty_columns_removed']}")
        print(f"Normalized missing placeholders: {report['normalized_missing_placeholders']}")

    def _normalize_column_names(self) -> None:
        self.original_row_count = len(self.df)
        self.df.columns = [str(column).strip() for column in self.df.columns]

    def _normalize_missing_placeholders(self) -> int:
        replacement_count = 0
        object_columns = self.df.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            series = self.df[column].astype("string")
            normalized = series.str.strip()
            mask = normalized.str.lower().isin(self.MISSING_PLACEHOLDERS)
            replacement_count += int(mask.fillna(False).sum())
            self.df.loc[mask.fillna(False), column] = np.nan
        return replacement_count

    def _remove_duplicates(self) -> int:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return before - len(self.df)

    def _drop_fully_empty_columns(self) -> List[str]:
        empty_columns = [column for column in self.df.columns if self.df[column].isna().all()]
        if empty_columns:
            self.df = self.df.drop(columns=empty_columns)
        return empty_columns

    def _strip_text_columns(self) -> None:
        object_columns = self.df.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            normalized = self.df[column].astype("string").str.strip()
            self.df[column] = normalized.where(normalized.notna(), np.nan).astype(object)

    def _convert_datetime_columns(self) -> None:
        for column in self.df.columns:
            if not (
                pd.api.types.is_object_dtype(self.df[column])
                or pd.api.types.is_string_dtype(self.df[column])
            ):
                continue
            sample = self.df[column].dropna().astype(str).head(20)
            if sample.empty:
                continue
            candidate_ratio = sample.str.contains(r"\d", regex=True).mean()
            separator_ratio = sample.str.contains(r"[-/:]", regex=True).mean()
            if candidate_ratio < 0.8 or separator_ratio < 0.5:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            if parsed.notna().mean() >= 0.8:
                self.df[column] = pd.to_datetime(self.df[column], errors="coerce", utc=False)
