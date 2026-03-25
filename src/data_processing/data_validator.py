from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    report: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class DataValidator:
    """Validate datasets before AutoML execution."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate(self, target_column: Optional[str] = None) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if self.df is None:
            errors.append("Dataset is None.")
            return ValidationResult(False, {}, errors, warnings)

        if self.df.empty:
            errors.append("Dataset is empty.")

        report = self._build_report()

        if target_column is not None:
            if target_column not in self.df.columns:
                errors.append(f"Target column '{target_column}' is missing from the dataset.")
            else:
                target_series = self.df[target_column]
                target_assessment = self.assess_target_candidates()
                blocked_targets = {item["column"]: item["reason"] for item in target_assessment["blocked_targets"]}
                if target_column in blocked_targets:
                    errors.append(f"Target column '{target_column}' is not suitable: {blocked_targets[target_column]}")
                if target_series.dropna().empty:
                    errors.append(f"Target column '{target_column}' contains only missing values.")
                if target_series.nunique(dropna=True) <= 1:
                    errors.append(f"Target column '{target_column}' must contain at least two unique values.")

        if report["duplicate_rows"] > 0:
            warnings.append(f"Dataset contains {report['duplicate_rows']} duplicate rows.")

        if report["num_rows"] < 30:
            warnings.append("Dataset has fewer than 30 rows; model quality may be unstable.")

        return ValidationResult(not errors, report, errors, warnings)

    def assess_target_candidates(self) -> Dict[str, List[Dict[str, str]]]:
        recommended_targets: List[Dict[str, str]] = []
        blocked_targets: List[Dict[str, str]] = []

        for column in self.df.columns:
            reason = self._target_block_reason(column)
            payload = {"column": column, "reason": reason or "Suitable target candidate."}
            if reason:
                blocked_targets.append(payload)
            else:
                recommended_targets.append(payload)

        return {
            "recommended_targets": recommended_targets,
            "blocked_targets": blocked_targets,
        }

    def _build_report(self) -> Dict[str, Any]:
        n_rows, n_cols = self.df.shape
        missing_counts = self.df.isna().sum()
        missing_percent = (missing_counts / max(n_rows, 1) * 100).round(2)
        unique_counts = self.df.nunique(dropna=False)

        return {
            "num_rows": int(n_rows),
            "num_columns": int(n_cols),
            "column_names": list(self.df.columns),
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_values": missing_counts.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "unique_counts": unique_counts.to_dict(),
            "constant_columns": [column for column, value in unique_counts.items() if value <= 1],
        }

    def print_report(self, target_column: Optional[str] = None) -> ValidationResult:
        result = self.validate(target_column=target_column)
        report = result.report

        print("\nDATA VALIDATION REPORT")
        print("-" * 60)

        if report:
            print(f"Rows: {report['num_rows']} | Columns: {report['num_columns']}")
            print(f"Duplicate rows: {report['duplicate_rows']}")

            if report["constant_columns"]:
                print(f"Constant columns: {report['constant_columns']}")

            print("\nMissing values:")
            for column in report["column_names"]:
                missing = report["missing_values"].get(column, 0)
                if missing:
                    pct = report["missing_percent"].get(column, 0.0)
                    print(f"  - {column}: {missing} ({pct}%)")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        return result

    def _target_block_reason(self, column: str) -> Optional[str]:
        series = self.df[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = unique_count / max(len(non_null), 1)
        name = str(column).strip().lower()

        if non_null.empty:
            return "All values are missing."
        if unique_count <= 1:
            return "Target needs at least two unique values."
        if name.startswith("unnamed"):
            return "Looks like an exported index column."
        if any(token in name for token in ["id", "uuid", "guid", "index", "serial", "identifier"]):
            return "Looks like an identifier column, not a supervised target."
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Datetime targets are not supported in this app yet."
        if unique_ratio >= 0.98 and len(non_null) > 50:
            return "Almost every row is unique, so this behaves like an identifier."
        if pd.api.types.is_numeric_dtype(series) and non_null.is_monotonic_increasing and unique_ratio > 0.8:
            return "Column appears monotonic, which usually indicates an index or ordered key."
        return None
