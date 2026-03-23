from typing import Dict, Any

import pandas as pd
from pandas.api import types as pdtypes


class DataProfiler:
    """Generate profiling summaries for a pandas DataFrame.

    Provides feature-type detection, summary statistics for numeric and
    categorical columns, missing-value summaries, and simple target balance.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_feature_types(self) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            return {"numerical_features": [], "categorical_features": [], "datetime_features": []}

        numerical = [c for c in self.df.columns if pdtypes.is_numeric_dtype(self.df[c])]
        categorical = [c for c in self.df.columns if pdtypes.is_object_dtype(self.df[c]) or pdtypes.is_categorical_dtype(self.df[c])]
        datetime = [c for c in self.df.columns if pdtypes.is_datetime64_any_dtype(self.df[c])]

        return {
            "numerical_features": numerical,
            "categorical_features": categorical,
            "datetime_features": datetime,
        }

    def summary_statistics(self) -> Dict[str, Any]:
        """Return descriptive statistics for numeric and categorical columns."""
        stats: Dict[str, Any] = {}

        if self.df is None or self.df.empty:
            return stats

        num_cols = self.get_feature_types()["numerical_features"]
        cat_cols = self.get_feature_types()["categorical_features"]

        if num_cols:
            stats["numeric"] = self.df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()

        if cat_cols:
            # top, freq, unique for categorical
            cat_stats = {}
            for c in cat_cols:
                top = self.df[c].mode(dropna=True)
                top_val = top.iloc[0] if not top.empty else None
                cat_stats[c] = {
                    "unique": int(self.df[c].nunique(dropna=True)),
                    "top": top_val,
                    "top_count": int(self.df[c].value_counts(dropna=True).iloc[0]) if not self.df[c].value_counts(dropna=True).empty else 0,
                }
            stats["categorical"] = cat_stats

        return stats

    def missing_values(self) -> Dict[str, Any]:
        if self.df is None:
            return {}

        counts = self.df.isnull().sum()
        pct = (counts / self.df.shape[0] * 100).round(2)

        return {"counts": counts.to_dict(), "percent": pct.to_dict()}

    def class_balance(self, target_column: str) -> Dict[str, Any]:
        if self.df is None or target_column not in self.df.columns:
            return {}

        counts = self.df[target_column].value_counts(dropna=False)
        pct = (counts / counts.sum() * 100).round(2)

        return {"counts": counts.to_dict(), "percent": pct.to_dict()}

    def generate_profile(self) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "feature_types": self.get_feature_types(),
            "summary_statistics": self.summary_statistics(),
            "missing_values": self.missing_values(),
        }

        return profile

    def print_profile(self) -> None:
        profile = self.generate_profile()

        print("\n📊 DATA PROFILING REPORT")
        print("-" * 60)

        print("\nFeature types:")
        for k, v in profile["feature_types"].items():
            print(f"  - {k}: {v}")

        print("\nMissing values (showing columns with > 0):")
        for col, cnt in profile["missing_values"]["counts"].items():
            if cnt > 0:
                pct = profile["missing_values"]["percent"].get(col, 0.0)
                print(f"  - {col}: {cnt} ({pct}%)")

        print("\nNumeric summary (top-level keys):")
        for k in profile["summary_statistics"].keys():
            print(f"  - {k}")
