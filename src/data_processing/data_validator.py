from typing import Dict, Any

import pandas as pd


class DataValidator:
    """Validate a pandas DataFrame and produce a compact report.

    Example:
        validator = DataValidator(df)
        report = validator.validate()
        validator.print_report()
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("DataFrame is None")

        n_rows, n_cols = self.df.shape

        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / max(1, n_rows) * 100).round(2)

        data_types = self.df.dtypes.astype(str).to_dict()

        unique_counts = self.df.nunique(dropna=False).to_dict()

        constant_columns = [c for c, u in unique_counts.items() if u <= 1]

        report: Dict[str, Any] = {
            "num_rows": int(n_rows),
            "num_columns": int(n_cols),
            "column_names": list(self.df.columns),
            "data_types": data_types,
            "missing_values": missing_counts.to_dict(),
            "missing_percent": missing_pct.to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "unique_counts": unique_counts,
            "constant_columns": constant_columns,
        }

        return report

    def print_report(self) -> None:
        report = self.validate()

        print("\n📊 DATA VALIDATION REPORT")
        print("-" * 60)

        print(f"Rows: {report['num_rows']}  Columns: {report['num_columns']}")
        print(f"Duplicate rows: {report['duplicate_rows']}")

        if report["constant_columns"]:
            print(f"Constant columns: {report['constant_columns']}")

        print("\nColumn Types:")
        for col, dtype in report["data_types"].items():
            print(f"  - {col}: {dtype}")

        print("\nMissing Values (count / %):")
        for col in report["column_names"]:
            cnt = report["missing_values"].get(col, 0)
            pct = report["missing_percent"].get(col, 0.0)
            if cnt > 0:
                print(f"  - {col}: {cnt} / {pct}%")

        print("\nSample unique counts (first 10):")
        for col, u in list(report["unique_counts"].items())[:10]:
            print(f"  - {col}: {u}")