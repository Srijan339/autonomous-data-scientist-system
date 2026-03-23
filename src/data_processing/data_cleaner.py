import pandas as pd
import numpy as np
import os


class DataCleaner:
    """
    Handles automated dataset cleaning with professional reporting.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = {
            "duplicates_removed": 0,
            "columns_fixed": [],
            "outliers_detected": 0,
            "rows_before": df.shape[0]
        }

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        duplicates_removed = before - after
        self.cleaning_log["duplicates_removed"] = duplicates_removed
        print(f"✅ Removed {duplicates_removed} duplicate rows")
        return self.df

    def fix_numeric_types(self):
        \"\"\"Convert string numeric columns to float\"\"\"
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                except:
                    pass
        return self.df

    def handle_missing_values(self):
        missing_cols = []
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                missing_cols.append(column)
                if self.df[column].dtype in ["int64", "float64"]:
                    fill_value = self.df[column].median()
                    if pd.isna(fill_value):
                        fill_value = self.df[column].mean()
                    if pd.isna(fill_value):
                        fill_value = 0
                    self.df[column] = self.df[column].fillna(fill_value)
                else:
                    mode_val = self.df[column].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else "Missing"
                    self.df[column] = self.df[column].fillna(fill_value)
        self.cleaning_log["columns_fixed"] = missing_cols
        print(f"✅ Missing values handled in {len(missing_cols)} columns")
        return self.df

    def handle_embarked(self):
        if "Embarked" in self.df.columns:
            mode_emb = self.df["Embarked"].mode()
            if len(mode_emb) > 0:
                self.df["Embarked"] = self.df["Embarked"].fillna(mode_emb[0])
                print(f"✅ Filled Embarked missing with mode: {mode_emb[0]}")

    def drop_problematic_columns(self):
        \"\"\"Drop useless ID cols and high-missing columns\"\"\"
        cols_to_drop = []
        # Useless columns
        useless = ['PassengerId', 'Name', 'Ticket']
        for col in useless:
            if col in self.df.columns:
                cols_to_drop.append(col)
        # High missing (>70%)
        high_missing = self.df.columns[self.df.isnull().sum() / len(self.df) > 0.7].tolist()
        cols_to_drop.extend(high_missing)
        # Protect critical Titanic features
        protected_cols = ["Sex", "Embarked", "Age", "Pclass", "SibSp", "Parch", "Fare"]
        cols_to_drop = [col for col in cols_to_drop if col not in protected_cols]
        self.df = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns], errors='ignore')
        print(f"✅ Dropped {len(cols_to_drop)} problematic columns: {list(set(cols_to_drop))}")
        return self.df

    def detect_outliers(self):
        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns
        outlier_report = {}
        total_outliers = 0
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = self.df[
                (self.df[col] < lower) | (self.df[col] > upper)
            ]
            outlier_report[col] = len(outliers)
            total_outliers += len(outliers)
        self.cleaning_log["outliers_detected"] = total_outliers
        return outlier_report

    def remove_outliers(self):
        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns
        before = self.df.shape[0]
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df = self.df[
                (self.df[col] >= lower) &
                (self.df[col] <= upper)
            ]
        after = self.df.shape[0]
        rows_removed = before - after
        print(f"✅ Removed {rows_removed} outlier rows (IQR method)")
        return self.df

    def save_dataset(self, output_path="reports/clean_dataset.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Clean dataset saved: {output_path}")
        return output_path

    def print_cleaning_report(self):
        print("\n" + "=" * 60)
        print("📋 DATA CLEANING REPORT")
        print("=" * 60)
        print(f"Duplicates removed: {self.cleaning_log['duplicates_removed']}")
        print(f"Columns with missing values fixed: {len(self.cleaning_log['columns_fixed'])}")
        if self.cleaning_log['columns_fixed']:
            print(f"  → {', '.join(self.cleaning_log['columns_fixed'])}")
        print(f"Outliers detected: {self.cleaning_log['outliers_detected']}")
        print(f"Final dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print("=" * 60 + "\n")

    def clean_dataset(self, remove_outliers=False):
        print("\n🧹 Starting data cleaning pipeline...\n")
        self.fix_numeric_types()
        self.handle_missing_values()
        self.handle_embarked()
        self.drop_problematic_columns()
        self.remove_duplicates()
        self.detect_outliers()
        if remove_outliers:
            self.remove_outliers()
        remaining_nans = self.df.isna().sum().sum()
        if remaining_nans > 0:
            self.df = self.df.fillna(0)
            print(f"⚠️ Final fillna(0) applied for {remaining_nans} NaNs")
        print("Final NaN count:", self.df.isna().sum().sum())
        self.print_cleaning_report()
        self.save_dataset()
        return self.df
