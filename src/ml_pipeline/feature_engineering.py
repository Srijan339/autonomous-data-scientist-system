from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ProblemDetectionResult:
    problem_type: str
    unique_values: int
    dtype: str


class ProblemTypeDetector:
    """Infer whether the target describes a classification or regression task."""

    @staticmethod
    def detect(target: pd.Series) -> ProblemDetectionResult:
        non_null = target.dropna()
        unique_values = int(non_null.nunique())
        dtype = str(target.dtype)

        if unique_values <= 1:
            raise ValueError("Target column must contain at least two unique non-null values.")

        if pd.api.types.is_bool_dtype(target) or pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
            return ProblemDetectionResult("classification", unique_values, dtype)

        unique_ratio = unique_values / max(len(non_null), 1)
        if pd.api.types.is_integer_dtype(target):
            if unique_values <= 20 or unique_ratio <= 0.05:
                return ProblemDetectionResult("classification", unique_values, dtype)

        if pd.api.types.is_numeric_dtype(target) and unique_values > 20:
            return ProblemDetectionResult("regression", unique_values, dtype)

        if unique_ratio <= 0.1:
            return ProblemDetectionResult("classification", unique_values, dtype)

        return ProblemDetectionResult("regression", unique_values, dtype)


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Reusable feature creation layer for domain and interaction features."""

    def __init__(self, max_interaction_pairs: int = 6):
        self.max_interaction_pairs = max_interaction_pairs

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureGenerator":
        X = self._ensure_dataframe(X)
        self.high_cardinality_text_columns_ = self._identify_high_cardinality_text_columns(X)
        self.numeric_columns_ = [
            column
            for column in X.select_dtypes(include="number").columns
            if X[column].nunique(dropna=True) > 2
        ]
        ranked_columns = sorted(
            self.numeric_columns_,
            key=lambda column: float(X[column].fillna(0).var()),
            reverse=True,
        )
        selected = ranked_columns[:4]
        self.interaction_pairs_ = list(itertools.islice(itertools.combinations(selected, 2), self.max_interaction_pairs))
        self.datetime_columns_ = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_dataframe(X).copy()

        for column in getattr(self, "datetime_columns_", []):
            if column not in X.columns:
                continue
            X[f"{column}_year"] = X[column].dt.year
            X[f"{column}_month"] = X[column].dt.month
            X[f"{column}_day"] = X[column].dt.day
            X = X.drop(columns=[column])

        X = self._handle_text_columns(X)
        self._add_domain_features(X)
        self._add_interaction_features(X)
        return self._sanitize_output(X)

    def _add_domain_features(self, X: pd.DataFrame) -> None:
        if {"SibSp", "Parch"}.issubset(X.columns):
            X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
            X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        if {"Fare", "SibSp", "Parch"}.issubset(X.columns):
            family_size = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
            X["FarePerPerson"] = X["Fare"] / family_size.replace(0, 1)

        if "Cabin" in X.columns:
            X["CabinKnown"] = X["Cabin"].notna().astype(int)

        if "Name" in X.columns:
            extracted_title = X["Name"].astype("string").str.extract(r",\s*([^\.]+)\.", expand=False)
            X["NameTitle"] = extracted_title.fillna("Unknown")

    def _add_interaction_features(self, X: pd.DataFrame) -> None:
        for left, right in getattr(self, "interaction_pairs_", []):
            if left in X.columns and right in X.columns:
                X[f"{left}_x_{right}"] = X[left].fillna(0) * X[right].fillna(0)

    def _handle_text_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in getattr(self, "high_cardinality_text_columns_", []):
            if column not in X.columns:
                continue
            series = X[column].astype("string")
            X[f"{column}_char_length"] = series.str.len().fillna(0)
            X[f"{column}_word_count"] = series.str.split().str.len().fillna(0)
            X[f"{column}_prefix"] = series.str.extract(r"([A-Za-z]+)", expand=False).fillna("unknown")
            X = X.drop(columns=[column])
        return X

    @staticmethod
    def _identify_high_cardinality_text_columns(X: pd.DataFrame) -> List[str]:
        columns: List[str] = []
        text_columns = X.select_dtypes(include=["object", "string", "category"]).columns
        for column in text_columns:
            series = X[column].dropna().astype(str)
            if series.empty:
                continue
            unique_ratio = series.nunique() / max(len(series), 1)
            average_length = series.str.len().mean()
            if unique_ratio > 0.3 or (average_length > 18 and series.nunique() > 20):
                columns.append(column)
        return columns

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureGenerator expects a pandas DataFrame as input.")
        return X

    @staticmethod
    def _sanitize_output(X: pd.DataFrame) -> pd.DataFrame:
        sanitized = X.copy()
        for column in sanitized.columns:
            if pd.api.types.is_string_dtype(sanitized[column]):
                sanitized[column] = sanitized[column].astype(object)
        return sanitized.replace({pd.NA: np.nan})


def build_preprocessing_pipeline() -> ColumnTransformer:
    continuous_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    discrete_numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _create_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous_numeric", continuous_numeric_pipeline, _select_continuous_numeric_columns),
            ("discrete_numeric", discrete_numeric_pipeline, _select_discrete_numeric_columns),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer, input_frame: pd.DataFrame) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        numeric_columns = input_frame.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = input_frame.select_dtypes(exclude=np.number).columns.tolist()
        return numeric_columns + categorical_columns


def _create_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _select_continuous_numeric_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for column in df.select_dtypes(include=np.number).columns:
        series = df[column].dropna()
        if series.empty:
            columns.append(column)
            continue
        if pd.api.types.is_float_dtype(df[column]) or series.nunique() > 10:
            columns.append(column)
    return columns


def _select_discrete_numeric_columns(df: pd.DataFrame) -> List[str]:
    continuous = set(_select_continuous_numeric_columns(df))
    return [column for column in df.select_dtypes(include=np.number).columns if column not in continuous]
