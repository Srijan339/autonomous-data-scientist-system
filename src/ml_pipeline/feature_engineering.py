import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class FeatureEngineer:

    def __init__(self, df, target_column):
        self.df = df
        self.target = target_column

        self.scaler = StandardScaler()

        self.target_encoder = None

        self.categorical_cols = []
        self.numeric_cols = []


    def separate_features(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        return X, y


    def split_data(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() < 20 else None
        )


    def fit_transform_train(self, X):
        X = X.copy()

        # Identify columns
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Handle missing values
        for col in self.numeric_cols:
            X[col] = X[col].fillna(X[col].mean())

        for col in self.categorical_cols:
            X[col] = X[col].fillna("Unknown")

        print("Encoding columns:", list(self.categorical_cols))
        # Encode categorical using pd.get_dummies (drop_first=True)
        if len(self.categorical_cols) > 0:
            X = pd.get_dummies(
                X,
                columns=self.categorical_cols,
                drop_first=True
            )

        # Scale numeric columns only (exclude low cardinality)
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

        # REMOVE target-like columns (low unique values)
        numeric_cols = [col for col in numeric_cols if X[col].nunique() > 2]

        self.numeric_cols = numeric_cols

        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        # Final safety for ML readiness
        X = X.fillna(0)
        assert not X.isna().any().any(), "NaNs remain in training features"
        print("X_train NaN-free:", X.shape)

        return X


    def transform_test(self, X):
        X = X.copy()

        # Handle missing values
        for col in self.numeric_cols:
            X[col] = X[col].fillna(X[col].mean())

        for col in self.categorical_cols:
            X[col] = X[col].fillna("Unknown")

        print("Encoding test columns:", list(self.categorical_cols))
        # Encode categorical using pd.get_dummies (drop_first=True)
        if len(self.categorical_cols) > 0:
            X = pd.get_dummies(
                X,
                columns=self.categorical_cols,
                drop_first=True
            )

        # Scale filtered numeric columns
        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        # Final safety for ML readiness
        X = X.fillna(0)
        print("X_test NaN-free:", X.shape)

        return X


    def encode_target(self, y, fit=True):
        if y.dtype == "object":
            if fit:
                self.target_encoder = LabelEncoder()
                return self.target_encoder.fit_transform(y)
            else:
                return self.target_encoder.transform(y)
        return y


    def run_feature_engineering(self):

        print("\n⚙ Running Feature Engineering (Correct Pipeline)...\n")

        X, y = self.separate_features()

        # Encode target if needed
        y = self.encode_target(y, fit=True)

        # Split FIRST (no leakage)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Fit on train
        X_train = self.fit_transform_train(X_train)

        # Transform test
        X_test = self.transform_test(X_test)

        print("✅ Feature Engineering Completed")
        print("Training shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        return X_train, X_test, y_train, y_test