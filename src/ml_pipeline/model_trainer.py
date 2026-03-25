from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.analytics.eda_engine import EDAEngine
from src.analytics.insight_generator import InsightGenerator
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.data_profiler import DataProfiler
from src.data_processing.data_validator import DataValidator
from src.ml_pipeline.feature_engineering import (
    FeatureGenerator,
    ProblemTypeDetector,
    build_preprocessing_pipeline,
    get_feature_names,
)


LOGGER = logging.getLogger(__name__)


class AutoMLSystem:
    """Production-style AutoML workflow for arbitrary tabular datasets."""

    def __init__(
        self,
        output_dir: str = "reports",
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        random_search_iterations: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.random_search_iterations = random_search_iterations
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        validation = DataValidator(df).print_report(target_column=target_column)
        if not validation.is_valid:
            raise ValueError("Dataset validation failed: " + "; ".join(validation.errors))

        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean_dataset()
        cleaner.print_cleaning_report()

        if target_column not in cleaned_df.columns:
            raise ValueError(f"Target column '{target_column}' was removed during cleaning.")

        before_drop = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[target_column]).reset_index(drop=True)
        dropped_targets = before_drop - len(cleaned_df)
        if dropped_targets:
            self.logger.warning("Dropped %s rows with missing target values.", dropped_targets)
        if cleaned_df.empty:
            raise ValueError("Dataset is empty after removing rows with missing target values.")

        profiler = DataProfiler(cleaned_df)
        profile = profiler.generate_profile()
        eda_summary = EDAEngine(cleaned_df, output_folder=str(self.output_dir), target_column=target_column).run_full_eda()

        detection = ProblemTypeDetector.detect(cleaned_df[target_column])
        problem_type = detection.problem_type

        X = cleaned_df.drop(columns=[target_column])
        y = cleaned_df[target_column]
        prediction_schema = self._build_prediction_schema(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if problem_type == "classification" and y.nunique(dropna=True) <= max(20, int(len(y) * 0.2)) else None,
        )

        leaderboard, trained_models = self._train_candidates(X_train, y_train, problem_type)
        best_model_name = leaderboard.iloc[0]["model"]
        best_pipeline = trained_models[best_model_name]

        best_pipeline.fit(X_train, y_train)
        metrics = self._evaluate(best_pipeline, X_test, y_test, problem_type)
        feature_importance = self._compute_feature_importance(best_pipeline, X_test)
        shap_output = self._generate_shap_outputs(best_pipeline, X_test, problem_type)
        evaluation_artifacts = self._generate_evaluation_artifacts(best_pipeline, X_test, y_test, problem_type)
        insight_payload = InsightGenerator(
            df=cleaned_df,
            target_column=target_column,
            problem_type=problem_type,
            metrics=metrics,
            feature_importance=feature_importance,
            profile=profile,
            eda_summary=eda_summary,
            output_dir=str(self.output_dir),
            best_model=best_model_name,
        ).generate()

        preprocessing_pipeline = Pipeline(best_pipeline.steps[:-1])
        artifact_paths = self._save_artifacts(
            best_pipeline=best_pipeline,
            preprocessing_pipeline=preprocessing_pipeline,
            leaderboard=leaderboard,
            metrics=metrics,
            eda_summary=eda_summary,
            profile=profile,
            feature_importance=feature_importance,
            shap_output=shap_output,
            target_column=target_column,
            prediction_schema=prediction_schema,
            cleaned_df=cleaned_df,
            problem_type=problem_type,
            best_model_name=best_model_name,
            insights_payload=insight_payload,
            evaluation_artifacts=evaluation_artifacts,
        )

        results = {
            "problem_type": problem_type,
            "best_model": best_model_name,
            "leaderboard": leaderboard.to_dict(orient="records"),
            "metrics": metrics,
            "feature_importance": feature_importance[:20],
            "artifacts": artifact_paths,
            "prediction_schema": prediction_schema,
            "eda_charts": eda_summary.get("chart_paths", []),
            "insights": insight_payload["insights"],
        }

        self.print_summary(results)
        return results

    def _train_candidates(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
        candidate_models = self._build_models(problem_type)
        scorer = "f1_weighted" if problem_type == "classification" else "r2"
        cv = self._build_cv(problem_type)

        leaderboard_rows: List[Dict[str, Any]] = []
        trained_models: Dict[str, Pipeline] = {}

        for model_name, estimator in candidate_models.items():
            self.logger.info("Training candidate model: %s", model_name)
            pipeline = self._build_model_pipeline(estimator)

            if "Random Forest" in model_name:
                pipeline, cv_score = self._tune_random_forest(pipeline, X_train, y_train, problem_type, cv, scorer)
            else:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1)
                cv_score = float(np.mean(cv_scores))

            trained_models[model_name] = pipeline
            leaderboard_rows.append({"model": model_name, "mean_cv_score": round(cv_score, 6)})

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_cv_score", ascending=False).reset_index(drop=True)
        return leaderboard, trained_models

    def _build_models(self, problem_type: str) -> Dict[str, Any]:
        if problem_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(random_state=self.random_state),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=self.random_state),
            }

        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=self.random_state),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=self.random_state),
        }

    def _build_model_pipeline(self, estimator: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("feature_generator", FeatureGenerator()),
                ("preprocessor", build_preprocessing_pipeline()),
                ("model", estimator),
            ]
        )

    def _build_cv(self, problem_type: str) -> Any:
        if problem_type == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _tune_random_forest(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        cv: Any,
        scorer: str,
    ) -> Tuple[Pipeline, float]:
        if problem_type == "classification":
            param_distributions = {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }
        else:
            param_distributions = {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=self.random_search_iterations,
            cv=cv,
            random_state=self.random_state,
            scoring=scorer,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        self.logger.info("Best params for random forest: %s", search.best_params_)
        return search.best_estimator_, float(search.best_score_)

    def _evaluate(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> Dict[str, float]:
        predictions = pipeline.predict(X_test)

        if problem_type == "classification":
            return {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
                "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 6),
            }

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        return {
            "r2": round(float(r2_score(y_test, predictions)), 6),
            "rmse": round(rmse, 6),
        }

    def _compute_feature_importance(self, pipeline: Pipeline, X_test: pd.DataFrame) -> List[Dict[str, float]]:
        transformed_input = pipeline.named_steps["feature_generator"].transform(X_test.copy())
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(preprocessor, transformed_input)
        estimator = pipeline.named_steps["model"]

        importances: np.ndarray
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)
            importances = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        else:
            return []

        pairs = [
            {"feature": str(name), "importance": round(float(score), 6)}
            for name, score in zip(feature_names, importances)
        ]
        return sorted(pairs, key=lambda item: item["importance"], reverse=True)

    def _generate_shap_outputs(self, pipeline: Pipeline, X_test: pd.DataFrame, problem_type: str) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt
            import shap
        except ImportError:
            self.logger.warning("SHAP is not installed. Skipping explainability artifacts.")
            return {"enabled": False, "reason": "shap not installed"}

        transformed_input = pipeline.named_steps["feature_generator"].transform(X_test.copy())
        transformed_matrix = pipeline.named_steps["preprocessor"].transform(transformed_input)
        if hasattr(transformed_matrix, "toarray"):
            transformed_matrix = transformed_matrix.toarray()

        sample_size = min(200, len(transformed_input))
        transformed_matrix = transformed_matrix[:sample_size]
        feature_names = get_feature_names(pipeline.named_steps["preprocessor"], transformed_input)
        estimator = pipeline.named_steps["model"]

        try:
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(transformed_matrix, check_additivity=False)
            else:
                explainer = shap.Explainer(estimator.predict, transformed_matrix, feature_names=feature_names)
                shap_values = explainer(transformed_matrix)
            plt.figure()
            shap.summary_plot(shap_values, transformed_matrix, feature_names=feature_names, show=False)
            output_path = self.output_dir / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            return {"enabled": True, "summary_plot": str(output_path)}
        except Exception as exc:
            self.logger.warning("Unable to generate SHAP outputs: %s", exc)
            return {"enabled": False, "reason": str(exc), "problem_type": problem_type}

    def _generate_evaluation_artifacts(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
    ) -> Dict[str, str]:
        import matplotlib.pyplot as plt

        predictions = pipeline.predict(X_test)
        artifacts: Dict[str, str] = {}

        if problem_type == "classification":
            labels = [str(label) for label in sorted(pd.Series(y_test).dropna().unique().tolist())]
            matrix = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(6, 5))
            plt.imshow(matrix, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
            plt.tight_layout()
            path = self.output_dir / "confusion_matrix.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            artifacts["confusion_matrix"] = str(path.resolve())
        else:
            plt.figure(figsize=(6, 5))
            plt.scatter(y_test, predictions, alpha=0.7)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.tight_layout()
            path = self.output_dir / "actual_vs_predicted.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            artifacts["actual_vs_predicted"] = str(path.resolve())

        return artifacts

    def _save_artifacts(
        self,
        best_pipeline: Pipeline,
        preprocessing_pipeline: Pipeline,
        leaderboard: pd.DataFrame,
        metrics: Dict[str, float],
        eda_summary: Dict[str, Any],
        profile: Dict[str, Any],
        feature_importance: List[Dict[str, float]],
        shap_output: Dict[str, Any],
        target_column: str,
        prediction_schema: List[Dict[str, Any]],
        cleaned_df: pd.DataFrame,
        problem_type: str,
        best_model_name: str,
        insights_payload: Dict[str, Any],
        evaluation_artifacts: Dict[str, str],
    ) -> Dict[str, str]:
        model_path = self.output_dir / "best_model.pkl"
        preprocessing_path = self.output_dir / "preprocessing_pipeline.pkl"
        metrics_path = self.output_dir / "metrics.json"
        leaderboard_path = self.output_dir / "leaderboard.json"
        feature_importance_path = self.output_dir / "feature_importance.json"
        profile_path = self.output_dir / "data_profile.json"
        metadata_path = self.output_dir / "model_metadata.json"

        joblib.dump(best_pipeline, model_path)
        joblib.dump(preprocessing_pipeline, preprocessing_path)

        metrics_payload = {"metrics": metrics, "shap": shap_output}
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        leaderboard_path.write_text(leaderboard.to_json(orient="records", indent=2), encoding="utf-8")
        feature_importance_path.write_text(json.dumps(feature_importance[:50], indent=2), encoding="utf-8")
        profile_path.write_text(json.dumps({"profile": profile, "eda_summary": eda_summary}, indent=2, default=str), encoding="utf-8")
        metadata_payload = {
            "target_column": target_column,
            "problem_type": problem_type,
            "best_model": best_model_name,
            "training_columns": list(cleaned_df.columns),
            "prediction_columns": [column for column in cleaned_df.columns if column != target_column],
            "prediction_schema": prediction_schema,
            "row_count": int(cleaned_df.shape[0]),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, default=str), encoding="utf-8")

        return {
            "model": str(model_path.resolve()),
            "preprocessing_pipeline": str(preprocessing_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "feature_importance": str(feature_importance_path.resolve()),
            "data_profile": str(profile_path.resolve()),
            "model_metadata": str(metadata_path.resolve()),
            "insights": insights_payload["insights_path"],
            "knowledge_base": insights_payload["knowledge_path"],
            "eda_summary_text": str((self.output_dir / "EDA_SUMMARY.txt").resolve()),
            "eda_summary_json": str((self.output_dir / "eda_summary.json").resolve()),
            **evaluation_artifacts,
        }

    @staticmethod
    def _build_prediction_schema(X: pd.DataFrame) -> List[Dict[str, Any]]:
        schema: List[Dict[str, Any]] = []
        for column in X.columns:
            non_null = X[column].dropna()
            dtype = str(X[column].dtype)
            sample_values = non_null.head(5).tolist()
            schema.append(
                {
                    "name": column,
                    "dtype": dtype,
                    "required": bool(non_null.shape[0] > 0),
                    "example": None if not sample_values else sample_values[0],
                    "sample_values": sample_values,
                }
            )
        return schema

    @staticmethod
    def print_summary(results: Dict[str, Any]) -> None:
        print("\nAUTOML RESULTS")
        print("-" * 60)
        print(f"Problem type: {results['problem_type']}")
        print(f"Best model: {results['best_model']}")
        print(f"Metrics: {results['metrics']}")
        print("Top feature importance:")
        for item in results["feature_importance"][:10]:
            print(f"  - {item['feature']}: {item['importance']}")


class ModelTrainer(AutoMLSystem):
    """Backward-compatible name for the training engine."""
