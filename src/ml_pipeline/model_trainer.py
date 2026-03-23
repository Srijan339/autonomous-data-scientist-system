import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import joblib


class ModelTrainer:

    def __init__(self, X_train, X_test, y_train, y_test, task_type="regression"):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type

        if task_type == "classification":
            self.models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier()
            }
        else:
            self.models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

    def train_models(self):

        results = []

        print("\n🚀 Training Models...\n")

        for name, model in self.models.items():

            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)

            if self.task_type == "classification":
                # Classification metrics
                accuracy = accuracy_score(self.y_test, predictions)
                precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
                
                results.append({
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                })
                
                print(f"{name} trained - Accuracy: {accuracy:.4f}")
            else:
                # Regression metrics
                r2 = r2_score(self.y_test, predictions)
                rmse = mean_squared_error(self.y_test, predictions, squared=False)

                results.append({
                    "Model": name,
                    "R2 Score": r2,
                    "RMSE": rmse
                })
                
                print(f"{name} trained - R2: {r2:.4f}")

        results_df = pd.DataFrame(results)

        return results_df

    def save_best_model(self, results_df):

        if self.task_type == "classification":
            best_model_name = results_df.sort_values(
                "Accuracy",
                ascending=False
            ).iloc[0]["Model"]
            metric_name = "Accuracy"
        else:
            best_model_name = results_df.sort_values(
                "R2 Score",
                ascending=False
            ).iloc[0]["Model"]
            metric_name = "R2 Score"

        best_model = self.models[best_model_name]

        joblib.dump(best_model, "reports/best_model.pkl")

        print(f"\nBest model saved: {best_model_name} ({metric_name})")

        return best_model_name

