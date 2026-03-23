print("Starting data pipeline...")

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_validator import DataValidator
from src.data_processing.data_profiler import DataProfiler
from src.data_processing.data_cleaner import DataCleaner
from src.analytics.eda_engine import EDAEngine


# Load dataset
loader = DataLoader("data/2019.csv")
df = loader.load_data()

# Validate dataset
validator = DataValidator(df)
validator.print_report()

# Profile dataset
profiler = DataProfiler(df)
profiler.print_profile()

# Clean dataset (with optional outlier removal)
cleaner = DataCleaner(df)
df_clean = cleaner.clean_dataset(remove_outliers=False)  # Set to True to remove outliers

# Print outlier detection report  
outliers = cleaner.detect_outliers()
print("\n🔍 OUTLIER DETECTION REPORT")
print("-" * 40)
for col, count in outliers.items():
    if count > 0:
        print(f"  {col}: {count} outliers detected")

# Run EDA
eda = EDAEngine(df_clean)
eda.run_full_eda()


from src.ml_pipeline.feature_engineering import FeatureEngineer


target_column = "Score"

feature_engineer = FeatureEngineer(df_clean, target_column)

X_train, X_test, y_train, y_test = feature_engineer.run_feature_engineering()


from src.ml_pipeline.model_trainer import ModelTrainer


trainer = ModelTrainer(
    X_train,
    X_test,
    y_train,
    y_test
)

results = trainer.train_models()

print("\nMODEL LEADERBOARD")
print(results)

best_model = trainer.save_best_model(results)