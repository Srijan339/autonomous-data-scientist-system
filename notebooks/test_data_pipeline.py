import sys
sys.path.insert(0, '..')

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_validator import DataValidator
from src.data_processing.data_profiler import DataProfiler
from src.data_processing.data_cleaner import DataCleaner


# Load dataset
loader = DataLoader("data/2019.csv")
df = loader.load_data()

# Validate dataset
validator = DataValidator(df)
validator.print_report()

# Profile dataset
profiler = DataProfiler(df)
profiler.print_profile()

# Clean dataset
cleaner = DataCleaner(df)

outliers = cleaner.detect_outliers()
print("\nOutlier Report:")
print(outliers)

df_clean = cleaner.clean_dataset()

print("\nClean dataset shape:", df_clean.shape)