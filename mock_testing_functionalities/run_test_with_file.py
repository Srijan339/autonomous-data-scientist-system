import sys
from pathlib import Path
import traceback


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_test_with_file.py <path-to-csv>")
        raise SystemExit(2)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        raise SystemExit(3)

    # Ensure src is importable
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    try:
        from data_processing.data_loader import DataLoader
        from data_processing.data_validator import DataValidator
        from data_processing.data_profiler import DataProfiler
    except Exception as e:
        print("Failed to import modules:")
        traceback.print_exc()
        raise

    try:
        loader = DataLoader(file_path, verbose=True)
        df = loader.load_data()
        print('\n-- Preview --')
        print(df.head(3))

        validator = DataValidator(df)
        print('\n-- Validation Report --')
        validator.print_report()

        profiler = DataProfiler(df)
        print('\n-- Profiling Report --')
        profiler.print_profile()

    except Exception as exc:
        print('Error during test run:')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
