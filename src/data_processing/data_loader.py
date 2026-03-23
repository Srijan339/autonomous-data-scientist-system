from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd


class DataLoader:
    """Utility for loading tabular datasets from common formats.

    Supported formats: CSV, JSON, JSONL, Excel, Parquet, Feather.

    Example:
        loader = DataLoader("data/mydata.csv")
        df = loader.load_data()
    """

    def __init__(self, file_path: Union[str, Path], verbose: bool = True):
        self.path = Path(file_path)
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def load_data(self, read_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load the dataset and return a pandas DataFrame.

        Args:
            read_kwargs: optional keyword args forwarded to the pandas reader.

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the file format is unsupported or reading fails.
        """

        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        read_kwargs = read_kwargs or {}
        suffix = self.path.suffix.lower()

        try:
            if suffix == ".csv":
                df = pd.read_csv(self.path, **read_kwargs)
            elif suffix in {".json", ".jsonl"}:
                df = pd.read_json(self.path, **read_kwargs)
            elif suffix in {".xls", ".xlsx"}:
                df = pd.read_excel(self.path, **read_kwargs)
            elif suffix == ".parquet":
                df = pd.read_parquet(self.path, **read_kwargs)
            elif suffix == ".feather":
                df = pd.read_feather(self.path, **read_kwargs)
            else:
                raise ValueError("Unsupported file format: %s" % suffix)

            self._log("✅ Dataset loaded successfully", f"shape={df.shape}")
            return df

        except Exception as exc:  # re-raise as ValueError for consistent handling
            raise ValueError(f"Failed to read {self.path}: {exc}") from exc

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first `n` rows without re-loading if possible.

        This convenience method attempts to load the data and return `head(n)`.
        """

        df = self.load_data()
        return df.head(n)