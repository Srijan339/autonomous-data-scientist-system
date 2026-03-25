from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import pandas as pd


LOGGER = logging.getLogger(__name__)


class DataLoader:
    """Load tabular datasets from disk paths or uploaded file-like objects."""

    SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".xls", ".xlsx", ".parquet", ".feather"}

    def __init__(self, source: Optional[Union[str, Path, BinaryIO]] = None):
        self.source = source

    def load_data(
        self,
        source: Optional[Union[str, Path, BinaryIO]] = None,
        read_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        dataset_source = source if source is not None else self.source
        if dataset_source is None:
            raise ValueError("A dataset source must be provided.")

        read_kwargs = read_kwargs or {}

        if hasattr(dataset_source, "read"):
            return self._load_from_buffer(dataset_source, read_kwargs=read_kwargs)

        path = Path(dataset_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file format: {suffix}")

        LOGGER.info("Loading dataset from %s", path)
        return self._read_by_suffix(path, suffix, read_kwargs)

    def preview(
        self,
        n: int = 5,
        source: Optional[Union[str, Path, BinaryIO]] = None,
    ) -> pd.DataFrame:
        return self.load_data(source=source).head(n)

    def _load_from_buffer(
        self,
        buffer: BinaryIO,
        read_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        read_kwargs = read_kwargs or {}
        file_name = getattr(buffer, "name", "uploaded.csv")
        suffix = Path(file_name).suffix.lower()

        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported uploaded file format: {suffix}")

        raw_bytes = buffer.read()
        if hasattr(buffer, "seek"):
            buffer.seek(0)

        LOGGER.info("Loading uploaded dataset: %s", file_name)
        byte_stream = io.BytesIO(raw_bytes)
        return self._read_by_suffix(byte_stream, suffix, read_kwargs)

    @staticmethod
    def _read_by_suffix(source: Union[Path, io.BytesIO], suffix: str, read_kwargs: Dict[str, Any]) -> pd.DataFrame:
        try:
            if suffix == ".csv":
                df = pd.read_csv(source, **read_kwargs)
            elif suffix in {".json", ".jsonl"}:
                df = pd.read_json(source, **read_kwargs)
            elif suffix in {".xls", ".xlsx"}:
                df = pd.read_excel(source, **read_kwargs)
            elif suffix == ".parquet":
                df = pd.read_parquet(source, **read_kwargs)
            elif suffix == ".feather":
                df = pd.read_feather(source, **read_kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        except Exception as exc:
            raise ValueError(f"Failed to load dataset: {exc}") from exc

        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        return df
