import os
from abc import ABC, abstractmethod

from pandas import DataFrame


class LocalWriter(ABC):
    """Abstract base class for file-based writers."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True)

    @property
    def is_local(self) -> bool:
        return True

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this writer (e.g., '.parquet', '.csv')."""
        ...

    def _get_file_path(self, name: str) -> str:
        return os.path.join(self.folder_path, f"{name}{self.extension}")

    @abstractmethod
    def write(self, name: str, data: DataFrame, overwrite: bool = True) -> None:
        ...


class ParquetWriter(LocalWriter):
    """Writer for Parquet files."""

    @property
    def extension(self) -> str:
        return ".parquet"

    def write(self, name: str, data: DataFrame, overwrite: bool = True) -> None:
        file_path = self._get_file_path(name)
        if overwrite or not os.path.exists(file_path):
            data.to_parquet(file_path, index=False)
        else:
            # Append by reading existing, concatenating, and writing
            import pandas as pd

            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, data], ignore_index=True)
            combined.to_parquet(file_path, index=False)


class CSVWriter(LocalWriter):
    """Writer for CSV files."""

    @property
    def extension(self) -> str:
        return ".csv"

    def write(self, name: str, data: DataFrame, overwrite: bool = True) -> None:
        file_path = self._get_file_path(name)
        if overwrite or not os.path.exists(file_path):
            data.to_csv(file_path, index=False)
        else:
            # Append without header
            data.to_csv(file_path, mode="a", header=False, index=False)
