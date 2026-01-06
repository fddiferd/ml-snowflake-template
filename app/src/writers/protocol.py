from typing import Protocol

from pandas import DataFrame


class Writer(Protocol):
    """Protocol for writing DataFrames to various destinations."""

    @property
    def is_local(self) -> bool:
        """Returns True if this writer writes to local filesystem."""
        ...

    @property
    def folder_path(self) -> str | None:
        """Returns the folder path for local writers, None for remote writers."""
        ...

    def write(self, name: str, data: DataFrame, overwrite: bool = True) -> None:
        """Write a DataFrame to the destination.

        Args:
            name: The name/identifier for the data (table name or file name without extension)
            data: The DataFrame to write
            overwrite: If True, overwrite existing data. If False, append.
        """
        ...
