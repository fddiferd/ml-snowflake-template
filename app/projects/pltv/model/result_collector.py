import logging
from collections import defaultdict

import pandas as pd
from pandas import DataFrame

from src.writers import Writer


logger = logging.getLogger(__name__)


class ResultCollector:
    """Collects DataFrames and writes them in batch at the end.

    This is optimized for Snowflake writes where individual writes are slow.
    Instead of writing after each step, we accumulate all DataFrames and
    write them in a single batch at the end.
    """

    def __init__(self):
        self._pending: dict[str, list[DataFrame]] = defaultdict(list)

    def add(self, name: str, data: DataFrame) -> None:
        """Accumulate a DataFrame for later writing.

        Args:
            name: The destination name (table name or file name without extension)
            data: The DataFrame to accumulate
        """
        if len(data) == 0:
            logger.debug(f"Skipping empty DataFrame for {name}")
            return
        self._pending[name].append(data.copy())

    def flush(self, writer: Writer) -> None:
        """Concatenate and write all accumulated DataFrames.

        Args:
            writer: The Writer to use for writing the data
        """
        for name, dfs in self._pending.items():
            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Writing {len(combined)} rows to {name}")
            writer.write(name, combined, overwrite=False)

        self._pending.clear()

    def clear(self) -> None:
        """Clear all pending DataFrames without writing."""
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Return the number of pending destinations."""
        return len(self._pending)

    @property
    def pending_names(self) -> list[str]:
        """Return the names of pending destinations."""
        return list(self._pending.keys())
