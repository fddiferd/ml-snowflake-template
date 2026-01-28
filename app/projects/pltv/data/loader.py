"""
PLTV Dataset Loader
===================

Loads PLTV datasets with environment-aware caching.
Extends BaseDatasetLoader for shared caching logic.
"""

from typing import cast

import pandas as pd
from snowflake.snowpark import Session

from src.data import BaseDatasetLoader
from projects.pltv.config import Level, CACHE_PATH
from projects.pltv.data.dataset import get_dataset


class DatasetLoader(BaseDatasetLoader):
    """Loads PLTV datasets by level with caching support."""

    def __init__(self, session: Session, cache_path: str = CACHE_PATH):
        super().__init__(session, cache_path)

    def _get_cache_key(self, level: Level) -> str:
        """Cache key based on the aggregation level."""
        return level.name

    def _load_from_snowflake(self, level: Level) -> pd.DataFrame:
        """Fetch PLTV dataset from Snowflake for the given level."""
        dataset = get_dataset(self.session, level)
        return cast(pd.DataFrame, dataset.to_pandas())
