import os
from typing import cast
import logging

import pandas as pd
from snowflake.snowpark import Session

from projects.pltv.core.enums import Level
from projects.pltv.data.dataset import get_dataset
from projects.pltv.data.utils import CACHE_PATH
from src.environment import environment as env


logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads datasets with caching based on environment.

    DEV: Loads from local cache if exists, otherwise fetches and caches
    STAGING/PROD: Fetches directly from Snowflake (no caching)
    """

    def __init__(self, session: Session, cache_path: str = CACHE_PATH):
        self.session = session
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)

    def _get_cache_file_path(self, level: Level) -> str:
        return os.path.join(self.cache_path, f"{level.name}.parquet")

    def load(self, level: Level) -> pd.DataFrame:
        """Load dataset - from cache if local writer, otherwise from Snowflake."""
        if env.target.is_dev and env.use_cache:
            return self._load_with_cache(level)
        return self._load_from_snowflake(level)

    def _load_with_cache(self, level: Level) -> pd.DataFrame:
        cache_file = self._get_cache_file_path(level)
        if os.path.exists(cache_file):
            logger.info(f"Loading {level.name} from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"Cache miss for {level.name}, fetching from Snowflake")
        df = self._load_from_snowflake(level)

        logger.info(f"Saving {level.name} to cache: {cache_file}")
        df.to_parquet(cache_file, index=False)

        return df

    def _load_from_snowflake(self, level: Level) -> pd.DataFrame:
        logger.info(f"Fetching {level.name} dataset from Snowflake")
        dataset = get_dataset(self.session, level)
        return cast(pd.DataFrame, dataset.to_pandas())
