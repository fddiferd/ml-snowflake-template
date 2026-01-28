"""
Base Dataset Loader
===================

Abstract base class for dataset loaders with environment-aware caching.

Features:
- DEV mode: Loads from local parquet cache if exists, otherwise fetches and caches
- STAGING/PROD: Always fetches fresh from Snowflake (no local caching)
- Handles read-only filesystem in Snowflake stored procedures

Usage:
    from src.data import BaseDatasetLoader
    from projects.myproject.config import CACHE_PATH
    
    class DatasetLoader(BaseDatasetLoader):
        def __init__(self, session, cache_path: str = CACHE_PATH):
            super().__init__(session, cache_path)
        
        def _get_cache_key(self, level: str) -> str:
            return f"data_{level}"
        
        def _load_from_snowflake(self, level: str) -> pd.DataFrame:
            return self.session.sql(f"SELECT * FROM table WHERE level = '{level}'").to_pandas()
"""

from abc import ABC, abstractmethod
import os
import logging
from typing import cast

import pandas as pd
from snowflake.snowpark import Session

from src.environment import environment as env


logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders with environment-aware caching.
    
    Subclasses must implement:
        - _get_cache_key(*args, **kwargs) -> str
        - _load_from_snowflake(*args, **kwargs) -> pd.DataFrame
    
    The load() method handles caching logic automatically based on environment.
    """

    def __init__(self, session: Session, cache_path: str):
        """Initialize the loader.
        
        Args:
            session: Snowflake Session for querying data
            cache_path: Local directory path for caching parquet files
        """
        self.session = session
        self.cache_path = cache_path
        
        # Only create cache directory in DEV mode
        # Snowflake's stored procedure filesystem is read-only
        if env.target.is_dev:
            try:
                os.makedirs(self.cache_path, exist_ok=True)
            except OSError:
                logger.warning(f"Could not create cache directory: {self.cache_path}")

    def load(self, *args, **kwargs) -> pd.DataFrame:
        """Load dataset - from cache if DEV mode with caching, otherwise from Snowflake.
        
        Args:
            *args, **kwargs: Passed to _get_cache_key and _load_from_snowflake
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        if env.target.is_dev and env.use_cache:
            return self._load_with_cache(*args, **kwargs)
        return self._load_from_snowflake(*args, **kwargs)

    def _load_with_cache(self, *args, **kwargs) -> pd.DataFrame:
        """Load from cache if exists, otherwise fetch and cache.
        
        Args:
            *args, **kwargs: Passed to _get_cache_key and _load_from_snowflake
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        cache_key = self._get_cache_key(*args, **kwargs)
        cache_file = os.path.join(self.cache_path, f"{cache_key}.parquet")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading {cache_key} from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"Cache miss for {cache_key}, fetching from Snowflake")
        df = self._load_from_snowflake(*args, **kwargs)

        logger.info(f"Saving {cache_key} to cache: {cache_file}")
        df.to_parquet(cache_file, index=False)

        return df

    @abstractmethod
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Return a unique cache key for the given parameters.
        
        This key is used as the filename (without extension) for the cached parquet file.
        
        Args:
            *args, **kwargs: Same arguments passed to load()
        
        Returns:
            str: Unique cache key (e.g., "level_1", "dataset_main")
        """
        ...

    @abstractmethod
    def _load_from_snowflake(self, *args, **kwargs) -> pd.DataFrame:
        """Fetch data from Snowflake (project-specific implementation).
        
        Args:
            *args, **kwargs: Same arguments passed to load()
        
        Returns:
            pd.DataFrame: The fetched dataset
        """
        ...
