"""
VBB Dataset Loader
==================

Loads VBB datasets with environment-aware caching.
Extends BaseDatasetLoader for shared caching logic.
"""

import logging

import pandas as pd
from snowflake.snowpark import Session

from src.data import BaseDatasetLoader
from projects.vbb.config import CACHE_PATH


logger = logging.getLogger(__name__)


class DatasetLoader(BaseDatasetLoader):
    """Loads VBB datasets with caching support."""

    def __init__(self, session: Session, cache_path: str = CACHE_PATH):
        super().__init__(session, cache_path)

    def _get_cache_key(self) -> str:
        """Cache key for VBB data."""
        return "vbb_data"

    def _load_from_snowflake(self) -> pd.DataFrame:
        """Fetch VBB dataset from Snowflake.
        
        TODO: Replace this placeholder with your actual data source query.
        
        Example:
            from projects.vbb.data.queries import MAIN_QUERY
            return self.session.sql(MAIN_QUERY).to_pandas()
        """
        logger.info("Fetching VBB dataset from Snowflake")
        
        # Placeholder - replace with your actual query
        logger.warning("VBB data query not implemented - returning empty DataFrame")
        return pd.DataFrame()
