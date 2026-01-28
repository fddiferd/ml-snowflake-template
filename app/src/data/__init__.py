"""
Data Utilities
==============

Shared data loading, caching, and result collection utilities for all ML projects.

Usage:
    from src.data import BaseDatasetLoader, ResultCollector
    
    # Dataset loader with caching
    class MyLoader(BaseDatasetLoader):
        def _get_cache_key(self, *args) -> str:
            return "my_data"
        
        def _load_from_snowflake(self, *args) -> pd.DataFrame:
            return self.session.sql("SELECT ...").to_pandas()
    
    # Batch result collection
    collector = ResultCollector()
    collector.add("predictions", df)
    collector.flush(writer)
"""

from src.data.base_loader import BaseDatasetLoader
from src.data.result_collector import ResultCollector

__all__ = ["BaseDatasetLoader", "ResultCollector"]
