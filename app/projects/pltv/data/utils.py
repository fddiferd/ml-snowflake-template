import os

from projects.pltv.config import CACHE_PATH


def get_file_path(file_name: str, csv: bool = False) -> str:
    """Get the file path for a cache file.
    
    Note: This is kept for backward compatibility with get_df/get_df_from_cache.
    New code should use DatasetLoader instead.
    """
    os.makedirs(CACHE_PATH, exist_ok=True)
    if csv:
        return os.path.join(CACHE_PATH, file_name + '.csv')
    else:
        return os.path.join(CACHE_PATH, file_name + '.parquet')
