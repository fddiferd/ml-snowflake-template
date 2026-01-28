"""
Data Module
===========

Dataset creation and feature engineering for PLTV.

Modules:
    dataset:             get_dataset - fetch training data as Snowpark DataFrame
    loader:              DatasetLoader - load data with caching support
    feature_engineering: clean_df - timestamp conversion, avg net billings calc
    queries/:            SQL templates for spine and feature views

Note: Import data functions from package root to avoid circular imports:
    from projects.pltv import DatasetLoader, clean_df

ResultCollector is now in shared src/data:
    from src.data import ResultCollector
"""

from projects.pltv.data.queries.feature_views import (
    RETENTION_METRICS_QUERY,
    BILLING_METRICS_QUERY,
)
from projects.pltv.data.queries.spine import QUERY as SPINE_QUERY

__all__ = [
    "RETENTION_METRICS_QUERY",
    "BILLING_METRICS_QUERY",
    "SPINE_QUERY",
]
