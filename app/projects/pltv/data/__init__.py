"""
Data Module
===========

Dataset creation and feature engineering for PLTV.

Modules:
    dataset:             get_df, get_df_from_cache - fetch/cache training data
    feature_engineering: clean_df - timestamp conversion, avg net billings calc
    queries/:            SQL templates for spine and feature views

Note: Import data functions from package root to avoid circular imports:
    from projects.pltv import get_df, clean_df
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

