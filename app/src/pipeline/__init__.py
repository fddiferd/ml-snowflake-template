"""
Machine Learning Pipeline Modules

This package provides reusable, sklearn-compatible ML pipelines for different algorithms.

Available Pipelines:
- XGBoost: Best for cross-sectional/cohort predictions
- Prophet: Best for time series forecasting

Quick Start - XGBoost:
    from src.pipeline.xgboost import get_pipeline, prepare_data
    
    model, preprocessor = get_pipeline(
        cat_cols=['brand', 'channel'],
        num_cols=['price', 'days_active']
    )

Quick Start - Prophet:
    from src.pipeline.prophet import get_pipeline, prepare_data_for_prophet
    
    model, preprocessor = get_pipeline(
        cat_cols=['brand', 'channel'],
        num_cols=['price', 'days_active']
    )
"""

# Expose main components at package level for easier imports
from src.pipeline.xgboost import (
    XGBoostRegressorWrapper,
    get_pipeline as get_xgboost_pipeline,
    prepare_data as prepare_xgboost_data,
)

from src.pipeline.prophet import (
    ProphetRegressor,
    get_pipeline as get_prophet_pipeline,
    prepare_data_for_prophet,
)

__all__ = [
    # XGBoost
    'XGBoostRegressorWrapper',
    'get_xgboost_pipeline',
    'prepare_xgboost_data',
    # Prophet
    'ProphetRegressor',
    'get_prophet_pipeline',
    'prepare_data_for_prophet',
]

