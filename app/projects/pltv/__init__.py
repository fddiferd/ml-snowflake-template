"""
PLTV (Predicted Lifetime Value) Package
========================================

Predicts average net billings across time horizons (30-1460 days) using XGBoost.
Data is partitioned by promo/non-promo plans and grouped by brand/sku/channel.

Usage:
    from projects.pltv import Level, get_session, clean_df, DatasetLoader, ModelService
    from src.writers import create_writer
    
    session = get_session()
    writer = create_writer("parquet", folder_path="app/projects/pltv/data/cache")
    loader = DatasetLoader(session, cache_path="app/projects/pltv/data/cache")
    df = loader.load(Level.CHANNEL)
    clean_df(df)
    ModelService(Level.CHANNEL, df, writer).run()

Exports:
    Session:    get_session
    Config:     fv_configs, constants, and helper functions
    Types:      Partition, PartitionItem, FeatureViewConfig
    Enums:      Level, TimeHorizon, ModelStep
    Data:       DatasetLoader, clean_df
    Model:      ModelService
"""

if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)


from typing import TYPE_CHECKING
from snowflake.snowpark import Session

# Type hints for lazy imports (enables IDE autocomplete & silences linter)
if TYPE_CHECKING:
    from projects.pltv.data.feature_engineering import clean_df as clean_df
    from projects.pltv.data.loader import DatasetLoader as DatasetLoader
    from projects.pltv.models import ModelService as ModelService

from projects import Project

# Config exports (enums, constants, helpers)
from projects.pltv.config import (
    # Enums
    Level,
    Levels,
    levels,
    Partition,
    Partitions,
    partitions,
    partition_fields,
    partition_sql_fields,
    TimeHorizon,
    TimeHorizons,
    time_horizons,
    ModelStep,
    ModelSteps,
    model_steps,
    # Column constants
    GROSS_ADDS_COL,
    TIMESTAMP_COL,
    IS_PREDICTED_COL,
    # Table constants
    TABLE_SPINE_DATA,
    TABLE_RAW_RESULTS,
    TABLE_MODEL_METADATA,
    # TABLE_DATASET,
    # Status constants
    STATUS_TRAINED,
    STATUS_PREDICTED,
    STATUS_BYPASSED,
    # Configuration values
    VERSION_NUMBER,
    MIN_COHORT_SIZE,
    PREDICTION_BASE_THRESHOLD,
    CAT_COLS,
    NUM_COLS,
    BOOLEAN_COLS,
    # Column helpers
    get_gross_adds_created_over_days_ago_col,
    get_net_billings_col,
    get_avg_net_billings_col,
    get_total_net_billings_col,
    get_model_status_col,
    # Key helpers
    get_cat_cols,
    get_num_cols,
    get_join_keys,
    # Feature view configs
    FeatureViewConfig,
    FeatureViewConfigs,
    fv_configs,
)

# Runtime models
from projects.pltv.models import (
    PartitionItem,
    ModelStatus,
    ModelMetadata,
    ModelStepMetadata,
    ModelStepResults,
    ModelStepPredictionMetadata,
    ModelStepPredictionResults,
)


def get_session() -> Session:
    """Get a Snowflake session configured for the PLTV project."""
    # Lazy import to avoid loading src.environment at module level
    # (environment.py requires TARGET env var which isn't set during Snowflake procedure validation)
    from src.connection.session import get_session as get_snowflake_session
    return get_snowflake_session(Project.PLTV)


# ============================================================================
# Lazy imports for data functions
# ============================================================================
# These are loaded on first access to avoid circular imports.
# The pattern: config imports queries, but data modules import config.
# Lazy loading breaks this cycle.

_lazy_imports = {
    # Data functions
    "clean_df": ("projects.pltv.data.feature_engineering", "clean_df"),
    "DatasetLoader": ("projects.pltv.data.loader", "DatasetLoader"),
    # Model service
    "ModelService": ("projects.pltv.models", "ModelService"),
}


def __getattr__(name: str):
    """Lazy import handler for data and model components."""
    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Session
    "get_session",
    # Enums
    "Level",
    "Levels",
    "levels",
    "Partition",
    "Partitions",
    "partitions",
    "partition_fields",
    "partition_sql_fields",
    "TimeHorizon",
    "TimeHorizons",
    "time_horizons",
    "ModelStep",
    "ModelSteps",
    "model_steps",
    # Column constants
    "GROSS_ADDS_COL",
    "TIMESTAMP_COL",
    "IS_PREDICTED_COL",
    # Table constants
    "TABLE_SPINE_DATA",
    "TABLE_RAW_RESULTS",
    "TABLE_MODEL_METADATA",
    # "TABLE_DATASET",
    # Status constants
    "STATUS_TRAINED",
    "STATUS_PREDICTED",
    "STATUS_BYPASSED",
    # Configuration values
    "VERSION_NUMBER",
    "MIN_COHORT_SIZE",
    "PREDICTION_BASE_THRESHOLD",
    "CAT_COLS",
    "NUM_COLS",
    "BOOLEAN_COLS",
    # Column helpers
    "get_gross_adds_created_over_days_ago_col",
    "get_net_billings_col",
    "get_avg_net_billings_col",
    "get_total_net_billings_col",
    "get_model_status_col",
    # Key helpers
    "get_cat_cols",
    "get_num_cols",
    "get_join_keys",
    # Feature view configs
    "FeatureViewConfig",
    "FeatureViewConfigs",
    "fv_configs",
    # Runtime models
    "PartitionItem",
    "ModelStatus",
    "ModelMetadata",
    "ModelStepMetadata",
    "ModelStepResults",
    "ModelStepPredictionMetadata",
    "ModelStepPredictionResults",
    # Data functions (lazy loaded)
    "DatasetLoader",
    "clean_df",
    # Model (lazy loaded)
    "ModelService",
]

if __name__ == "__main__":
    get_session()
