"""
PLTV (Predicted Lifetime Value) Package
========================================

Predicts average net billings across time horizons (30-730 days) using XGBoost.
Data is partitioned by promo/non-promo plans and grouped by brand/sku/channel.

Usage:
    from projects.pltv import Level, config, get_session, get_df, clean_df, ModelService
    
    session = get_session()
    df = get_df(session, Level(group_bys=["brand", "sku_type", "channel"]))
    clean_df(df)
    ModelService(level, df).run()

Exports:
    Session:    get_session
    Config:     config, fv_configs
    Types:      Level, Config, Partition, PartitionItem, FeatureViewConfig
    Enums:      TimeHorizon, ModelStep
    Data:       get_df, get_df_from_cache, clean_df
    Model:      ModelService
"""

from typing import TYPE_CHECKING
from snowflake.snowpark import Session

# Type hints for lazy imports (enables IDE autocomplete & silences linter)
if TYPE_CHECKING:
    from projects.pltv.data.dataset import get_df as get_df
    from projects.pltv.data.dataset import get_df_from_cache as get_df_from_cache
    from projects.pltv.data.feature_engineering import clean_df as clean_df
    from projects.pltv.model.model_service import ModelService as ModelService

from projects import Project
from src.connection.session import get_session as get_snowflake_session

# Core components (eager imports - no circular dependency issues)
from projects.pltv.core.config import config, fv_configs
from projects.pltv.core.enums import (
    TimeHorizon, 
    ModelStep, 
    ModelSteps, 
    TimeHorizons
)
from projects.pltv.core.base_models import (
    Config, 
    Level, 
    Partition, 
    PartitionItem, 
    FeatureViewConfig, 
    FeatureViewConfigs,
    ModelStepResult,
    ModelStepResults
)


def get_session() -> Session:
    """Get a Snowflake session configured for the PLTV project."""
    return get_snowflake_session(Project.PLTV)


# ============================================================================
# Lazy imports for data functions
# ============================================================================
# These are loaded on first access to avoid circular imports.
# The pattern: core/config imports queries, but data modules import config.
# Lazy loading breaks this cycle.

_lazy_imports = {
    # Data functions
    "get_df": ("projects.pltv.data.dataset", "get_df"),
    "get_df_from_cache": ("projects.pltv.data.dataset", "get_df_from_cache"),
    "clean_df": ("projects.pltv.data.feature_engineering", "clean_df"),
    # Model service
    "ModelService": ("projects.pltv.model.model_service", "ModelService"),
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
    # Config
    "config", 
    "fv_configs",
    # Enums
    "TimeHorizon", 
    "ModelStep", 
    "ModelSteps", 
    "TimeHorizons", 
    # Types
    "Config", 
    "Level", 
    "Partition", 
    "PartitionItem", 
    "FeatureViewConfig", 
    "FeatureViewConfigs",
    "ModelStepResult",
    "ModelStepResults",
    # Data functions (lazy loaded)
    "get_df",
    "get_df_from_cache", 
    "clean_df",
    # Model (lazy loaded)
    "ModelService",
]