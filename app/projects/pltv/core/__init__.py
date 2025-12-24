"""
Core Module
===========

Configuration, types, and enums for the PLTV model.

Modules:
    enums:       TimeHorizon (30-730 days), ModelStep (prediction targets)
    base_models: Level, Config, Partition, FeatureViewConfig, ModelStepResult
    config:      Project config instance and feature view configurations
"""

from projects.pltv.core.enums import TimeHorizon, ModelStep, TimeHorizons, ModelSteps
from projects.pltv.core.base_models import (
    Level,
    Config,
    Partition,
    PartitionItem,
    FeatureViewConfig,
    FeatureViewConfigs,
    ModelStepResult,
    ModelStepResults,
)
from projects.pltv.core.config import config, fv_configs

__all__ = [
    "TimeHorizon",
    "ModelStep",
    "TimeHorizons",
    "ModelSteps",
    "Level",
    "Config",
    "Partition",
    "PartitionItem",
    "FeatureViewConfig",
    "FeatureViewConfigs",
    "ModelStepResult",
    "ModelStepResults",
    "config",
    "fv_configs",
]

