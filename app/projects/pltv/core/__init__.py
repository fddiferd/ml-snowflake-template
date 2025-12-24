"""
Core Module
===========

Configuration, types, and enums for the PLTV model.

Modules:
    enums:       TimeHorizon (30-730 days), ModelStep (prediction targets)
    base_models: Level, Config, PartitionItem, FeatureViewConfig, ModelStepResult
    config:      Project config instance and feature view configurations
"""

from projects.pltv.core.enums import TimeHorizon, ModelStep, TimeHorizons, ModelSteps, Partition
from projects.pltv.core.base_models import (
    Level,
    Config,
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
    "Partition",
    "Level",
    "Config",
    "PartitionItem",
    "FeatureViewConfig",
    "FeatureViewConfigs",
    "ModelStepResult",
    "ModelStepResults",
    "config",
    "fv_configs",
]

