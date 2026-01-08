"""
Models Module
=============

Pydantic models and ML service for PLTV.

Modules:
    types:   Runtime data structures (PartitionItem, ModelMetadata, etc.)
    service: ModelService for running ML pipeline

Usage:
    from projects.pltv.models import ModelService, PartitionItem, ModelMetadata
"""

from projects.pltv.models.types import (
    FeatureViewConfigModel,
    FeatureViewConfigModels,
    PartitionItem,
    ModelStatus,
    ModelMetadata,
    ModelStepBase,
    ModelStepMetadata,
    ModelStepResults,
    ModelStepPredictionMetadata,
    ModelStepPredictionResults,
)
from projects.pltv.models.service import ModelService

__all__ = [
    # Types
    "FeatureViewConfigModel",
    "FeatureViewConfigModels",
    "PartitionItem",
    "ModelStatus",
    "ModelMetadata",
    "ModelStepBase",
    "ModelStepMetadata",
    "ModelStepResults",
    "ModelStepPredictionMetadata",
    "ModelStepPredictionResults",
    # Service
    "ModelService",
]
