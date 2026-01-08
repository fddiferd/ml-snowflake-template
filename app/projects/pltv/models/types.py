"""
PLTV Runtime Models
===================

Pydantic models for runtime data structures: metadata, results, partition items.

These models are used for:
- Serializing/deserializing model results
- Type safety for partition handling
- DataFrame conversion for storage

Usage:
    from projects.pltv.models import (
        PartitionItem, ModelMetadata, ModelStepMetadata,
    )
"""

from datetime import datetime, timezone
import hashlib
from enum import Enum
import logging
from pydantic import BaseModel, Field, ConfigDict, computed_field
from typing import Any, TypeAlias
from pandas import DataFrame

from projects.pltv.config import ModelStep, Partition
from src.base_models.evaluation import EvaluationResult
from src.pipeline.xgboost import XGBoostRegressorWrapper


logger = logging.getLogger(__name__)


# =============================================================================
# MARK: - Feature View Config (Pydantic version)
# =============================================================================

class FeatureViewConfigModel(BaseModel):
    """Configuration for a feature view (Pydantic version)."""
    name: str
    query: str


FeatureViewConfigModels: TypeAlias = list[FeatureViewConfigModel]


# =============================================================================
# MARK: - Partition Item
# =============================================================================

class PartitionItem(BaseModel):
    """A specific partition with its value."""
    partition: Partition
    value: Any


# =============================================================================
# MARK: - Model Status
# =============================================================================

class ModelStatus(Enum):
    """Status of a model run."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# MARK: - Model Metadata
# =============================================================================

class ModelMetadata(BaseModel):
    """Metadata for a complete model run."""
    id: str
    created_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level_name: str
    level_group_bys: list[str]
    status: ModelStatus
    message: str | None = None

    @computed_field
    @property
    def run_time(self) -> float | None:
        return (self.completed_at - self.created_at).total_seconds() if self.completed_at else None

    def to_dataframe(self) -> DataFrame:
        """Convert model metadata to a DataFrame for writing."""
        data = self.model_dump(mode='json')
        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        return df


# =============================================================================
# MARK: - Model Step Base
# =============================================================================

class ModelStepBase(BaseModel):
    """Base model for model step metadata and prediction metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def id(self) -> str:
        """sha256 hash of model_id, step.name, and partition_item."""
        return hashlib.sha256(
            f"{self.model_id}|{self.step.name}|{self.partition_item.model_dump()}".encode()
        ).hexdigest()

    model_id: str
    partition_item: PartitionItem
    step: ModelStep
    cat_cols: list[str]
    num_cols: list[str]
    boolean_cols: list[str]


# =============================================================================
# MARK: - Model Step Metadata
# =============================================================================

class ModelStepMetadata(ModelStepBase):
    """Metadata for a single model training step (train/test split)."""
    training_rows: int
    test_rows: int
    eval_result: EvaluationResult
    feature_importances: DataFrame
    model: XGBoostRegressorWrapper

    def to_metadata_dataframe(self) -> DataFrame:
        """Convert step metadata to a DataFrame for writing."""
        data = self.model_dump(mode='json', exclude={'model', 'feature_importances'})

        # Extract step and partition info
        data['step_number'] = self.step.value
        data['step_name'] = self.step.name
        data['partition_item'] = self.partition_item.model_dump(mode='json')
        data['eval_result'] = self.eval_result.model_dump(mode='json')

        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        return df

    def to_feature_importances_dataframe(self) -> DataFrame:
        """Convert feature importances to a DataFrame for writing."""
        feature_importances_df = self.feature_importances.copy().reset_index(drop=True)
        feature_importances_df['id'] = self.id
        feature_importances_df.columns = feature_importances_df.columns.str.upper()
        return feature_importances_df


ModelStepResults: TypeAlias = list[ModelStepMetadata]


# =============================================================================
# MARK: - Model Step Prediction Metadata
# =============================================================================

class ModelStepPredictionMetadata(ModelStepBase):
    """Metadata for a single model prediction step."""
    prediction_rows: int
    output_df: DataFrame
    result_df: DataFrame

    def to_metadata_dataframe(self) -> DataFrame:
        """Convert prediction metadata to a DataFrame for writing."""
        data = self.model_dump(mode='json', exclude={'output_df', 'result_df'})

        # Extract step and partition info
        data['step_number'] = self.step.value
        data['step_name'] = self.step.name
        data['partition_item'] = self.partition_item.model_dump(mode='json')

        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        return df

    def to_prediction_results_dataframe(self) -> DataFrame:
        """Convert prediction results to a DataFrame for writing."""
        prediction_results_df = self.result_df.copy().reset_index(drop=True)
        prediction_results_df['id'] = self.id
        prediction_results_df.columns = prediction_results_df.columns.str.upper()
        return prediction_results_df


ModelStepPredictionResults: TypeAlias = list[ModelStepPredictionMetadata]
