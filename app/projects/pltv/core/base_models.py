from datetime import datetime, timezone
import hashlib
from enum import Enum
import logging
import json
from pydantic import BaseModel, Field, model_validator, ConfigDict, computed_field
from typing import Callable, Any, TypeAlias
from pandas import DataFrame

from snowflake.snowpark import Session

from projects.pltv.core.enums import ModelStep, ModelSteps, TimeHorizons, Partitions, Partition, Levels, Level
from src.base_models.evaluation import EvaluationResult
from src.pipeline.xgboost import XGBoostRegressorWrapper


logger = logging.getLogger(__name__)


# MARK: - FV Config
class FeatureViewConfig(BaseModel):
    name: str
    query: str

FeatureViewConfigs: TypeAlias = list[FeatureViewConfig]

# Mark: - Partition Item
class PartitionItem(BaseModel):
    partition: Partition
    value: Any


# MARK: - Config
class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    version_number: int
    min_cohort_size: int
    prediction_base_threshold: float = Field(description="The threshold for the prediction base column to be considered baked", ge=0.0, le=1.0)
    timestamp_col: str
    partitions: Partitions
    levels: Levels
    time_horizons: TimeHorizons
    model_steps: ModelSteps

    cat_cols: list[str]
    num_cols: list[str]
    boolean_cols: list[str]
    
    # Column naming callables (using Any to allow project-specific TimeHorizon enums)
    get_gross_adds_created_over_days_ago_column: Callable[[Any], str]
    get_net_billings_days_column: Callable[[Any], str]
    get_avg_net_billings_column: Callable[[Any], str]

    @model_validator(mode='before')
    def capitalize_fields(cls, values):
        if 'timestamp_col' in values and values['timestamp_col']:
            values['timestamp_col'] = values['timestamp_col'].upper()
        if 'primary_key_col' in values and values['primary_key_col']:
            values['primary_key_col'] = values['primary_key_col'].upper()
        if 'cat_cols' in values:
            values['cat_cols'] = [col.upper() for col in values['cat_cols']]
        if 'num_cols' in values:
            values['num_cols'] = [col.upper() for col in values['num_cols']]
        if 'boolean_cols' in values:
            values['boolean_cols'] = [col.upper() for col in values['boolean_cols']]
        return values

    @property
    def partition_fields(self) -> list[str]:
        return [p.name.upper() for p in self.partitions]

    def get_key_names(self, level: Level) -> list[str]:
        return [f"{level_name}_KEY" for level_name, _ in level.get_key_fields()
        ]

    def get_keys_sql_fields(self, level: Level):
        """Create a select statement for keys based on the current level"""

        def list_to_key(level_name: str, fields: list[str]) -> str:
            return f"sha2(concat_ws('|', {', '.join(fields)}), 256) as {level_name}_KEY"

        joined_key_fields = [
            list_to_key(
                level_name,
                [self.timestamp_col] + self.partition_fields + key_field
            ) for level_name, key_field in level.get_key_fields()
        ]
         
        return ", ".join(joined_key_fields) + ","

    def get_cat_cols(self, level: Level) -> list[str]:
        """Return cat cols for the level as well as the global group bys"""
        return self.cat_cols + [col.upper() for col in level.group_bys]

    def get_num_cols(self, partition_item: PartitionItem, step: ModelStep) -> list[str]:
        """Return additional regressors based on the step and partition item as well as the global num cols"""
        return self.num_cols + step.get_additional_regressor_cols(partition_item.partition, partition_item.value)

    def get_join_keys(self, level: Level) -> list[str]:
        return [self.timestamp_col] + [partition.name.upper() for partition in self.partitions] + [col.upper() for col in level.group_bys]

# MARK: - Model Objects
class ModelStatus(Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ModelMetadata(BaseModel):
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

    def to_db(
        self, 
        session: Session, 
        table_name: str, 
    ) -> None:
        logger.info(f"Saving model result for level {self.level_name}")
        data = self.model_dump(mode='json')
        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        session.write_pandas(df, table_name, auto_create_table=True, overwrite=False)

class ModelStepBase(BaseModel):
    """Base model for model step metadata and prediction metadata"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def id(self) -> str:
        """sha256 hash of model_id, step.name, and partition_item"""
        return hashlib.sha256(f"{self.model_id}|{self.step.name}|{self.partition_item.model_dump()}".encode()).hexdigest()

    model_id: str
    partition_item: PartitionItem
    step: ModelStep
    cat_cols: list[str]
    num_cols: list[str]
    boolean_cols: list[str]

class ModelStepMetadata(ModelStepBase):
    training_rows: int
    test_rows: int
    eval_result: EvaluationResult
    feature_importances: DataFrame
    model: XGBoostRegressorWrapper

    def to_db(
        self, 
        session: Session,
        level_name: str,
        metadata_table_name: str, 
        feature_importances_table_name: str
    ) -> None:
        logger.info(f"Saving model step result {self.step.name} for level {level_name}")
        data = self.model_dump(mode='json', exclude={'model', 'feature_importances'})

        # extract step and partition
        data['step_number'] = self.step.value
        data['step_name'] = self.step.name
        data['partition_item'] = self.partition_item.model_dump(mode='json')
        data['eval_result'] = self.eval_result.model_dump(mode='json')

        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        session.write_pandas(df, f'{level_name}_{metadata_table_name}', auto_create_table=True, overwrite=False)

        feature_importances_df = self.feature_importances.copy().reset_index(drop=True)
        feature_importances_df['id'] = self.id
        feature_importances_df.columns = feature_importances_df.columns.str.upper()

        session.write_pandas(feature_importances_df, f'{level_name}_{feature_importances_table_name}', auto_create_table=True, overwrite=False)

ModelStepResults: TypeAlias = list[ModelStepMetadata]


class ModelStepPredictionMetadata(ModelStepBase):
    prediction_rows: int
    output_df: DataFrame
    result_df: DataFrame

    def to_db(
        self, 
        session: Session, 
        level_name: str,
        metadata_table_name: str, 
        prediction_results_table_name: str
    ) -> None:
        logger.info(f"Saving model step prediction result {self.step.name}  for level {level_name}")
        data = self.model_dump(mode='json', exclude={'output_df', 'result_df'})

        # extract step and partition
        data['step_number'] = self.step.value
        data['step_name'] = self.step.name
        data['partition_item'] = self.partition_item.model_dump(mode='json')

        df = DataFrame([data]).reset_index(drop=True)
        df.columns = df.columns.str.upper()
        session.write_pandas(df, f'{level_name}_{metadata_table_name}', auto_create_table=True, overwrite=False)

        prediction_results_df = self.result_df.copy().reset_index(drop=True)
        prediction_results_df['id'] = self.id
        prediction_results_df.columns = prediction_results_df.columns.str.upper()

        session.write_pandas(prediction_results_df, f'{level_name}_{prediction_results_table_name}', auto_create_table=True, overwrite=False)

ModelStepPredictionResults: TypeAlias = list[ModelStepPredictionMetadata]