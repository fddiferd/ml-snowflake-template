from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Callable, Any, TypeAlias
from pandas import DataFrame

from projects.pltv.core.enums import ModelStep, ModelSteps, TimeHorizons, Partitions, Partition
from src.base_models.evaluation import EvaluationResult
from src.pipeline.xgboost import XGBoostRegressorWrapper


# MARK: - FV Config
class FeatureViewConfig(BaseModel):
    name: str
    query: str

FeatureViewConfigs: TypeAlias = list[FeatureViewConfig]


# MARK: - Levels
class Level(BaseModel):
    """A level is a group of columns that are used to group the data."""
    group_bys: list[str]

    @model_validator(mode='before')
    def capitalize_join_keys(cls, values):
        if 'group_bys' in values:
            values['group_bys'] = [key.upper() for key in values['group_bys']]
        return values

    @property
    def name(self) -> str:
        """return the last item in the group_bys list, capitalized"""
        return f'{self.group_bys[-1].upper()}_LEVEL'

    @property
    def sql_fields(self) -> str:
        # Ensure there is a comma after every item, including the last one
        return ", ".join(self.group_bys) + ("," if self.group_bys else "")

Levels: TypeAlias = list[Level]


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
        if 'cat_cols' in values:
            values['cat_cols'] = [col.upper() for col in values['cat_cols']]
        if 'num_cols' in values:
            values['num_cols'] = [col.upper() for col in values['num_cols']]
        if 'boolean_cols' in values:
            values['boolean_cols'] = [col.upper() for col in values['boolean_cols']]
        return values

    def get_cat_cols(self, level: Level) -> list[str]:
        """Return cat cols for the level as well as the global group bys"""
        return self.cat_cols + [col.upper() for col in level.group_bys]

    def get_num_cols(self, partition_item: PartitionItem, step: ModelStep) -> list[str]:
        """Return additional regressors based on the step and partition item as well as the global num cols"""
        return self.num_cols + step.get_additional_regressor_cols(partition_item.partition, partition_item.value)

    def get_join_keys(self, level: Level) -> list[str]:
        return [self.timestamp_col] + [partition.name.upper() for partition in self.partitions] + [col.upper() for col in level.group_bys]


# MARK: - ModelStepResults
class ModelStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    training_rows: int
    test_rows: int
    partition_item: PartitionItem
    step: ModelStep
    eval_result: EvaluationResult
    feature_importances: DataFrame
    cat_cols: list[str]
    num_cols: list[str]
    boolean_cols: list[str]
    model: XGBoostRegressorWrapper

ModelStepResults: TypeAlias = list[ModelStepResult]


class ModelStepPredictionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prediction_rows: int
    partition_item: PartitionItem
    step: ModelStep
    cat_cols: list[str]
    num_cols: list[str]
    boolean_cols: list[str]
    output_df: DataFrame
    result_df: DataFrame

ModelStepPredictionResults: TypeAlias = list[ModelStepPredictionResult]