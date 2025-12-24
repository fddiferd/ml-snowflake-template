from pydantic import BaseModel, model_validator, ConfigDict
from typing import Callable, Any, TypeAlias
from pandas import DataFrame

from projects.pltv.core.enums import ModelStep, ModelSteps, TimeHorizons
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

    def get_join_keys(self, timestamp_col: str | None = None, partition_col: str | None = None) -> list[str]:
        keys = []
        if timestamp_col:
            keys.append(timestamp_col)
        if partition_col:
            keys.append(partition_col)
        keys.extend(self.group_bys)
        return keys

    def get_all_cat_cols(self, cat_cols: list[str]) -> list[str]:
        return [col.upper() for col in cat_cols] + [col.upper() for col in self.group_bys]

Levels: TypeAlias = list[Level]


# MARK: - Partitions
class PartitionItem(BaseModel):
    value: Any
    additional_regressor_cols: list[str]

    @model_validator(mode='before')
    def capitalize_fields(cls, values):
        if 'additional_regressor_cols' in values:
            values['additional_regressor_cols'] = [col.upper() for col in values['additional_regressor_cols']]
        return values
        
PartitionItems: TypeAlias = list[PartitionItem]

class Partition(BaseModel):
    name: str
    items: PartitionItems

    @model_validator(mode='before')
    def capitalize_fields(cls, values):
        if 'name' in values and values['name']:
            values['name'] = values['name'].upper()
        return values


# MARK: - Config
class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    version_number: int
    min_cohort_size: int
    timestamp_col: str
    partition: Partition
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
        return self.cat_cols + [col.upper() for col in level.group_bys]

    def get_num_cols(self, step: ModelStep) -> list[str]:
        return self.num_cols + step.additional_regressor_cols


# MARK: - ModelStepResults
class ModelStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    eval_result: EvaluationResult
    feature_importances: DataFrame
    model: XGBoostRegressorWrapper

ModelStepResults: TypeAlias = list[ModelStepResult]