import logging
from sklearn.model_selection import train_test_split
from typing import cast
from pandas import DataFrame, Series, concat

from src.pipeline.xgboost import run_pipeline
from src.utils.model import evaluate_model

from projects.pltv import (
    config, 
    Level,
    ModelStep,
    ModelStepResult,
    ModelStepResults
)


logger = logging.getLogger(__name__)


class ModelService:
    def __init__(
        self, 
        level: Level, # level of granularity the dataset is aggregated to
        df: DataFrame # dataset
    ):
        self.level: Level = level
        self.df: DataFrame = df
        self.results: ModelStepResults = []


    def run(self):
        for partition_item in config.partition.items:
            # partition the data
            logger.info(f"Partitioning data where {config.partition.name} = {partition_item.value}")
            partition_df = self.df[self.df[config.partition.name] == partition_item.value]
            # run steps
            for step in config.model_steps:
                self._run_step(cast(DataFrame, partition_df), step)


    def _run_step(self, df: DataFrame, step: ModelStep):
        logger.name = f"run_step_{step.name}"
        logger.info(f"Running step {step.name}")

        # run train test split
        result = self._run_train_test_split(df, step)
        self.results.append(result)
    
    
    def _run_train_test_split(self, df: DataFrame, step: ModelStep) -> ModelStepResult:
        df = df.copy()

        # create training mask
        training_mask = Series(True, index=df.index)
        for col in step.previous_step_min_cohort_cols:
            logger.info(f"Adding mask {col} >= {config.min_cohort_size}")
            training_mask &= df[col] >= config.min_cohort_size
        
        # split train_df into train and test
        train_df, test_df = train_test_split(df[training_mask], test_size=0.1, random_state=42)
        logger.info(f"Training on {len(train_df)} rows and testing on {len(test_df)} rows")

        # run pipeline
        result_df, trained_model = run_pipeline(
            cast(DataFrame, train_df),
            cast(DataFrame, test_df),
            target_col=step.target_col,
            cat_cols=config.cat_cols,
            num_cols=config.num_cols,
            pred_col=step.pred_col,
            pred_lower_col=step.pred_lower_col,
            pred_upper_col=step.pred_upper_col,
        )

        # evaluate model
        eval_result = evaluate_model(
            cast(Series, result_df[step.target_col]), 
            y_pred=cast(Series, result_df[step.pred_col])
        )
        logger.info(f"Evaluation result: {eval_result.model_dump()}")

        # get feature importance
        feature_importances = trained_model.get_feature_importance()
        logger.info(f"Feature importances: {feature_importances.to_dict()}")

        return ModelStepResult(
            eval_result=eval_result,
            feature_importances=feature_importances,
            model=trained_model
        )