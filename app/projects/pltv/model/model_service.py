import logging
from sklearn.model_selection import train_test_split
from typing import cast
from pandas import DataFrame, Series

from src.pipeline.xgboost import run_pipeline
from src.utils.model import evaluate_model

from projects.pltv import (
    config, 
    PartitionItem,
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
        # loop through partitions
        for partition in config.partitions:
            # loop through partition values
            for partition_value in partition.values:
                # partition the data
                logger.info(f"Partitioning data where {partition.name} = {partition_value}")
                partition_df = self.df[self.df[partition.name] == partition_value]
                # run steps
                for step in config.model_steps:
                    if step.is_step_in_partition(partition, partition_value):
                        self._run_step(
                            PartitionItem(partition=partition, value=partition_value),
                            step,
                            cast(DataFrame, partition_df)
                        )
        # log results
        self._log_result()


    def _run_step(self, partition_item: PartitionItem, step: ModelStep, df: DataFrame):
        step_logger = logger.getChild(step.name)
        step_logger.info(f"Running step {step.name}")

        # run train test split
        result = self._run_train_test_split(partition_item, step, df, step_logger)
        self.results.append(result)
    
    
    def _run_train_test_split(
        self, 
        partition_item: PartitionItem, 
        step: ModelStep, 
        df: DataFrame,
        step_logger: logging.Logger
    ) -> ModelStepResult:
        df = df.copy()

        # create training mask
        training_mask = Series(True, index=df.index)
        for col in [step.previous_step_min_cohort_col]: # currently a string - keep loop incase we add more in the future
            step_logger.info(f"Adding mask {col} >= {config.min_cohort_size}")
            training_mask &= df[col] >= config.min_cohort_size
        
        # split train_df into train and test
        train_df, test_df = train_test_split(df[training_mask], test_size=0.1, random_state=42)
        step_logger.info(f"Training on {len(train_df)} rows and testing on {len(test_df)} rows")

        # run pipeline
        cat_cols = config.get_cat_cols(self.level)
        num_cols = config.get_num_cols(partition_item, step)
        boolean_cols = config.boolean_cols
        result_df, trained_model = run_pipeline(
            cast(DataFrame, train_df),
            cast(DataFrame, test_df),
            target_col=step.target_col,
            cat_cols=cat_cols,
            num_cols=num_cols,
            boolean_cols=boolean_cols,
            pred_col=step.pred_col,
            pred_lower_col=step.pred_lower_col,
            pred_upper_col=step.pred_upper_col,
        )

        # evaluate model
        eval_result = evaluate_model(
            cast(Series, result_df[step.target_col]), 
            y_pred=cast(Series, result_df[step.pred_col])
        )
        step_logger.info(f"Evaluation result: {eval_result.model_dump()}")

        # get feature importance
        feature_importances = trained_model.get_feature_importance()
        step_logger.info(f"Feature importances: {feature_importances.to_dict()}")

        return ModelStepResult(
            training_rows=len(train_df),
            test_rows=len(test_df),
            partition_item=partition_item,
            step=step,
            eval_result=eval_result,
            feature_importances=feature_importances,
            cat_cols=cat_cols,
            num_cols=num_cols,
            boolean_cols=boolean_cols,
            model=trained_model
        )

    def _log_result(self):
        from pprint import pformat
        
        result_logger = logger.getChild("results")

        for result in self.results:
            # Format evaluation metrics
            eval_dict = result.eval_result.model_dump()
            pretty_eval = pformat(eval_dict, indent=4, compact=False)
            
            # Get top 10 feature importances as {feature: importance} dict (sorted by importance desc)
            top_10_df = result.feature_importances.head(10)
            top_10_features = {
                row['feature']: round(float(row['importance']), 2) 
                for _, row in top_10_df.iterrows()
            }
            pretty_features = pformat(top_10_features, indent=4, compact=False, sort_dicts=False)
            
            result_logger.info(
                f"\nResults for Partition {result.partition_item.partition.name} = {result.partition_item.value} for Model Step {result.step.name}:\n"
                f"  - training rows: {result.training_rows}\n"
                f"  - test rows: {result.test_rows}\n"
                f"  - model eval: {pretty_eval}\n"
                f"  - feature importances (top 10): {pretty_features}\n"
                f"  - cat cols: {result.cat_cols}\n"
                f"  - num cols: {result.num_cols}\n"
                f"  - boolean cols: {result.boolean_cols}\n"
            )
