import logging
import uuid
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from typing import cast
from pandas import DataFrame, Series, concat
from snowflake.snowpark import Session

from src.pipeline.xgboost import run_pipeline, PRED_YHAT_COL, PRED_YHAT_LOWER_COL, PRED_YHAT_UPPER_COL
from src.utils.model import evaluate_model

from projects.pltv import (
    config, 
    PartitionItem,
    Level,
    ModelMetadata,
    ModelStatus,
    ModelStep,
    ModelStepMetadata,
    ModelStepResults,
    ModelStepPredictionMetadata,
    ModelStepPredictionResults,
)


logger = logging.getLogger(__name__)

INITIAL_DF_TABLE_NAME = 'SPINE_DATA'
FINAL_DF_TABLE_NAME = 'RAW_RESULTS'
MODEL_RESULT_TABLE_NAME = 'MODEL_METADATA'


class ModelService:
    def __init__(
        self, 
        session: Session,
        level: Level, # level of granularity the dataset is aggregated to
        df: DataFrame, # dataset
        test_train_split: bool = True,
        save_to_db: bool = True # save to database
    ):
        self.session: Session = session
        self.level: Level = level
        self.df: DataFrame = df
        self.test_train_split: bool = test_train_split
        self.save_to_db: bool = save_to_db

        self.train_test_metadata: ModelStepResults = []
        self.prediction_metadata: ModelStepPredictionResults = []

        self.model_id: str = str(uuid.uuid4())
        self.model_created_at: datetime = datetime.now(timezone.utc)


    def run(self):
        try:
            if self.save_to_db:
                self._save_df(INITIAL_DF_TABLE_NAME, self.df)
            final_df = self._run()
            if self.save_to_db:
                self._save_df(FINAL_DF_TABLE_NAME, final_df)
                self._save_model_metadata(ModelStatus.COMPLETED)                
        except Exception as e:
            logger.error(f"Error running model: {e}")
            if self.save_to_db:
                self._save_model_metadata(ModelStatus.FAILED, message=str(e))
            raise e

    def _run(self) -> DataFrame:
        final_df = DataFrame()
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
                        output_df = self._run_step(
                            partition_item=PartitionItem(partition=partition, value=partition_value),
                            step=step,
                            df=cast(DataFrame, partition_df)
                        )
                        partition_df = output_df
                final_df = concat([final_df, partition_df])
        # log results
        self._log_result()

        return cast(DataFrame, final_df)


    def _run_step(
        self,
        partition_item: PartitionItem, 
        step: ModelStep, 
        df: DataFrame
    ) -> DataFrame:
        step_logger = logger.getChild(step.name)
        step_logger.info(f"Running step {step.name}")

        if self.test_train_split:
            train_metadata = self._run_train_test_split(partition_item, step, df)
            self.train_test_metadata.append(train_metadata)

        prediction_metadata = self._run_train_predict(partition_item, step, df)        
        self.prediction_metadata.append(prediction_metadata)

        return prediction_metadata.output_df
    
    def _run_train_test_split(
        self, 
        partition_item: PartitionItem, 
        step: ModelStep, 
        df: DataFrame,
    ) -> ModelStepMetadata:
        df = df.copy()
        step_logger = logger.getChild(step.name + "_TRAIN_TEST_SPLIT")

        # create training mask
        training_mask = Series(True, index=df.index)
        min_cohort_col = step.min_cohort_col
        step_logger.info(f"Adding mask {min_cohort_col} >= {config.min_cohort_size}")
        training_mask &= df[min_cohort_col] >= config.min_cohort_size
        
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
            # pred_col=step.pred_col,
            # pred_lower_col=step.pred_lower_col,
            # pred_upper_col=step.pred_upper_col,
        )

        # evaluate model
        eval_result = evaluate_model(
            cast(Series, result_df[step.target_col]), 
            y_pred=cast(Series, result_df[PRED_YHAT_COL])
        )
        step_logger.info(f"Evaluation result: {eval_result.model_dump()}")

        # get feature importance
        feature_importances = trained_model.get_feature_importance()
        step_logger.info(f"Feature importances: {feature_importances.to_dict()}")

        metadata = ModelStepMetadata(
            model_id=self.model_id,
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

        if self.save_to_db:
            base_table_name = f'TEST_TRAIN_SPLIT'
            metadata.to_db(
                session=self.session,
                level_name=self.level.name,
                metadata_table_name=f'{base_table_name}_METADATA',
                feature_importances_table_name=f'{base_table_name}_FEATURE_IMPORTANCES'
            )

        return metadata

    def _run_train_predict(
        self, 
        partition_item: PartitionItem, 
        step: ModelStep, 
        df: DataFrame,
    ) -> ModelStepPredictionMetadata:
        df = df.copy()
        step_logger = logger.getChild(step.name + "_TRAIN_PREDICT")

        # create training mask
        training_mask = Series(True, index=df.index)
        min_cohort_col = step.min_cohort_col
        step_logger.info(f"Adding mask {min_cohort_col} >= {config.min_cohort_size}")
        training_mask &= df[min_cohort_col] >= config.min_cohort_size
        
        # create prediction mask
        prediction_mask = Series(True, index=df.index)
        pb_col = step.get_prediction_base_col(partition_item.partition, partition_item.value)
        step_logger.info(f"Adding mask {min_cohort_col} / {pb_col} < {config.prediction_base_threshold} or {min_cohort_col} is na or {pb_col} is na")
        prediction_mask &= (
                (
                    (df[min_cohort_col] < config.min_cohort_size) # lower than min cohort size
                    & (df[min_cohort_col] / df[pb_col] < config.prediction_base_threshold) # ratio of min cohort col to base col is less than threshold
                )
                | df[min_cohort_col].isna()
                | df[pb_col].isna()
            )
        
        step_logger.info(f"Predicting on {len(prediction_mask)} rows")

        # run pipeline
        cat_cols = config.get_cat_cols(self.level)
        num_cols = config.get_num_cols(partition_item, step)
        boolean_cols = config.boolean_cols

        train_df = cast(DataFrame, df[training_mask].copy())
        predict_df = cast(DataFrame, df[prediction_mask].copy())
        passthrough_df = cast(DataFrame, df[~prediction_mask].copy())

        result_df, _ = run_pipeline(
            train_df,
            predict_df,
            target_col=step.target_col,
            cat_cols=cat_cols,
            num_cols=num_cols,
            boolean_cols=boolean_cols,
            # pred_col=step.pred_col,
            # pred_lower_col=step.pred_lower_col,
            # pred_upper_col=step.pred_upper_col,
        )

        # clean result df - only include consistent columns across all steps
        # (num_cols vary between steps, so we exclude them from the saved results)
        result_clean_df = cast(DataFrame, result_df[
            config.get_key_names(self.level) + [
                config.timestamp_col,
                partition_item.partition.name,
            ] + self.level.group_bys + [
                PRED_YHAT_COL,
                PRED_YHAT_LOWER_COL,
                PRED_YHAT_UPPER_COL,
            ]
        ].copy())

        # construct output df to feed into next step
        model_status_col = f'{step.name}_MODEL'
        passthrough_df[model_status_col] = training_mask.loc[passthrough_df.index].map({True: 'TRAINED', False: 'BYPASSED'})
        predict_df[model_status_col] = 'PREDICTED'
        predict_df[step.target_col] = result_df[PRED_YHAT_COL].values # use .values to bypass index alignment
        output_df = cast(DataFrame, concat([passthrough_df, predict_df]))

        prediction_metadata = ModelStepPredictionMetadata(
            model_id=self.model_id,
            prediction_rows=len(prediction_mask),
            partition_item=partition_item,
            step=step,
            cat_cols=cat_cols,
            num_cols=num_cols,
            boolean_cols=boolean_cols,
            output_df=output_df,
            result_df=result_clean_df
        )

        if self.save_to_db:
            base_table_name = f'TRAIN_PREDICT'
            prediction_metadata.to_db(
                session=self.session,
                level_name=self.level.name,
                metadata_table_name=f'{base_table_name}_METADATA',
                prediction_results_table_name=f'{base_table_name}_RESULTS'
            )

        return prediction_metadata

    def _log_result(self):
        from pprint import pformat
        
        result_logger = logger.getChild("results")

        for metadata in self.train_test_metadata:
            # Format evaluation metrics
            eval_dict = metadata.eval_result.model_dump()
            pretty_eval = pformat(eval_dict, indent=4, compact=False)
            
            # Get top 10 feature importances as {feature: importance} dict (sorted by importance desc)
            top_10_df = metadata.feature_importances.head(10)
            top_10_features = {
                row['feature']: round(float(row['importance']), 2) 
                for _, row in top_10_df.iterrows()
            }
            pretty_features = pformat(top_10_features, indent=4, compact=False, sort_dicts=False)
            
            result_logger.info(
                f"\nResults for Partition {metadata.partition_item.partition.name} = {metadata.partition_item.value} for Model Step {metadata.step.name}:\n"
                f"  - training rows: {metadata.training_rows}\n"
                f"  - test rows: {metadata.test_rows}\n"
                f"  - model eval: {pretty_eval}\n"
                f"  - feature importances (top 10): {pretty_features}\n"
                f"  - cat cols: {metadata.cat_cols}\n"
                f"  - num cols: {metadata.num_cols}\n"
                f"  - boolean cols: {metadata.boolean_cols}\n"
            )

    def _save_model_metadata(self, status: ModelStatus, message: str | None = None):
        model_metadata = ModelMetadata(
            id=self.model_id,
            created_at=self.model_created_at,
            level_name=self.level.name,
            level_group_bys=self.level.group_bys,
            status=status,
            message=message,
        )
        model_metadata.to_db(
            session=self.session,
            table_name=MODEL_RESULT_TABLE_NAME,
        )

    def _save_df(self, table_name: str, final_df: DataFrame):
        self.session.write_pandas(final_df.reset_index(drop=True), f'{self.level.name.upper()}_{table_name}', auto_create_table=True, overwrite=False)