import logging
import uuid
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from typing import Any, cast
from pandas import DataFrame, Series, concat

from src.pipeline.xgboost import run_pipeline, run_multi_output_pipeline, PRED_YHAT_COL, PRED_YHAT_LOWER_COL, PRED_YHAT_UPPER_COL
from src.utils.model import evaluate_model
from src.writers import Writer

from projects.pltv.data.result_collector import ResultCollector
from projects.pltv.config import (
    # Enums
    Level,
    ModelStep,
    # Column constants
    GROSS_ADDS_COL,
    TIMESTAMP_COL,
    IS_PREDICTED_COL,
    PREDICTION_BASE_COL,
    # Day-1 baked columns (no model prediction needed)
    GROSS_ADDS_CANCELED_DAY_ONE_COL,
    CROSS_SELL_ADDS_DAY_ONE_COL,
    # Table constants
    TABLE_SPINE_DATA,
    TABLE_RAW_RESULTS,
    TABLE_MODEL_METADATA,
    TABLE_TEST_TRAIN_SPLIT_METADATA,
    TABLE_TEST_TRAIN_SPLIT_FEATURE_IMPORTANCES,
    TABLE_TRAIN_PREDICT_METADATA,
    TABLE_TRAIN_PREDICT_RESULTS,
    TABLE_DATASET,
    # Status constants
    STATUS_TRAINED,
    STATUS_PREDICTED,
    STATUS_BYPASSED,
    # Configuration values
    MIN_COHORT_SIZE,
    PREDICTION_BASE_THRESHOLD,
    BOOLEAN_COLS,
    NUM_COLS,
    # Lists
    partitions,
    model_steps,
    # Helper functions
    get_cat_cols,
    get_num_cols,
    get_model_status_col,
    get_total_net_billings_col,
    get_avg_net_billings_col,
    get_join_keys,
    get_gross_adds_created_over_days_ago_col,
    time_horizons,
)
from projects.pltv.models.types import (
    PartitionItem,
    ModelMetadata,
    ModelStatus,
    ModelStepMetadata,
    ModelStepResults,
    ModelStepPredictionMetadata,
    ModelStepPredictionResults,
)


logger = logging.getLogger(__name__)

    
class ModelService:
    def __init__(
        self, 
        level: Level,
        df: DataFrame,
        writer: Writer | None = None,
        test_train_split: bool = True,
    ):
        """Initialize the ModelService.
        
        Args:
            level: Level of granularity the dataset is aggregated to
            df: The dataset to run the model on
            writer: Writer to use for saving results. If None, no results are saved.
            test_train_split: Whether to run train/test split evaluation
        """
        self.level: Level = level
        self.df: DataFrame = df
        self.writer: Writer | None = writer
        self.test_train_split: bool = test_train_split

        self.train_test_metadata: ModelStepResults = []
        self.prediction_metadata: ModelStepPredictionResults = []

        self.model_id: str = str(uuid.uuid4())
        self.model_created_at: datetime = datetime.now(timezone.utc)
        
        # Result collector for batched writes
        self._collector = ResultCollector()

    def run(self):
        """Run the model pipeline and write results."""
        try:
            self._add_df_to_collector(TABLE_SPINE_DATA, self.df)
            final_df = self._run()
            self._add_df_to_collector(TABLE_RAW_RESULTS, final_df)
            self._add_model_metadata_to_collector(ModelStatus.COMPLETED)
            
            # Flush all collected results at the end
            self._flush_results()
                            
        except Exception as e:
            logger.error(f"Error running model: {e}")
            self._add_model_metadata_to_collector(ModelStatus.FAILED, message=str(e))
            self._flush_results()
            raise e

    def _run(self) -> DataFrame:
        final_df = DataFrame()
        # loop through partitions
        for partition in partitions:
            # loop through partition values
            for partition_value in partition.values:
                # partition the data
                logger.info(f"Partitioning data where {partition.name} = {partition_value}")
                partition_df = self.df[self.df[partition.name] == partition_value]
                # run steps
                for step in model_steps:
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

        # Build and add dashboard dataset
        dataset_df = self._build_dataset_df(cast(DataFrame, final_df))
        self._add_df_to_collector(TABLE_DATASET, dataset_df)

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
            
            # Add train/test split results to collector
            self._collector.add(
                f'{self.level.name}_{TABLE_TEST_TRAIN_SPLIT_METADATA}',
                train_metadata.to_metadata_dataframe()
            )
            self._collector.add(
                f'{self.level.name}_{TABLE_TEST_TRAIN_SPLIT_FEATURE_IMPORTANCES}',
                train_metadata.to_feature_importances_dataframe()
            )

        prediction_metadata = self._run_train_predict(partition_item, step, df)        
        self.prediction_metadata.append(prediction_metadata)
        
        # Add prediction results to collector
        self._collector.add(
            f'{self.level.name}_{TABLE_TRAIN_PREDICT_METADATA}',
            prediction_metadata.to_metadata_dataframe()
        )
        self._collector.add(
            f'{self.level.name}_{TABLE_TRAIN_PREDICT_RESULTS}',
            prediction_metadata.to_prediction_results_dataframe()
        )

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
        training_mask = self._get_training_mask(df, step, step_logger)
        
        # split train_df into train and test
        train_df, test_df = train_test_split(df[training_mask], test_size=0.1, random_state=42)
        step_logger.info(f"Training on {len(train_df)} rows and testing on {len(test_df)} rows")

        # run pipeline
        cat_cols = get_cat_cols(self.level)
        num_cols = get_num_cols(partition_item.partition, partition_item.value, step)
        boolean_cols = BOOLEAN_COLS
        target_cols = step.target_cols
        
        if step.is_multi_target:
            # Multi-output pipeline
            result_df, models_dict = run_multi_output_pipeline(
                cast(DataFrame, train_df),
                cast(DataFrame, test_df),
                target_cols=target_cols,
                cat_cols=cat_cols,
                num_cols=num_cols,
                boolean_cols=boolean_cols,
            )
            
            # Evaluate first target (primary metric for logging)
            primary_target = target_cols[0]
            eval_result = evaluate_model(
                cast(Series, result_df[primary_target]), 
                y_pred=cast(Series, result_df[f"{primary_target}_YHAT"])
            )
            step_logger.info(f"Evaluation result (primary target {primary_target}): {eval_result.model_dump()}")
            
            # Log evaluation for all targets
            for target_col in target_cols:
                target_eval = evaluate_model(
                    cast(Series, result_df[target_col]), 
                    y_pred=cast(Series, result_df[f"{target_col}_YHAT"])
                )
                step_logger.info(f"  {target_col}: {target_eval.model_dump()}")
            
            # Get feature importance from first model
            trained_model = models_dict[primary_target]
            feature_importances = trained_model.get_feature_importance()
        else:
            # Single-output pipeline
            target_col = target_cols[0]
            result_df, trained_model = run_pipeline(
                cast(DataFrame, train_df),
                cast(DataFrame, test_df),
                target_col=target_col,
                cat_cols=cat_cols,
                num_cols=num_cols,
                boolean_cols=boolean_cols,
            )

            # evaluate model
            eval_result = evaluate_model(
                cast(Series, result_df[target_col]), 
                y_pred=cast(Series, result_df[PRED_YHAT_COL])
            )
            step_logger.info(f"Evaluation result: {eval_result.model_dump()}")

            # get feature importance
            feature_importances = trained_model.get_feature_importance()
        
        step_logger.debug(f"Feature importances: {feature_importances.to_dict()}")

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

        return metadata

    def _run_train_predict(
        self, 
        partition_item: PartitionItem, 
        step: ModelStep, 
        df: DataFrame,
    ) -> ModelStepPredictionMetadata:
        df = df.copy()
        step_logger = logger.getChild(step.name + "_TRAIN_PREDICT")

        # --- Setup ---
        pb_col = step.get_prediction_base_col(partition_item.partition, partition_item.value)
        cat_cols = get_cat_cols(self.level)
        num_cols = get_num_cols(partition_item.partition, partition_item.value, step)
        boolean_cols = BOOLEAN_COLS
        model_status_col = get_model_status_col(step)
        target_cols = step.target_cols
        
        # Common columns for clean result DataFrames
        key_cols = get_join_keys(self.level)

        logging.info(f"--------------------------------")
        logging.info(f"Key cols: {key_cols}")
        logging.info(f"Num cols: {num_cols}")
        logging.info(f"Group bys: {self.level.group_bys}")
        logging.info(f"Target cols: {target_cols}")

        # --- Create masks and split data ---
        training_mask = self._get_training_mask(df, step, step_logger)
        prediction_mask = self._get_prediction_mask(df, step, pb_col, step_logger)
        
        train_df = cast(DataFrame, df[training_mask].copy())
        predict_df = cast(DataFrame, df[prediction_mask].copy())
        passthrough_df = cast(DataFrame, df[~prediction_mask].copy())
        
        step_logger.info(f"Predicting on {len(predict_df)} rows")

        # --- Handle empty prediction case ---
        if len(predict_df) == 0:
            step_logger.info("No rows require prediction - all data is baked")
            passthrough_df[model_status_col] = training_mask.loc[passthrough_df.index].map(
                {True: STATUS_TRAINED, False: STATUS_BYPASSED}
            )
            
            return ModelStepPredictionMetadata(
                model_id=self.model_id,
                prediction_rows=0,
                partition_item=partition_item,
                step=step,
                cat_cols=cat_cols,
                num_cols=num_cols,
                boolean_cols=boolean_cols,
                output_df=passthrough_df,
                result_df=DataFrame(),
            )

        # --- Run prediction pipeline ---
        if step.is_multi_target:
            result_df, _ = run_multi_output_pipeline(
                train_df, predict_df,
                target_cols=target_cols,
                cat_cols=cat_cols,
                num_cols=num_cols,
                boolean_cols=boolean_cols,
            )
            
            # Build clean result DataFrame with all target predictions
            result_clean_df = self._build_multi_target_result_df(
                source_df=result_df,
                key_cols=key_cols,
                target_cols=target_cols,
                base_values=predict_df[pb_col].values,
            )
            
            # Update predict_df with all predicted values
            passthrough_df[model_status_col] = training_mask.loc[passthrough_df.index].map(
                {True: STATUS_TRAINED, False: STATUS_BYPASSED}
            )
            predict_df[model_status_col] = STATUS_PREDICTED
            
            # Fill in target columns with predictions
            for target_col in target_cols:
                predict_df[target_col] = result_df[f"{target_col}_YHAT"].values
            
            # Fill survived columns with predicted rate * prediction base
            survived_cols = step.rate_target_survived_cols
            for i, (target_col, survived_col) in enumerate(zip(target_cols, survived_cols)):
                predict_df[survived_col] = result_df[f"{target_col}_YHAT"].values * predict_df[pb_col]
        else:
            target_col = target_cols[0]
            result_df, _ = run_pipeline(
                train_df, predict_df,
                target_col=target_col,
                cat_cols=cat_cols,
                num_cols=num_cols,
                boolean_cols=boolean_cols,
            )

            # Build clean result DataFrame for storage
            result_clean_df = self._build_clean_result_df(
                source_df=result_df,
                key_cols=key_cols,
                yhat_col=PRED_YHAT_COL,
                base_values=predict_df[pb_col].values,
            )

            # Build output DataFrame for next step
            passthrough_df[model_status_col] = training_mask.loc[passthrough_df.index].map(
                {True: STATUS_TRAINED, False: STATUS_BYPASSED}
            )
            predict_df[model_status_col] = STATUS_PREDICTED
            predict_df[target_col] = result_df[PRED_YHAT_COL].values
            
            survived_cols = step.rate_target_survived_cols
            if survived_cols:
                # fill the survived column with the predicted rate * prediction base
                predict_df[survived_cols[0]] = result_df[PRED_YHAT_COL].values * predict_df[pb_col]

            if step.is_retention_target:
                # fill the eligible column with the prediction base so we can recalculate the rate in aggregate
                predict_df[step.min_cohort_col] = predict_df[pb_col]
        
        output_df = cast(DataFrame, concat([passthrough_df, predict_df]))

        return ModelStepPredictionMetadata(
            model_id=self.model_id,
            prediction_rows=len(predict_df),
            partition_item=partition_item,
            step=step,
            cat_cols=cat_cols,
            num_cols=num_cols,
            boolean_cols=boolean_cols,
            output_df=output_df,
            result_df=result_clean_df,
        )
    
    def _build_multi_target_result_df(
        self,
        source_df: DataFrame,
        key_cols: list[str],
        target_cols: list[str],
        base_values: Any,
    ) -> DataFrame:
        """Build a clean result DataFrame for multi-target predictions."""
        # Start with key columns
        clean_df = cast(DataFrame, source_df[key_cols].copy())
        
        # Add prediction columns for each target
        for target_col in target_cols:
            yhat_col = f"{target_col}_YHAT"
            clean_df[yhat_col] = source_df[yhat_col].values
            clean_df[f"{target_col}_YHAT_LOWER"] = source_df[f"{target_col}_YHAT_LOWER"].values
            clean_df[f"{target_col}_YHAT_UPPER"] = source_df[f"{target_col}_YHAT_UPPER"].values
        
        clean_df[PREDICTION_BASE_COL] = base_values
        
        return clean_df

    def _get_prediction_mask(
        self, df: DataFrame, step: ModelStep, pb_col: str, step_logger: logging.Logger
    ) -> Series:
        """Create mask for rows that need prediction (not baked)."""
        min_cohort_col = step.min_cohort_col
        step_logger.info(
            f"Adding mask {min_cohort_col} / {pb_col} < {PREDICTION_BASE_THRESHOLD} "
            f"or {min_cohort_col} is na or {pb_col} is na"
        )
        return (
            ((df[min_cohort_col] < MIN_COHORT_SIZE) 
             & (df[min_cohort_col] / df[pb_col] < PREDICTION_BASE_THRESHOLD))
            | df[min_cohort_col].isna()
            | df[pb_col].isna()
        )

    def _build_clean_result_df(
        self,
        source_df: DataFrame,
        key_cols: list[str],
        yhat_col: str,
        base_values: Any,
    ) -> DataFrame:
        """Build a clean result DataFrame with consistent columns for storage."""
        clean_df = cast(DataFrame, source_df[key_cols + [yhat_col]].copy())
        
        # Normalize column name to YHAT
        if yhat_col != PRED_YHAT_COL:
            clean_df.rename(columns={yhat_col: PRED_YHAT_COL}, inplace=True)
        
        # Add bounds (only present for predicted rows)
        clean_df[PRED_YHAT_LOWER_COL] = source_df[PRED_YHAT_LOWER_COL].values
        clean_df[PRED_YHAT_UPPER_COL] = source_df[PRED_YHAT_UPPER_COL].values
        
        clean_df[PREDICTION_BASE_COL] = base_values
        
        return clean_df

    def _build_dataset_df(self, final_df: DataFrame) -> DataFrame:
        """Build a clean dataset for dashboard consumption."""
        df = final_df.copy()
        
        # Add model_id for filtering by model run
        df['MODEL_ID'] = self.model_id
        
        # Determine IS_PREDICTED from model status columns
        status_cols = [get_model_status_col(step) for step in model_steps]
        existing_status_cols = [c for c in status_cols if c in df.columns]
        df[IS_PREDICTED_COL] = df[existing_status_cols].eq(STATUS_PREDICTED).any(axis=1)
        
        # Convert avg billing cols to totals using time horizons
        for time_horizon in time_horizons:
            avg_col = get_avg_net_billings_col(time_horizon)
            total_col = get_total_net_billings_col(time_horizon)
            if avg_col in df.columns:
                df[total_col] = df[avg_col] * df[GROSS_ADDS_COL]
        
        # Transform numerical columns (avg_* or *_rate) to totals
        all_num_cols = set(c.upper() for c in NUM_COLS)
        for partition in partitions:
            for value in partition.values:
                all_num_cols.update(c.upper() for c in partition.get_additional_regressor_cols(value))

        transformed_num_cols = []
        for col in all_num_cols:
            if col not in df.columns:
                continue
            if col.startswith('AVG_') or col.endswith('_RATE'):
                new_col = col.removeprefix('AVG_').removesuffix('_RATE')
                df[new_col] = df[col] * df[GROSS_ADDS_COL]
                transformed_num_cols.append(new_col)
        
        # Select final columns
        key_cols = get_join_keys(self.level)
        
        # Get retention metric columns dynamically
        retention_cols = []
        
        # Add day-1 baked columns (no model prediction needed)
        day_1_cols = [
            get_gross_adds_created_over_days_ago_col(1),  # GROSS_ADDS_CREATED_OVER_1_DAYS_AGO
            GROSS_ADDS_CANCELED_DAY_ONE_COL,
            CROSS_SELL_ADDS_DAY_ONE_COL,
        ]
        for col in day_1_cols:
            if col in df.columns:
                retention_cols.append(col)
        
        # Add columns from model steps (day 3, 7, etc.)
        for step in model_steps:
            if step.min_cohort_col and step.min_cohort_col in df.columns:
                retention_cols.append(step.min_cohort_col)
            for survived_col in step.rate_target_survived_cols:
                if survived_col in df.columns:
                    retention_cols.append(survived_col)
        retention_cols = list(dict.fromkeys(retention_cols))  # dedupe preserving order
        
        # Get total billing columns
        total_billing_cols = [c for c in df.columns if c.startswith('TOTAL_NET_BILLINGS')]
        
        output_cols = (
            ['MODEL_ID']
            + key_cols
            + [IS_PREDICTED_COL, GROSS_ADDS_COL] 
            + retention_cols 
            + sorted(transformed_num_cols)
            + sorted(total_billing_cols)
        )
        # Dedupe while preserving order (retention_cols take priority over transformed_num_cols)
        output_cols = list(dict.fromkeys(output_cols))
        
        return cast(DataFrame, df[[c for c in output_cols if c in df.columns]])

    def _get_training_mask(self, df: DataFrame, step: ModelStep, step_logger: logging.Logger) -> Series:       
        # create training mask
        training_mask = Series(True, index=df.index)
        training_mask &= df[step.min_cohort_col] >= MIN_COHORT_SIZE
        
        # check for nulls across all target columns
        target_cols = step.target_cols
        null_mask = Series(False, index=df.index)
        for target_col in target_cols:
            col_null_mask: Series = cast(Series, df[target_col]).isna()
            null_mask |= col_null_mask
        
        training_null_df = cast(DataFrame, df[null_mask & training_mask])
        if len(training_null_df) > 0:
            group_by_cols = self.level.group_bys
            # Add null values to collector for debugging
            self._collector.add(f'{self.level.name.upper()}_NULL_VALUES', training_null_df)
            logger.warning(
                f"Target columns {target_cols} have {len(training_null_df)} null values in {len(training_null_df.groupby(group_by_cols).size())} unique {group_by_cols} combinations"
            )
            training_mask &= ~null_mask
        step_logger.info(f"Training mask: {training_mask.sum()} rows")
        return training_mask

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

    def _add_model_metadata_to_collector(self, status: ModelStatus, message: str | None = None):
        model_metadata = ModelMetadata(
            id=self.model_id,
            created_at=self.model_created_at,
            level_name=self.level.name,
            level_group_bys=self.level.group_bys,
            status=status,
            message=message,
        )
        self._collector.add(TABLE_MODEL_METADATA, model_metadata.to_dataframe())

    def _add_df_to_collector(self, table_name: str, df: DataFrame):
        file_name = f'{self.level.name.upper()}_{table_name}'
        self._collector.add(file_name, df.reset_index(drop=True))
    
    def _flush_results(self):
        """Flush all collected results to the writer."""
        if self.writer is None:
            logger.info("No writer configured, skipping result flush")
            return
        
        logger.info(f"Flushing {self._collector.pending_count} result sets to writer")
        self._collector.flush(self.writer)
