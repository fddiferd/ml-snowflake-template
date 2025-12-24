import logging
import pandas as pd
from typing import cast

from projects.pltv import config, ModelStep
from projects.pltv.objects import Level
from src.pipeline.xgboost import run_pipeline


logger = logging.getLogger(__name__)


def run_step(level: Level, df: pd.DataFrame, step: ModelStep, test: bool):
    logger.name = f"run_step_{step.name}"
    df = df.copy()

    # create training mask
    training_mask = pd.Series(True, index=df.index)
    for col in step.previous_step_min_cohort_cols:
        logger.info(f"Adding mask {col} >= {config.min_cohort_size}")
        training_mask &= df[col] >= config.min_cohort_size

    train_df = df[training_mask]
    
    if test:
        from sklearn.model_selection import train_test_split
        # split train_df into train and test
        train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)
        logger.info(f"Training on {len(train_df)} rows and testing on {len(test_df)} rows")

        # run pipeline
        result_df, trained_model = run_pipeline(
            cast(pd.DataFrame, train_df),
            cast(pd.DataFrame, test_df),
            target_col=step.target_col,
            cat_cols=config.cat_cols,
            num_cols=config.num_cols,
        )

        # evaluate model (result_df has both actual and predicted with aligned indices)
        from src.utils.model import evaluate_model
        y_true = cast(pd.Series, result_df[step.target_col])
        y_pred = cast(pd.Series, result_df[f'PRED_{step.target_col}'])
        eval_result = evaluate_model(y_true, y_pred)
        logger.info(f"Evaluation result: {eval_result}")
    else:
        raise ValueError("Non Test mode not supported")


def run(level: Level, df: pd.DataFrame, test: bool = False):

    for col in df.columns:
        if col.startswith('AVG_NET_BILLINGS_'):
            print(col)


    for partition_item in config.partition.items:
        # partition the data
        logger.info(f"Partitioning data where {config.partition.name} = {partition_item.value}")
        partition_df = df[df[config.partition.name] == partition_item.value]

        for step in config.model_steps:
            logger.info(f"Running step {step.name}")
            run_step(level, cast(pd.DataFrame, partition_df), step, test)

        