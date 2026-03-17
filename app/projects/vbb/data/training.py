from src.initialize import load
load()

from datetime import datetime, timedelta
import os
import logging
from snowflake.snowpark import Session
from pandas import DataFrame
import pandas as pd
from typing import cast

from src.environment import environment as env
from projects.vbb.data.query import QUERY
from projects.vbb.data import CACHE_PATH
from projects.vbb.constants import (
    TRAINING_WINDOW_DAYS, 
    TRAINING_LOOKBACK_DAYS, 
    PREDICTION_CENSOR_HOURS,
    HOURS_TO_CANCEL_COL,
)


logger = logging.getLogger(__name__)


TRAINING_SET_CACHE_PATH = os.path.join(CACHE_PATH, "training_set.parquet")
SAMPLE_SET_CACHE_PATH = os.path.join(CACHE_PATH, "sample_set.csv")


def get_training_df(session: Session, force_refresh: bool = False) -> DataFrame:
    use_cache = env.target.is_dev and env.use_cache
    if use_cache and os.path.exists(TRAINING_SET_CACHE_PATH) and not force_refresh:
        logging.info(f"Loading training set from cache: {TRAINING_SET_CACHE_PATH}")
        return pd.read_parquet(TRAINING_SET_CACHE_PATH)
    filtered_df = _get_training_set(session)
    if use_cache:
        _cache_df(filtered_df)
    return filtered_df


def _get_training_set(session: Session) -> DataFrame:
    from_date = _get_start_date()
    to_date = _get_end_date()
    logger.info(f"Getting training set from {from_date} to {to_date}")
    formatted_query = QUERY.format(
        from_time=from_date,
        to_time=to_date,
    )
    df: DataFrame = session.sql(formatted_query).to_pandas()
    filtered_df = _filter_df(df)
    logger.info(f"Training set shape: {filtered_df.shape}")
    return filtered_df

def _get_start_date():
    """Get the start date for the training set. LOOKBACK_DAYS + WINDOW_DAYS."""
    return (
        datetime.now() - (
            timedelta(days=TRAINING_LOOKBACK_DAYS) + timedelta(days=TRAINING_WINDOW_DAYS)
        )
    ).strftime("%Y-%m-%d")

def _get_end_date():
    """Get the end date for the training set. LOOKBACK_DAYS."""
    return (
        datetime.now() - timedelta(days=TRAINING_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")


def _cache_df(df: DataFrame) -> None:
    """Cache the training set to the local filesystem."""
    df.to_parquet(TRAINING_SET_CACHE_PATH)
    df.sample(1000).to_csv(SAMPLE_SET_CACHE_PATH)


def _filter_df(df: DataFrame) -> DataFrame:
    """Only keep rows where the HOURS_TO_CANCEL_COL is null or greater than or equal to PREDICTION_CENSOR_HOURS"""
    return cast(DataFrame, df[df[HOURS_TO_CANCEL_COL].isnull() | (df[HOURS_TO_CANCEL_COL] >= PREDICTION_CENSOR_HOURS)])


if __name__ == "__main__":
    from projects.vbb import get_session
    session = get_session()
    df = get_training_df(session, force_refresh=True)
    print(df.head())