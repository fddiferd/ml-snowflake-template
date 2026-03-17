from src.initialize import load
load()

from datetime import datetime, timedelta
import os
import logging
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from pandas import DataFrame
import pandas as pd

from src.environment import environment as env
from projects.vbb.data.query import QUERY
from projects.vbb.data import CACHE_PATH, split_df_by_cancelation, cache_df
from projects.vbb.constants import (
    PREDICTION_RESULTS_TABLE_NAME,
    INITIAL_PREDICTION_FALLBACK_DAYS,
    TIME_COL,
    GCLID_COL,
)


logger = logging.getLogger(__name__)


PREDICTION_SET_CACHE_PATH = os.path.join(CACHE_PATH, "prediction_set.parquet")
PREDICTION_SAMPLE_SET_CACHE_PATH = os.path.join(CACHE_PATH, "prediction_sample_set.csv")
CANCELED_SET_CACHE_PATH = os.path.join(CACHE_PATH, "canceled_set.parquet")
CANCELED_SAMPLE_SET_CACHE_PATH = os.path.join(CACHE_PATH, "canceled_sample_set.csv")
VALID_GCLID_PREFIXES = ('Cj', 'EA')


def get_prediction_dfs(session: Session, force_refresh: bool = False) -> tuple[DataFrame, DataFrame]:
    use_cache = env.target.is_dev and env.use_cache
    if use_cache and os.path.exists(PREDICTION_SET_CACHE_PATH) and os.path.exists(CANCELED_SET_CACHE_PATH) and not force_refresh:
        logging.info(f"Loading prediction set from cache: {PREDICTION_SET_CACHE_PATH}")
        return pd.read_parquet(PREDICTION_SET_CACHE_PATH), pd.read_parquet(CANCELED_SET_CACHE_PATH)
    prediction_df, canceled_df = _get_prediction_dfs(session)
    if use_cache:
        cache_df(prediction_df, PREDICTION_SET_CACHE_PATH, PREDICTION_SAMPLE_SET_CACHE_PATH)
        cache_df(canceled_df, CANCELED_SET_CACHE_PATH, CANCELED_SAMPLE_SET_CACHE_PATH)
    return prediction_df, canceled_df


def _get_prediction_dfs(session: Session) -> tuple[DataFrame, DataFrame]:
    from_date = _get_start_date(session)
    to_date = datetime.now()
    logger.info(f"Getting prediction set from {from_date} to {to_date}")
    formatted_query = QUERY.format(
        from_time=from_date,
        to_time=to_date,
    )
    df: DataFrame = session.sql(formatted_query).to_pandas()
    df = _filter_valid_gclids(df)
    prediction_df, canceled_df = split_df_by_cancelation(df)
    logger.info(f"Prediction set shape: {prediction_df.shape}")
    logger.info(f"Canceled set shape: {canceled_df.shape}")
    return prediction_df, canceled_df


def _filter_valid_gclids(df: DataFrame) -> DataFrame:
    """Keep only rows with a valid Google Click ID (starts with Cj or EAI)."""
    before = len(df)
    has_gclid = df[GCLID_COL].notna() & df[GCLID_COL].astype(str).str.startswith(VALID_GCLID_PREFIXES)
    df = DataFrame(df[has_gclid])
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Filtered {dropped:,} rows without valid GCLID ({len(df):,} remaining)")
    return df


def _get_start_date(session: Session):
    """Select the max gross add created date from the prediction results table."""
    try:
        return session.sql(f"SELECT MAX({TIME_COL}) FROM {PREDICTION_RESULTS_TABLE_NAME}").to_pandas()["MAX(GROSS_ADD__CREATED)"].max()
    except SnowparkSQLException as e:
        if "does not exist or not authorized" in str(e):
            fallback = (datetime.now() - timedelta(days=INITIAL_PREDICTION_FALLBACK_DAYS)).strftime("%Y-%m-%d")
            logger.warning(f"Results table not found, falling back to {fallback}")
            return fallback
        raise e


if __name__ == "__main__":
    from projects.vbb import get_session
    session = get_session()
    df, canceled_df = get_prediction_dfs(session)
    print(df.shape)
    print(canceled_df.shape)