import pandas as pd
import logging

from projects.pltv.config import (
    TIMESTAMP_COL,
    time_horizons,
    get_gross_adds_created_over_days_ago_col,
    get_net_billings_col,
    get_avg_net_billings_col,
)


logger = logging.getLogger(__name__)


def _timestamp_col_to_datetime(df: pd.DataFrame) -> None:
    logger.info(f"Converting {TIMESTAMP_COL} to datetime")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])


def _convert_net_billings_to_avg_net_billings(df: pd.DataFrame) -> None:
    logger.info(f"Converting net billings to avg net billings for {len(time_horizons)} time horizons")
    for time_horizon in time_horizons:
        # get col names from config
        ga_col = get_gross_adds_created_over_days_ago_col(time_horizon.value)
        nb_col = get_net_billings_col(time_horizon)
        avg_nb_col = get_avg_net_billings_col(time_horizon)
        # convert net billings to avg net billings
        df[avg_nb_col] = df[nb_col] / df[ga_col]
        # drop the raw billings column
        df.drop(columns=[nb_col], inplace=True)


def clean_df(df: pd.DataFrame) -> None:
    _timestamp_col_to_datetime(df)
    _convert_net_billings_to_avg_net_billings(df)
    # log columns
    logger.info(f"Cleaned columns: {df.columns.tolist()}")
