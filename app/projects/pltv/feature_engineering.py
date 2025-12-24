import pandas as pd
import logging
from projects.pltv import config

logger = logging.getLogger(__name__)

def _timestamp_col_to_datetime(df: pd.DataFrame) -> None:
    logger.info(f"Converting {config.timestamp_col} to datetime")
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])

def _convert_net_billings_to_avg_net_billings(df: pd.DataFrame) -> None:
    logger.info(f"Converting net billings to avg net billings for {len(config.time_horizons)} time horizons")
    for time_horizon in config.time_horizons:
        # get col name from config
        ga_col = config.get_gross_adds_created_over_days_ago_column(time_horizon)
        nb_col = config.get_net_billings_days_column(time_horizon)
        avg_nb_col = config.get_avg_net_billings_column(time_horizon)
        # convert net billings to avg net billings
        df[avg_nb_col] = df[nb_col] / df[ga_col]
        # drop columns
        # df.drop(columns=[ga_col], inplace=True)
        df.drop(columns=[nb_col], inplace=True)


def clean_df(df: pd.DataFrame) -> None:
    _timestamp_col_to_datetime(df)
    _convert_net_billings_to_avg_net_billings(df)