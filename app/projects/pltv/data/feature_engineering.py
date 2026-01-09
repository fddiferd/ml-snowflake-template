import pandas as pd
import logging

from projects.pltv.config import (
    TIMESTAMP_COL,
    time_horizons,
    get_gross_adds_created_over_days_ago_col,
    get_net_billings_col,
    get_avg_net_billings_col,
    # Cross-sell columns
    CROSS_SELL_ADDS_DAY_ONE_COL,
    CROSS_SELL_ADDS_DAY_THREE_COL,
    CROSS_SELL_ADDS_DAY_SEVEN_COL,
    CROSS_SELL_ADDS_DAY_ONE_RATE_COL,
    CROSS_SELL_ADDS_DAY_THREE_RATE_COL,
    CROSS_SELL_ADDS_DAY_SEVEN_RATE_COL,
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


def _convert_cross_sell_to_rates(df: pd.DataFrame) -> None:
    """Convert cross-sell counts to rates using gross_adds_created_over_N_days_ago."""
    logger.info("Converting cross-sell counts to rates")
    mappings = [
        (CROSS_SELL_ADDS_DAY_ONE_COL, 1, CROSS_SELL_ADDS_DAY_ONE_RATE_COL),
        (CROSS_SELL_ADDS_DAY_THREE_COL, 3, CROSS_SELL_ADDS_DAY_THREE_RATE_COL),
        (CROSS_SELL_ADDS_DAY_SEVEN_COL, 7, CROSS_SELL_ADDS_DAY_SEVEN_RATE_COL),
    ]
    for count_col, days, rate_col in mappings:
        denom_col = get_gross_adds_created_over_days_ago_col(days)
        if count_col in df.columns and denom_col in df.columns:
            df[rate_col] = df[count_col] / df[denom_col]
            logger.debug(f"Created {rate_col} = {count_col} / {denom_col}")


def clean_df(df: pd.DataFrame) -> None:
    _timestamp_col_to_datetime(df)
    _convert_net_billings_to_avg_net_billings(df)
    _convert_cross_sell_to_rates(df)
    # log columns
    logger.info(f"Cleaned columns: {df.columns.tolist()}")
