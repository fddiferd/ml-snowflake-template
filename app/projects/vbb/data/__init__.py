import os
from pandas import DataFrame
from typing import cast

from projects.vbb.constants import HOURS_TO_CANCEL_COL, PREDICTION_CENSOR_HOURS

CACHE_PATH = "app/projects/vbb/data/cache"

# create directory if it doesn't exist
os.makedirs(CACHE_PATH, exist_ok=True)



def split_df_by_cancelation(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
        Return the filtered data in a tuple, first where it meets the condition and where it doesnt 
        - HOURS_TO_CANCEL_COL is null or greater than or equal to PREDICTION_CENSOR_HOURS
    """
    filtered_df = cast(DataFrame, df[df[HOURS_TO_CANCEL_COL].isnull() | (df[HOURS_TO_CANCEL_COL] >= PREDICTION_CENSOR_HOURS)])
    not_filtered_df = cast(DataFrame, df[df[HOURS_TO_CANCEL_COL].notnull() & (df[HOURS_TO_CANCEL_COL] < PREDICTION_CENSOR_HOURS)])
    return (filtered_df, not_filtered_df)



def cache_df(df: DataFrame, parquet_path: str, csv_sample_path: str) -> None:
    """Cache the dataframe to the local filesystem."""
    df.to_parquet(parquet_path)
    df.sample(1000).to_csv(csv_sample_path)