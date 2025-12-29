"""
Timeframe: Last 6 months
Filters: brand / channels / traffic sources / creatives / bins / plan offers
Insights: Areas that have a higher risk of fraud chargeback + disputes
"""

from enum import Enum
from pandas import DataFrame, concat
from typing import cast, TypeAlias

# variables
DELTA_THRESHOLD = 0.30
MIN_FRAUD_DISPUTES_COUNT = 100

# input fields
TYPE_FIELD = "TYPE"
COUNT_FIELD = "COUNT"

# output fields
LEVEL_FIELD = "LEVEL"
VALUE_FIELD = "VALUE"
FRAUD_DISPUTES_COUNT_FIELD = "FRAUD_DISPUTES_COUNT"
SETTLED_COUNT_FIELD = "SETTLED_COUNT"
RATIO_FIELD = "RATIO"
GLOBAL_RATIO_FIELD = "GLOBAL_RATIO"
DELTA_FIELD = "DELTA"
DELTA_PERCENTAGE_FIELD = "DELTA_PERCENTAGE"

# enums
class Type(Enum):
    FRAUD = "fraud"
    DISPUTE = "dispute"
    SETTLED = "settled"


class Level(Enum):
    MERCHANT_ACCOUNT = "MERCHANT_ACCOUNT"
    BRAND = "BRAND_SLUG"
    CHANNEL = "CHANNEL_SLUG"
    TRAFFIC_SOURCE = "TRAFFIC_SOURCE_SHORT_NAME"
    OFFER_TYPE = "OFFER_TYPE"
    BIN = "BIN"

Levels: TypeAlias = list[Level]
levels = [l for l in Level]

# functions
def get_ratio_df(df: DataFrame, level: Level) -> DataFrame:
    # group by level and calculate ratio
    result = df.groupby(level.value).apply(
        lambda g: DataFrame({
            FRAUD_DISPUTES_COUNT_FIELD: [g[g[TYPE_FIELD].isin([Type.FRAUD.value, Type.DISPUTE.value])][COUNT_FIELD].sum()],
            SETTLED_COUNT_FIELD: [g[g[TYPE_FIELD] == Type.SETTLED.value][COUNT_FIELD].sum()],
        }),
        include_groups=False
    ).reset_index(level=1, drop=True).reset_index()
    
    # calculate ratio
    result[RATIO_FIELD] = result[FRAUD_DISPUTES_COUNT_FIELD] / result[SETTLED_COUNT_FIELD]
    
    # Global ratio across all groups
    global_fraud_disputes = result[FRAUD_DISPUTES_COUNT_FIELD].sum()
    global_settled = result[SETTLED_COUNT_FIELD].sum()
    result[GLOBAL_RATIO_FIELD] = global_fraud_disputes / global_settled

    # add deltas
    result[DELTA_FIELD] = result[RATIO_FIELD] - result[GLOBAL_RATIO_FIELD]
    result[DELTA_PERCENTAGE_FIELD] = result[DELTA_FIELD] / result[GLOBAL_RATIO_FIELD]
    # result['DELTA_PERCENTAGE'] = ((result['DELTA'] / result['GLOBAL_RATIO']) * 100).round(1).astype(str) + '%'
    
    return result.sort_values(by=DELTA_FIELD, ascending=False)

def get_threshold_levels_df(df: DataFrame, level: Level) -> DataFrame:
    # get ratio df
    ratio_df = get_ratio_df(df, level)

    # add level column as first column
    ratio_df.insert(0, LEVEL_FIELD, level.value)

    # rename level column to level
    ratio_df.rename(columns={level.value: VALUE_FIELD}, inplace=True)

    # filter to keep on items that meet threshold
    return cast(DataFrame, ratio_df[
            (ratio_df[DELTA_PERCENTAGE_FIELD] > DELTA_THRESHOLD)
            & (ratio_df[FRAUD_DISPUTES_COUNT_FIELD] > MIN_FRAUD_DISPUTES_COUNT)
        ]
    ).sort_values(by=DELTA_PERCENTAGE_FIELD, ascending=False)

def get_threshold_df(df: DataFrame) -> DataFrame:
    output_df = DataFrame()
    for level in levels:
        threshold_df = get_threshold_levels_df(df, level)
        if threshold_df.empty:
            continue
        output_df = concat([output_df, threshold_df])
    return output_df