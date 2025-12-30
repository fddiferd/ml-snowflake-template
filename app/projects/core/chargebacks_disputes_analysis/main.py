"""
Timeframe: Last 6 months
Filters: brand / channels / traffic sources / creatives / bins / plan offers
Insights: Areas that have a higher risk of fraud chargeback + disputes
"""

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

import logging
from enum import Enum
from pandas import DataFrame, Series, concat, read_parquet
from typing import cast, TypeAlias
from itertools import combinations
from snowflake.snowpark import Session

from projects.core.connection import get_session
from projects.core.chargebacks_disputes_analysis.data import queries, get_file_path


logger = logging.getLogger(__name__)

# variables
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
    SETTLEMENT = "settlement"


class Level(Enum):
    PROCESSOR = "PROCESSOR"
    MERCHANT_ACCOUNT = "MERCHANT_ACCOUNT"
    CARD_NETWORK = "CARD_NETWORK"
    BRAND = "BRAND_SLUG"
    SKU_TYPE = "SKU_TYPE_SLUG"
    CHANNEL = "CHANNEL_SLUG"
    TRAFFIC_SOURCE = "TRAFFIC_SOURCE_SHORT_NAME"
    OFFER_TYPE = "OFFER_TYPE"
    BIN = "BIN"

Levels: TypeAlias = list[Level]
levels = [l for l in Level]
all_level_combinations = [list(combo) for i in range(2, len(levels) + 1) for combo in combinations(levels, i)]


class Service:
    def __init__(
        self,
        min_fraud_discounts_count: int = MIN_FRAUD_DISPUTES_COUNT,
        use_cache: bool = False,
    ):
        self.min_fraud_discounts_count = min_fraud_discounts_count
        self.use_cache = use_cache

        self.session: Session | None = None
        self.df = DataFrame()

    # Data
    def set_df(self, date: str) -> DataFrame:
        if self.use_cache:
            try:
                logger.info(f"Trying to load data from cache {get_file_path(date)}")
                self.df = self._get_df_from_cache(date)
            except FileNotFoundError:
                logger.warning(f"Cache file not found, querying data from snowflake")
                self.df = self._query_data(date)
        else:
            self.df = self._query_data(date)
        return self.df

    def _query_data(self, date: str) -> DataFrame:
        logger.info(f"Querying data from snowflake")
        session = self._get_session()
        query_count = len(queries)
        for index, query in enumerate(queries):
            query_formatted = query.format(date=date)
            logger.info(f"Running query {index + 1} of {query_count}")
            self.df = concat([
                self.df, 
                session.sql(query_formatted).to_pandas()
            ])
        # clean data
        self.df = cast(DataFrame, self.df)
        self.df['CARD_NETWORK'] = self.df['CARD_NETWORK'].str.lower().str.replace(" ", "")
        # remove paypal_account
        self.df = self.df[self.df['CARD_NETWORK'] != 'paypal_account']
        self.df = cast(DataFrame, self.df.reset_index(drop=True))
        # alias 'mc' to 'mastercard'
        self.df['CARD_NETWORK'] = self.df['CARD_NETWORK'].replace('mc', 'mastercard')
        # save to cache
        self.df.to_parquet(get_file_path(date))
        logger.info(f"Saved data to cache {get_file_path(date)}")
        return self.df

    def _get_session(self) -> Session:
        if self.session is None:
            self.session = get_session()
        return self.session

    def _get_df_from_cache(self, date: str) -> DataFrame:
        try:
            return read_parquet(get_file_path(date))
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache file {get_file_path(date)} not found")

    def _get_df(self) -> DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty")
        return cast(DataFrame, self.df)

    def get_ratio_df(self, level: Level | Levels) -> DataFrame:
        logger.info(f"Computing ratio for {level}")
        df = self._get_df()

        group_by_list = (
            [level.value] if isinstance(level, Level) else [l.value for l in level]
        )

        # group by level and calculate ratio (dropna=False to include None values)
        result = df.groupby(group_by_list, dropna=False).apply(
            lambda g: DataFrame({
                FRAUD_DISPUTES_COUNT_FIELD: [g[g[TYPE_FIELD].isin([Type.FRAUD.value, Type.DISPUTE.value])][COUNT_FIELD].sum()],
                SETTLED_COUNT_FIELD: [g[g[TYPE_FIELD] == Type.SETTLEMENT.value][COUNT_FIELD].sum()],
            }),
            include_groups=False
        ).reset_index(level=len(group_by_list), drop=True).reset_index()

        # fill Missing Levels with Missing
        for group_by in group_by_list:
            result[group_by] = result[group_by].fillna(f'MISSING_{group_by}')
        
        # calculate ratio
        result[RATIO_FIELD] = result[FRAUD_DISPUTES_COUNT_FIELD] / result[SETTLED_COUNT_FIELD]
        
        # Global ratio across all groups
        global_fraud_disputes = result[FRAUD_DISPUTES_COUNT_FIELD].sum()
        global_settled = result[SETTLED_COUNT_FIELD].sum()
        result[GLOBAL_RATIO_FIELD] = global_fraud_disputes / global_settled

        # add deltas
        result[DELTA_FIELD] = result[RATIO_FIELD] - result[GLOBAL_RATIO_FIELD]
        result[DELTA_PERCENTAGE_FIELD] = result[RATIO_FIELD] / result[GLOBAL_RATIO_FIELD]
        
        return result.sort_values(by=DELTA_FIELD, ascending=False)

    def get_threshold_levels_df(self, level: Level | Levels) -> DataFrame:
        logger.info(f"Computing threshold for {level}")
        # get ratio df
        ratio_df = self.get_ratio_df(level)

        # add level column as first column
        level_field_value = level.value if isinstance(level, Level) else '-'.join([l.value for l in level])
        ratio_df.insert(0, LEVEL_FIELD, level_field_value)

        # if multiple levels, concat the field values and remove level cols
        if isinstance(level, list):
            level_cols = [l.value for l in level]
            ratio_df.insert(
                1, 
                VALUE_FIELD, 
                cast(Series, ratio_df[level_cols].astype(str).apply(lambda x: '-'.join(x), axis=1))
            )
            ratio_df.drop(columns=level_cols, inplace=True)
        else:
            # rename level column to VALUE
            ratio_df.rename(columns={level.value: VALUE_FIELD}, inplace=True)

        # filter to keep on items that meet threshold
        return cast(DataFrame, ratio_df[
                (ratio_df[FRAUD_DISPUTES_COUNT_FIELD] > MIN_FRAUD_DISPUTES_COUNT)
            ]
        ).sort_values(by=DELTA_PERCENTAGE_FIELD, ascending=False).reset_index(drop=True)

    def get_threshold_all_levels_df(self, save_csv: bool = False) -> DataFrame:
        logger.info(f"Computing threshold for all levels")
        output_df = DataFrame()
        for level in levels:
            threshold_df = self.get_threshold_levels_df(level)
            if threshold_df.empty:
                continue
            output_df = concat([output_df, threshold_df])
        if save_csv:
            logger.info(f"Saving threshold for all levels to {get_file_path('threshold_all_levels', extension='csv')}")
            output_df.to_csv(get_file_path('threshold_all_levels', extension='csv'))
        return output_df


def main():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    logging.basicConfig(level=logging.INFO)

    svc = Service(use_cache=True)
    date = (datetime.now() - relativedelta(months=6)).strftime("%Y-%m-%d")
    svc.set_df(date)
    svc.get_threshold_all_levels_df(save_csv=True)


if __name__ == "__main__":
    main()

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

# def get_combination_threshold_df(df: DataFrame) -> DataFrame:
#     total = len(all_level_combinations)
    
#     def process_combination(combination: list) -> DataFrame | None:
#         threshold_df = get_threshold_levels_df(df, combination)
#         return threshold_df if not threshold_df.empty else None
    
#     results: list[DataFrame] = []
#     with ThreadPoolExecutor() as executor:
#         futures = {executor.submit(process_combination, combo): combo for combo in all_level_combinations}
#         for future in tqdm(as_completed(futures), total=total, desc="Processing combinations"):
#             result = future.result()
#             if result is not None:
#                 results.append(result)
    
#     return concat(results) if results else DataFrame()
