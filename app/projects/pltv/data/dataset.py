if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

import pandas as pd
import logging
from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from src.services.feature_store_service import FeatureStoreService

from projects import Project
from projects.pltv.config import (
    Level,
    TIMESTAMP_COL,
    VERSION_NUMBER,
    partition_sql_fields,
    fv_configs,
    get_join_keys,
)
from projects.pltv.data.queries.spine import QUERY as SPINE_QUERY
from projects.pltv.data.utils import get_file_path


logger = logging.getLogger(__name__)


def get_dataset(session: Session, level: Level) -> DataFrame:
    # init feature store service
    svc = FeatureStoreService(
        session, 
        name=Project.PLTV.name, 
        timestamp_col=TIMESTAMP_COL
    )
    # spine
    spine_sql = SPINE_QUERY.format(
        timestamp_col=TIMESTAMP_COL,
        group_bys=level.sql_fields,
        partitions=partition_sql_fields,
    )
    logger.info(f"Spine SQL: {spine_sql}")
    spine_df = session.sql(spine_sql)
    svc.set_spine(spine_df)
    # entity
    svc.set_entity(
        join_keys=get_join_keys(level), 
        name=f'{level.name}_ENTITY',
        recreate=True
    )
    # feature views
    for feature_view_config in fv_configs:
        logger.info(f'Setting feature view {feature_view_config.name} for level {level.name}')
        feature_sql = feature_view_config.query.format(
            timestamp_col=TIMESTAMP_COL,
            group_bys=level.sql_fields,
            partitions=partition_sql_fields
        )
        logger.info(f"Feature SQL: {feature_sql}")
        feature_df = session.sql(feature_sql)
        svc.set_feature_view(
            feature_df, 
            VERSION_NUMBER, 
            name=f'{level.name}_{feature_view_config.name}',
        )
    # get dataset
    return svc.get_dataset()

def get_df(session: Session, level: Level, save_to_cache: bool = False) -> pd.DataFrame:
    dataset = get_dataset(session, level)
    df = dataset.to_pandas()
    if save_to_cache:
        df.to_parquet(get_file_path(level.name))
    return df

def get_df_from_cache(level: Level) -> pd.DataFrame:
    try:
        return pd.read_parquet(get_file_path(level.name))
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache file {get_file_path(level.name)} not found")


if __name__ == "__main__":
    from projects.pltv import get_session

    session = get_session()
    get_df(session, Level.CHANNEL, save_to_cache=True)
