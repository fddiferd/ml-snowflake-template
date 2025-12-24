import pandas as pd
import logging
from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from src.services.feature_store_service import FeatureStoreService

from projects import Project
from projects.pltv.core.config import config, fv_configs
from projects.pltv.core.base_models import Level
from projects.pltv.data.queries.spine import QUERY as SPINE_QUERY


CACHE_FILE_NAME = "output_dataset.parquet"


logger = logging.getLogger(__name__)


def get_dataset(session: Session, level: Level) -> DataFrame:
    # init feature store service
    svc = FeatureStoreService(
        session, 
        name=Project.PLTV.schema_name, 
        timestamp_col=config.timestamp_col
    )
    # spine
    spine_df = session.sql(SPINE_QUERY.format(group_bys=level.sql_fields))
    svc.set_spine(spine_df)
    # entity
    svc.set_entity(
        join_keys=level.get_join_keys(config.timestamp_col, config.partition.name if config.partition else None), 
        name=f'{level.name}_ENTITY',
        recreate=True
    )
    # feature views
    for feature_view_config in fv_configs:
        logger.info(f'Setting feature view {feature_view_config.name} for level {level.name}')
        feature_df = session.sql(
            feature_view_config.query.format(group_bys=level.sql_fields)
        )
        svc.set_feature_view(
            feature_df, 
            config.version_number, 
            name=f'{level.name}_{feature_view_config.name}',
        )
    # get dataset
    return svc.get_dataset()

def get_df(session: Session, level: Level, save_to_cache: bool = False) -> pd.DataFrame:
    dataset = get_dataset(session, level)
    df = pd.DataFrame(dataset)
    if save_to_cache:
        df.to_parquet(CACHE_FILE_NAME)
    return df

def get_df_from_cache() -> pd.DataFrame:
    try:
        return pd.read_parquet(CACHE_FILE_NAME)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache file {CACHE_FILE_NAME} not found")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from projects.pltv import get_session

    session = get_session()
    dataset = get_dataset(session, Level(group_bys=["brand", "sku_type", "channel"]))
    df = dataset.to_pandas()
    df.to_parquet("output_dataset.parquet")