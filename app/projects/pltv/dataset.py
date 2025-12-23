from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from projects import Project
from projects.pltv.objects import Level
from projects.pltv.queries.spine import QUERY as SPINE_QUERY
from projects.pltv.queries.feature_views import feature_view_configs

from src.services.feature_store_service import FeatureStoreService


def get_dataset(session: Session, level: Level) -> DataFrame:
    # get config
    config = Project.PLTV.config
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
    for feature_view_config in feature_view_configs:
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from projects.pltv.session import get_session

    session = get_session()
    dataset = get_dataset(session, Level(group_bys=["brand", "sku_type", "channel"]))
    df = dataset.to_pandas()
    df.to_parquet("output_dataset.parquet")