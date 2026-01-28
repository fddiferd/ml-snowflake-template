"""
PLTV Dataset Module
===================

Loads PLTV dataset using Snowflake Feature Store.
"""

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


logger = logging.getLogger(__name__)


def get_dataset(session: Session, level: Level) -> DataFrame:
    """Load dataset using Snowflake Feature Store.
    
    Creates spine, entity, and feature views, then generates a dataset
    by joining them together using the Feature Store service.
    
    Args:
        session: Snowflake Session
        level: Aggregation level (determines join keys and grouping)
    
    Returns:
        DataFrame: The joined dataset with all features
    """
    # Initialize feature store service
    svc = FeatureStoreService(
        session, 
        name=Project.PLTV.name, 
        timestamp_col=TIMESTAMP_COL
    )
    
    # Create spine
    spine_sql = SPINE_QUERY.format(
        timestamp_col=TIMESTAMP_COL,
        group_bys=level.sql_fields,
        partitions=partition_sql_fields,
    )
    logger.info(f"Spine SQL: {spine_sql}")
    spine_df = session.sql(spine_sql)
    svc.set_spine(spine_df)
    
    # Create entity
    svc.set_entity(
        join_keys=get_join_keys(level), 
        name=f'{level.name}_ENTITY',
        recreate=True
    )
    
    # Create feature views
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
    
    # Generate and return dataset
    return svc.get_dataset()
