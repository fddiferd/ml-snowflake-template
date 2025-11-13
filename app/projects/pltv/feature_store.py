import logging
import pandas as pd

from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from src.services.feature_store_service import FeatureStoreService
from src.utils.typing import get_sql_str

from projects.pltv.constants import (
    SCHEMA_NAME, 
    VERSION_NUMBER, 
    TIME_STAMP_COL,
    JOIN_KEYS, 
    SUB_TREE_ID,
    FEATURE_TABLE_NAMES
)

def get_dataset_df(session: Session) -> DataFrame:
    # init feature store service
    svc = FeatureStoreService(session, SCHEMA_NAME, TIME_STAMP_COL)
    # spine
    spine_df = session.sql(get_sql_str(f'{SCHEMA_NAME}_TIME_SPINE', SUB_TREE_ID))
    svc.set_spine(spine_df)
    # entity
    svc.set_entity(JOIN_KEYS)
    # feature views
    for table_name in FEATURE_TABLE_NAMES:
        feature_df = session.sql(
            get_sql_str(table_name, SUB_TREE_ID)
        )
        svc.set_feature_view(feature_df, VERSION_NUMBER, f'{table_name}_FV')
    # get dataset
    return svc.get_dataset()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.connection import get_session
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    load_dotenv()
    
    session = get_session()
    df = get_dataset_df(session)
    print(df.to_pandas().head())