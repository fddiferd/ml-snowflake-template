import logging

from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from src.services.feature_store_service import FeatureStoreService
from src.utils.typing import get_sql_str, SqlWhere, SqlOperator


logger = logging.getLogger(__name__)


from projects.pltv.constants import (
    SCHEMA_NAME, 
    VERSION_NUMBER, 
    TIME_STAMP_COL,
    JOIN_KEYS, 
    FEATURE_TABLE_NAMES
)

def get_dataset(
    session: Session, 
    start_date: str, 
    end_date: str, 
    sub_tree_id: str | None = None # for testing
) -> DataFrame:
    # init feature store service
    svc = FeatureStoreService(session, SCHEMA_NAME, TIME_STAMP_COL)

    where = [
        SqlWhere(key=TIME_STAMP_COL, operator=SqlOperator.GREATER_THAN_OR_EQUAL_TO, value=start_date),
        SqlWhere(key=TIME_STAMP_COL, operator=SqlOperator.LESS_THAN_OR_EQUAL_TO, value=end_date),
    ]
    if sub_tree_id is not None:
        where.append(SqlWhere(key='SUB_TREE_ID', operator=SqlOperator.EQUALS, value=sub_tree_id))

    # spine
    spine_df = session.sql(get_sql_str(f'{SCHEMA_NAME}_TIME_SPINE', where))
    svc.set_spine(spine_df)
    # entity
    svc.set_entity(JOIN_KEYS)
    # feature views
    for table_name in FEATURE_TABLE_NAMES:
        feature_df = session.sql(
            get_sql_str(table_name, where)
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
    df = get_dataset(session, start_date='2025-01-01', end_date='2025-01-31')
    print(df.to_pandas().head())