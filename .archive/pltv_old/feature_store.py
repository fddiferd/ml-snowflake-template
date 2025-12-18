import logging

from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame

from src.services.feature_store_service import FeatureStoreService

from projects.pltv.queries import training_spine, training_metrics
from projects.pltv.constants import (
    SCHEMA_NAME, 
    VERSION_NUMBER, 
    JOIN_KEYS, 
    TIMESTAMP_COL,
)


logger = logging.getLogger(__name__)


def get_training_dataset(session: Session) -> DataFrame:
    # init feature store service
    svc = FeatureStoreService(session, f'{SCHEMA_NAME}_TRAINING', TIMESTAMP_COL)
    # spine
    spine_df = session.sql(training_spine)
    svc.set_spine(spine_df)
    # entity
    svc.set_entity(JOIN_KEYS)
    # feature views
    feature_df = session.sql(training_metrics)
    svc.set_feature_view(feature_df, VERSION_NUMBER)
    # get dataset
    return svc.get_dataset()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.connection import get_session
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    load_dotenv()
    
    session = get_session()
