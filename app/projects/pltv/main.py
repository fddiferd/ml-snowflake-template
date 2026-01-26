import logging
from snowflake.snowpark import Session

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

from src.writers import Writer, WriterType, create_writer
from projects.pltv import Level, get_session
from projects.pltv.data.utils import CACHE_PATH
from src.utils.slack import send_slack_notification
import time


logger = logging.getLogger(__name__)


def _reset_schema(session: Session) -> None:
    """Reset the schema for the model."""
    from src.environment import environment
    session.sql(f"DROP SCHEMA IF EXISTS {environment.schema_name} CASCADE").collect()
    session.sql(f"CREATE SCHEMA {environment.schema_name}").collect()


def main_level(session: Session, writer: Writer, level: Level):
    """Run the model for a single level.
    
    Args:
        session: The Snowflake session
        writer: The writer to use for saving results
        level: The level of granularity to run the model at
    """
    # Lazy imports to avoid loading FeatureStore at module level
    # (FeatureStore import can cause SQL issues in stored procedures)
    from projects.pltv import DatasetLoader, ModelService, clean_df
    
    # load dataset
    loader = DatasetLoader(session)
    df = loader.load(level)
    
    # clean dataset
    clean_df(df)
    
    # run model service
    model_service = ModelService(
        level=level,
        df=df,
        writer=writer,
        test_train_split=True,
    )
    model_service.run()


def main(
    session: Session | None = None,
    writer_type: WriterType = WriterType.CSV, 
    reset_schema: bool = False
):
    """Run the model for all levels.
    
    Args:
        session: Snowflake session. If None, creates one via get_session().
                 Pass session directly when running as stored procedure.
        writer_type: Type of writer to use:
            - WriterType.PARQUET: Local parquet files (default, for development)
            - WriterType.CSV: Local CSV files
            - WriterType.SNOWFLAKE: Write directly to Snowflake tables
        reset_schema: If True and writer_type is SNOWFLAKE, drops and recreates schema.
    """
    # Use provided session or create one (for local development)
    if session is None:
        session = get_session()
    
    start_time = time.time()

    if reset_schema and writer_type == WriterType.SNOWFLAKE:
        _reset_schema(session)

    writer = create_writer(
        writer_type, 
        session=session,
        folder_path=CACHE_PATH
    )
    
    # Run for CAMPAIGN level
    try:
        main_level(session, writer, Level.CAMPAIGN)
    except Exception as e:
        send_slack_notification(
            session=session,  # Use same session for notifications
            header="PLTV Model Run", 
            text=f"PLTV model run failed in {time.time() - start_time:.1f} seconds: {e}", 
            is_success=False
        )
        raise e

    send_slack_notification(
        session=session,  # Use same session for notifications
        header="PLTV Model Run", 
        text=f"PLTV model run completed successfully in {time.time() - start_time:.1f} seconds", 
        is_success=True
    )


if __name__ == "__main__":
    main(writer_type=WriterType.SNOWFLAKE, reset_schema=False)
