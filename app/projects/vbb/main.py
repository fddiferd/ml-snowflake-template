"""
VBB Main Entry Point
====================

Main entry point for the VBB model pipeline.
Can be run locally or as a Snowflake stored procedure.

Usage:
    # Local development
    python -m projects.vbb.main
    
    # Or import and call
    from projects.vbb.main import main
    main(writer_type=WriterType.CSV)
"""

import logging
from snowflake.snowpark import Session

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

from src.writers import Writer, WriterType, create_writer
from projects.vbb import get_session
from projects.vbb.config import CACHE_PATH
from src.utils.slack import send_slack_notification
import time


logger = logging.getLogger(__name__)


def run_pipeline(session: Session, writer: Writer):
    """Run the VBB model pipeline.
    
    Args:
        session: The Snowflake session
        writer: The writer to use for saving results
    
    TODO: Implement your model pipeline here
    """
    from projects.vbb import DatasetLoader
    
    # Load dataset
    loader = DatasetLoader(session)
    df = loader.load()
    
    logger.info(f"Loaded dataset with {len(df)} rows")
    
    # TODO: Add your model training/prediction logic here
    # Example:
    # model_service = ModelService(df=df, writer=writer)
    # model_service.run()
    
    # For now, just save the raw data as a placeholder
    writer.write("VBB_RAW_DATA", df)
    
    logger.info("VBB pipeline completed")


def main(
    session: Session | None = None,
    writer_type: WriterType = WriterType.CSV,
    reset_schema: bool = False
):
    """Run the VBB model.
    
    Args:
        session: Snowflake session. If None, creates one via get_session().
                 Pass session directly when running as stored procedure.
        writer_type: Type of writer to use:
            - WriterType.CSV: Local CSV files (default, for development)
            - WriterType.PARQUET: Local parquet files
            - WriterType.SNOWFLAKE: Write directly to Snowflake tables
        reset_schema: If True and writer_type is SNOWFLAKE, drops and recreates schema.
    """
    # Use provided session or create one (for local development)
    if session is None:
        session = get_session()
    
    start_time = time.time()
    
    # Reset schema if requested
    if reset_schema and writer_type == WriterType.SNOWFLAKE:
        from src.environment import environment
        session.sql(f"DROP SCHEMA IF EXISTS {environment.schema_name} CASCADE").collect()
        session.sql(f"CREATE SCHEMA {environment.schema_name}").collect()
    
    # Create writer
    writer = create_writer(
        writer_type,
        session=session,
        folder_path=CACHE_PATH
    )
    
    # Run pipeline
    run_pipeline(session, writer)
    
    # Send success notification (failure notifications handled by sproc)
    try:
        send_slack_notification(
            session=session,
            header="VBB Model Run",
            text=f"VBB model run completed successfully in {time.time() - start_time:.1f} seconds",
            is_success=True
        )
    except Exception:
        logger.warning("Failed to send success notification to Slack")


if __name__ == "__main__":
    main(writer_type=WriterType.CSV)
