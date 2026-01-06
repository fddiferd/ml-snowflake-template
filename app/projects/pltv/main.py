import logging
from snowflake.snowpark import Session

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

from src.writers import Writer, WriterType, create_writer
from projects.pltv import (
    Level,
    get_session,
    clean_df,
    DatasetLoader,
    ModelService,
)
from projects.pltv.data.utils import CACHE_PATH


logger = logging.getLogger(__name__)


def main_level(session: Session, writer: Writer, level: Level):
    """Run the model for a single level.
    
    Args:
        session: The Snowflake session
        writer: The writer to use for saving results
        level: The level of granularity to run the model at
    """
    
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


def main(writer_type: WriterType = WriterType.CSV):
    """Run the model for all levels.
    
    Args:
        writer_type: Type of writer to use:
            - WriterType.PARQUET: Local parquet files (default, for development)
            - WriterType.CSV: Local CSV files
            - WriterType.SNOWFLAKE: Write directly to Snowflake tables
    """
    session = get_session()

    writer = create_writer(
        writer_type, 
        session=session,
        folder_path=CACHE_PATH
    )
    
    # Run for each level
    for level in Level:
        main_level(session, writer, level)


if __name__ == "__main__":
    main(writer_type=WriterType.SNOWFLAKE)
