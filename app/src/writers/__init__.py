from snowflake.snowpark import Session

from src.writers.protocol import Writer
from src.writers.types import WriterType
from src.writers.local import LocalWriter, ParquetWriter, CSVWriter
from src.writers.snowflake import SnowflakeWriter


def create_writer(
    writer_type: WriterType,
    folder_path: str | None = None,
    session: Session | None = None,
) -> Writer:
    """Factory function to create the appropriate writer.

    Args:
        writer_type: Type of writer to create (WriterType enum)
        folder_path: Required for local writers (parquet, csv). Path to the folder for output files.
        session: Required for snowflake writer. Snowflake session object.

    Returns:
        A Writer instance of the requested type.

    Raises:
        ValueError: If required parameters are missing for the requested writer type.
    """
    if writer_type == WriterType.PARQUET:
        if folder_path is None:
            raise ValueError("folder_path is required for ParquetWriter")
        return ParquetWriter(folder_path)

    elif writer_type == WriterType.CSV:
        if folder_path is None:
            raise ValueError("folder_path is required for CSVWriter")
        return CSVWriter(folder_path)

    elif writer_type == WriterType.SNOWFLAKE:
        if session is None:
            raise ValueError("session is required for SnowflakeWriter")
        return SnowflakeWriter(session)

    else:
        raise ValueError(f"Unknown writer type: {writer_type}")


__all__ = [
    "Writer",
    "WriterType",
    "LocalWriter",
    "ParquetWriter",
    "CSVWriter",
    "SnowflakeWriter",
    "create_writer",
]
