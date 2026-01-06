from enum import Enum

class WriterType(Enum):
    PARQUET = "PARQUET"
    CSV = "CSV"
    SNOWFLAKE = "SNOWFLAKE"

    @property
    def is_local(self) -> bool:
        return self == WriterType.PARQUET or self == WriterType.CSV