from pandas import DataFrame
from snowflake.snowpark import Session


class SnowflakeWriter:
    """Writer for Snowflake tables."""

    def __init__(self, session: Session):
        self.session = session

    @property
    def is_local(self) -> bool:
        return False

    @property
    def folder_path(self) -> None:
        return None

    def write(self, name: str, data: DataFrame, overwrite: bool = True) -> None:
        if len(data) == 0:
            return

        # Reset index and ensure column names are uppercase for Snowflake
        df = data.reset_index(drop=True)
        df.columns = df.columns.str.upper()

        self.session.write_pandas(
            df,
            name,
            auto_create_table=True,
            overwrite=overwrite,
        )
