from typing import Protocol
from pandas import DataFrame
import os
from snowflake.snowpark import Session

class Writer(Protocol):
    def overwrite(self, file_name: str, data: DataFrame) -> None: ...
    def append(self, file_name: str, data: DataFrame) -> None: ...


class CSVWriter:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def _get_file_path(self, file_name: str) -> str:
        return os.path.join(self.folder_path, file_name + '.csv')

    def overwrite(self, file_name: str, data: DataFrame) -> None:
        file_path = self._get_file_path(file_name)
        data.to_csv(file_path)

    def append(self, file_name: str, data: DataFrame) -> None:
        file_path = self._get_file_path(file_name)
        data.to_csv(file_path, mode='a', header=False)

class ParquetWriter:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def _get_file_path(self, file_name: str) -> str:
        return os.path.join(self.folder_path, file_name + '.parquet')

    def overwrite(self, file_name: str, data: DataFrame) -> None:
        file_path = self._get_file_path(file_name)
        data.to_parquet(file_path)

    def append(self, file_name: str, data: DataFrame) -> None:
        file_path = self._get_file_path(file_name)
        data.to_parquet(file_path, mode='a', header=False)

class SnowflakeWriter:
    def __init__(self, session: Session):
        self.session = session

    def overwrite(self, file_name: str, data: DataFrame) -> None:
        self.session.write_pandas(data, file_name, overwrite=True)

    def append(self, file_name: str, data: DataFrame) -> None:
        self.session.write_pandas(data, file_name, mode={'append': True}, overwrite=False)