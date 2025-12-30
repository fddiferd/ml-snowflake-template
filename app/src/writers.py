from typing import Protocol
from pandas import DataFrame

class Writer(Protocol):
    def overwrite(self, data: DataFrame) -> None: ...
    def append(self, data: DataFrame) -> None: ...

