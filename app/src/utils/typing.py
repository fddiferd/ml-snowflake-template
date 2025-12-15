from pydantic import BaseModel
from enum import Enum
from typing import Union, List, Any
import logging

from src.constants import (
    BI_LAYER_DB,
    BI_LAYER_SCHEMA
)


logger = logging.getLogger(__name__)


class SqlOperator(Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL_TO = ">="
    LESS_THAN_OR_EQUAL_TO = "<="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"

class SqlWhere(BaseModel):
    key: str
    operator: SqlOperator
    value: Any

    def to_sql(self) -> str:
        is_value_string = isinstance(self.value, str)
        if is_value_string:
            return f"{self.key} {self.operator.value} '{self.value}'"
        else:
            return f"{self.key} {self.operator.value} {self.value}"

SqlWhereStatement = Union[SqlWhere, List[SqlWhere]]


def get_sql_str(table_name: str, where: SqlWhereStatement | None = None) -> str:
    where_clause = ""
    if where is not None:
        if isinstance(where, list):
            if where:  # Ensure list is not empty
                conditions = [w.to_sql() for w in where]
                where_clause = f" WHERE {' AND '.join(conditions)}"
        else:
            where_clause = f" WHERE {where.to_sql()}"

   
    sql: str = f"""
        select
        *
        from {BI_LAYER_DB}.{BI_LAYER_SCHEMA}.{table_name}
        {where_clause}
    """ 
    logger.debug(f"sql: {sql}")
    return sql


def get_version(version_number: int) -> str:
    return f'V{version_number}'