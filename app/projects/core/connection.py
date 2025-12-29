from snowflake.snowpark import Session

from projects import Project
from src.connection.session import get_session as get_snowflake_session

def get_session() -> Session:
    return get_snowflake_session(Project.CORE)