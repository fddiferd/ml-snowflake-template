from src.connection import get_session

from projects.pltv.constants import SCHEMA_NAME

session = get_session(
    schema_name=SCHEMA_NAME,
)