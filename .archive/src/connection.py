import logging
import os

from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

from src.utils.encryption import get_private_key
from src.constants import (
    ACCOUNT,
    DATABASE,
    WAREHOUSE,
    DEV_ROLE,
    DEV_USER,
    FEATURE_STORE_SCHEMA,
    ACCOUNT_ADMIN_ROLE
)


logger = logging.getLogger(__name__)


# MARK: Session
def get_session(
    database_name: str | None = None, 
    schema_name: str | None = None, 
    force_new: bool = False,
) -> Session:
    # if running in snowflake
    try:
        active_connection: Session | None = get_active_session()
        if active_connection is not None and not force_new:
            logger.info("Reinstating connections from active session")
            return active_connection
    except:
        pass

    # create session from private key
    logger.info("Creating session from private key")
    schema = schema_name or FEATURE_STORE_SCHEMA
    config_dict = {
        'account': ACCOUNT,
        'database': database_name or DATABASE,
        'schema': schema,
        'warehouse': WAREHOUSE,
        'role': DEV_ROLE,
        'user': DEV_USER,
        'private_key': get_private_key(),
    }

    session: Session = Session.builder.configs(config_dict).create()
    
    # Set database context explicitly
    database = database_name or DATABASE
    logging.info(f"Using database {database}")
    session.sql(f"USE DATABASE {database}").collect()
    
    logging.info(f"Creating schema {schema}")
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema.upper()}").collect()

    logging.info(f"Using schema {schema}")
    session.sql(f"USE SCHEMA {schema}").collect()
    return session

# MARK: Admin Session
def get_admin_session(
    external_browser_user: str | None = None,
) -> Session:
    logger.info("Getting admin session via external browser")
    external_browser_user = external_browser_user or os.getenv('EXTERNAL_BROWSER_USER')
    if external_browser_user is None:
        raise ValueError("EXTERNAL_BROWSER_USER is not provided or set in env")

    config_dict: dict[str, int | str] = {
        'account': ACCOUNT,
        'database': DATABASE,
        'schema': FEATURE_STORE_SCHEMA,
        'warehouse': WAREHOUSE,
        'role': ACCOUNT_ADMIN_ROLE,
        'authenticator': 'externalbrowser',
        'user': external_browser_user,
    }
    session: Session = Session.builder.configs(config_dict).create()
    logger.info("Session created from external browser")
    return session

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    session = get_session()
    # admin_session = get_admin_session()