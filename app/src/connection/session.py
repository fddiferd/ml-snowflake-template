import logging
import toml
from pathlib import Path
from cryptography.hazmat.primitives import serialization

from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

from projects import Project


logger = logging.getLogger(__name__)

def get_snowflake_active_session() -> Session | None:
    """Get the active Snowflake session if it exists"""
    try:
        active_connection: Session | None = get_active_session()
        if active_connection is not None:
            logger.info("Reinstating connections from active session")
            return active_connection
    except:
        pass

    return None

def get_session(project: Project) -> Session:
    """Create a Snowflake session using configuration from .snowflake/config.toml"""

    # If running in snowflake task it will use the active session
    active_session = get_snowflake_active_session()
    if active_session is not None:
        return active_session
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Load the Snowflake configuration
    config_path = project_root / ".snowflake" / "config.toml"
    config = toml.load(config_path)
    
    # Get the default connection settings
    conn_params = config["connections"]["default"]
    
    # Load and parse the private key
    private_key_path = project_root / conn_params["private_key_path"]
    with open(private_key_path, "rb") as key_file:
        private_key_data = key_file.read()
    
    # Parse the private key
    from cryptography.hazmat.backends import default_backend
    private_key = serialization.load_pem_private_key(
        private_key_data,
        password=None,
        backend=default_backend()
    )
    
    # Get the private key bytes in DER format
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Build the session with connection parameters
    session = Session.builder.configs({
        "account": conn_params["account"],
        "user": conn_params["user"],
        "role": conn_params.get("role"),
        "warehouse": conn_params.get("warehouse"),
        "database": conn_params.get("database"),
        "private_key": private_key_bytes, # type: ignore
        "schema": project.schema_name,
    }).create()

    logger.info(f"Connected to Snowflake for project {project.value}")
    logger.info(f"  Account: {session.get_current_account()}")
    logger.info(f"  User: {session.get_current_user()}")
    logger.info(f"  Role: {session.get_current_role()}")
    logger.info(f"  Database: {session.get_current_database()}")
    logger.info(f"  Warehouse: {session.get_current_warehouse()}")
    logger.info(f"  Schema: {project.schema_name}")

    return session