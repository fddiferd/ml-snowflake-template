import toml
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from snowflake.snowpark import Session

# Get the project root directory
project_root = Path(__file__).parent.parent

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
}).create()

print("\033[92m✔ Connected to Snowflake!\033[0m")
print(f"  Account: {session.get_current_account()}")
print(f"  User: {session.get_current_user()}")
print(f"  Role: {session.get_current_role()}")
print(f"  Database: {session.get_current_database()}")
print(f"  Warehouse: {session.get_current_warehouse()}")
