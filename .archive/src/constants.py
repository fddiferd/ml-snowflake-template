# Snowflake
ACCOUNT = 'vx70256-sw09563'
DATABASE = 'ML_LAYER_DB'
WAREHOUSE = 'ML_LAYER_WH'
ACCOUNT_ADMIN_ROLE = 'ACCOUNTADMIN'
DEV_ROLE = 'ML_LAYER_ROLE'
DEV_USER = 'ML_LAYER_USER'
FEATURE_STORE_SCHEMA = 'FEATURE_STORE'
FEATURE_STORE_VERSION_NUMBER = 1

BI_LAYER_DB = 'BI_LAYER_DB'
BI_LAYER_SCHEMA = 'STAGING'

SNOWFLAKE_DEPENDENCIES = [
    "snowflake-snowpark-python",
    "pandas>=2.2,<3",
    "scikit-learn>=1.5,<1.6",
    "numpy>=1.23,<2.0",
    "xgboost>=2.0,<2.1",
    "joblib",
    "python-dotenv",
    "pydantic>=2.0,<3.0",
    "snowflake-ml-python>=1.9,<2"
]

# Repository
PACKAGE_NAME = 'src'
PRIVATE_KEY_PEM_PATH = '.keys/rsa_key.p8'

# Slack
NOTIFICATION_INTEGRATION_NAME = f'ml_layer_notifications'