# Snowflake
ACCOUNT = 'sw09563.us-east-1'
DATABASE = 'ML_LAYER'
WAREHOUSE = 'SNOWPARK_WH'
ACCOUNT_ADMIN_ROLE = 'ACCOUNTADMIN'
DEV_ROLE = 'SNOWPARK_DEV'
DEV_USER = 'SNOWPARK_USER'
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