-- Using Snowflake VS Code Extension login to account vx70256-sw09563 - Signle Sign On
SET MY_USER_NAME = 'ML_LAYER_USER';
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_DB_NAME = 'ML_LAYER_DB';
SET MY_WH_NAME = 'ML_LAYER_WH';
SET ADMIN_ROLE = 'ACCOUNTADMIN';
SET MY_NETWORK_POLICY_NAME = 'ML_LAYER_CICD_POLICY';

-- 1. Context Setup
USE ROLE IDENTIFIER($ADMIN_ROLE);

-- 2. Create Role
CREATE OR REPLACE ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT ROLE IDENTIFIER($MY_ROLE_NAME) TO ROLE SYSADMIN;

-- 2b. Grant Account-Level Privileges
GRANT EXECUTE TASK ON ACCOUNT TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT MONITOR EXECUTION ON ACCOUNT TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT CREATE INTEGRATION ON ACCOUNT TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT CREATE DATABASE ON ACCOUNT TO ROLE IDENTIFIER($MY_ROLE_NAME);

-- 3. Create Warehouse
CREATE OR REPLACE WAREHOUSE IDENTIFIER($MY_WH_NAME)
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ML Snowpark workloads';

GRANT USAGE ON WAREHOUSE IDENTIFIER($MY_WH_NAME) TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT OPERATE ON WAREHOUSE IDENTIFIER($MY_WH_NAME) TO ROLE IDENTIFIER($MY_ROLE_NAME);

-- 4. Create Database
CREATE OR REPLACE DATABASE IDENTIFIER($MY_DB_NAME);
GRANT OWNERSHIP ON DATABASE IDENTIFIER($MY_DB_NAME) TO ROLE IDENTIFIER($MY_ROLE_NAME) COPY CURRENT GRANTS;

-- 5. Create User
CREATE OR REPLACE USER IDENTIFIER($MY_USER_NAME)
    PASSWORD = 'ChangeMe123!'
    DEFAULT_ROLE = $MY_ROLE_NAME
    DEFAULT_WAREHOUSE = $MY_WH_NAME
    MUST_CHANGE_PASSWORD = FALSE
    COMMENT = 'Service user for ML operations';

-- 6. Grant Role to User
GRANT ROLE IDENTIFIER($MY_ROLE_NAME) TO USER IDENTIFIER($MY_USER_NAME);

-- 7. Network Policy for CI/CD
-- Allow all IPs for the service user since it uses key-pair auth (no password)
-- The private key is the security control, not IP restrictions
CREATE OR REPLACE NETWORK POLICY ML_LAYER_CICD_POLICY
    ALLOWED_IP_LIST = ('0.0.0.0/0')
    COMMENT = 'Permissive policy for CI/CD service account with key-pair auth';

ALTER USER IDENTIFIER($MY_USER_NAME) UNSET NETWORK_POLICY;
ALTER USER IDENTIFIER($MY_USER_NAME) SET NETWORK_POLICY = ML_LAYER_CICD_POLICY;

-- Final check
SELECT 
    $MY_DB_NAME as database, 
    $MY_WH_NAME as warehouse, 
    $MY_ROLE_NAME as role, 
    $MY_USER_NAME as user,
    'Setup Complete - NOW RUN setup/setup_key_pair.sql' as status;

-- 8. Create Deployment Stage for Snowpark procedures
USE ROLE IDENTIFIER($MY_ROLE_NAME);
USE DATABASE $MY_DB_NAME;

-- Create stage for deploying stored procedures (used by Snowflake CLI)
CREATE STAGE IF NOT EXISTS PUBLIC.ML_LAYER_STAGE
    COMMENT = 'Stage for ML Layer Snowpark deployment artifacts';

-- Also create for PLTV database
CREATE DATABASE IF NOT EXISTS ML_LAYER_PLTV_DB;
CREATE SCHEMA IF NOT EXISTS ML_LAYER_PLTV_DB.PROD;
CREATE STAGE IF NOT EXISTS ML_LAYER_PLTV_DB.PROD.ML_LAYER_STAGE
    COMMENT = 'Stage for PLTV Snowpark deployment artifacts';

-- 9. Create Slack Notification Integration
USE DATABASE $MY_DB_NAME;
USE SCHEMA PUBLIC;

-- grant role usage to schema
GRANT USAGE ON SCHEMA PUBLIC TO ROLE IDENTIFIER($MY_ROLE_NAME);
GRANT CREATE SECRET ON SCHEMA PUBLIC TO ROLE IDENTIFIER($MY_ROLE_NAME);

-- Create Slack Notification Secret
CREATE OR REPLACE SECRET ml_layer_slack_webhook_secret
  TYPE = GENERIC_STRING
  SECRET_STRING = 'T02GL1CBD/B09FSUGR7NK/QNUa56H4Pptyxn46OR9gIFl5';

-- Grant READ on secret to role
GRANT READ ON SECRET ml_layer_slack_webhook_secret TO ROLE IDENTIFIER($MY_ROLE_NAME);

CREATE OR REPLACE NOTIFICATION INTEGRATION ml_layer_notifications
  TYPE = WEBHOOK
  ENABLED = TRUE
  WEBHOOK_URL = 'https://hooks.slack.com/services/SNOWFLAKE_WEBHOOK_SECRET'
  WEBHOOK_SECRET = ml_layer_slack_webhook_secret
  WEBHOOK_BODY_TEMPLATE = '{
    "text": ":rotating_light: *Task Alert*",
    "blocks": [
      {
        "type": "header",
        "text": {
          "type": "plain_text",
          "text": "Task Notification",
          "emoji": true
        }
      },
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "SNOWFLAKE_WEBHOOK_MESSAGE"
        }
      },
      {
        "type": "context",
        "elements": [
          {
            "type": "mrkdwn",
            "text": "Sent from ML Layer"
          }
        ]
      }
    ]
  }'
  WEBHOOK_HEADERS = ('Content-Type'='application/json');

-- Grant USAGE on notification integration to role
GRANT USAGE ON INTEGRATION ml_layer_notifications TO ROLE IDENTIFIER($MY_ROLE_NAME);

CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
  SNOWFLAKE.NOTIFICATION.TEXT_PLAIN(
    SNOWFLAKE.NOTIFICATION.SANITIZE_WEBHOOK_CONTENT(
      'Test notification: Snowflake-Slack integration is working! 🎉'
    )
  ),
  SNOWFLAKE.NOTIFICATION.INTEGRATION('ml_layer_notifications')
);