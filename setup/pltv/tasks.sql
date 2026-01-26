-- =============================================================================
-- PLTV Project - Stored Procedure and Task Setup
-- =============================================================================
-- This file creates the PLTV_RUN stored procedure and PLTV_WEEKLY_TASK.
-- Run via: snow sql -f setup/pltv/tasks.sql
-- =============================================================================

-- Configuration Variables
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_WH_NAME = 'ML_LAYER_WH';

-- Use the ML Layer role
USE ROLE IDENTIFIER($MY_ROLE_NAME);

-- Set database and schema
USE DATABASE ML_LAYER_PLTV_DB;
USE SCHEMA PROD;

-- =============================================================================
-- Stored Procedure
-- =============================================================================
CREATE OR REPLACE PROCEDURE PLTV_RUN()
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = (
        'snowflake-snowpark-python',
        'snowflake-ml-python',
        'pandas',
        'scikit-learn',
        'numpy',
        'xgboost',
        'shap',
        'joblib',
        'toml'
    )
    IMPORTS = ('@ML_LAYER_PLTV_DB.PROD.ML_LAYER_STAGE/app/app.zip')
    HANDLER = 'projects.pltv.sproc.run_sproc'
    EXECUTE AS OWNER;

-- =============================================================================
-- Scheduled Task
-- =============================================================================
CREATE OR REPLACE TASK PLTV_WEEKLY_TASK
    WAREHOUSE = IDENTIFIER($MY_WH_NAME)
    SCHEDULE = 'USING CRON 0 6 * * 1 America/Los_Angeles'  -- Monday 6am PT
    COMMENT = 'Weekly PLTV model training and prediction run'
AS
    CALL PLTV_RUN();

-- Enable error notifications (uncomment when notification integration is created)
-- ALTER TASK PLTV_WEEKLY_TASK SET
--     ERROR_INTEGRATION = 'ML_LAYER_NOTIFICATIONS';

-- Resume the task (tasks are created in suspended state)
ALTER TASK PLTV_WEEKLY_TASK RESUME;

-- Verify creation
SHOW TASKS LIKE 'PLTV_WEEKLY_TASK';

-- =============================================================================
-- Useful Commands
-- =============================================================================
-- Suspend task:    ALTER TASK PLTV_WEEKLY_TASK SUSPEND;
-- Resume task:     ALTER TASK PLTV_WEEKLY_TASK RESUME;
-- Execute now:     EXECUTE TASK PLTV_WEEKLY_TASK;
-- View history:
--   SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--       TASK_NAME => 'PLTV_WEEKLY_TASK',
--       SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP())
--   )) ORDER BY SCHEDULED_TIME DESC;
