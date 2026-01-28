-- =============================================================================
-- VBB Project - Stored Procedure and Task Setup
-- =============================================================================
-- This file creates the VBB_RUN stored procedure and VBB_WEEKLY_TASK.
-- Run via: snow sql -f setup/vbb/tasks.sql
-- =============================================================================

-- Configuration Variables (Single Source of Truth)
-- These should match the values in app/src/shared_config.py
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_WH_NAME = 'ML_LAYER_WH';

-- Use the ML Layer role
USE ROLE IDENTIFIER($MY_ROLE_NAME);

-- Set database and schema
USE DATABASE ML_LAYER_VBB_DB;
USE SCHEMA PROD;

-- =============================================================================
-- Stored Procedure
-- =============================================================================
CREATE OR REPLACE PROCEDURE VBB_RUN()
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
        'toml',
        'pydantic'
    )
    IMPORTS = ('@ML_LAYER_VBB_DB.PROD.ML_LAYER_STAGE/vbb/ml_layer.zip')
    HANDLER = 'projects.vbb.sproc.run_sproc'
    EXECUTE AS OWNER;

-- =============================================================================
-- Scheduled Task
-- =============================================================================
CREATE OR REPLACE TASK VBB_WEEKLY_TASK
    WAREHOUSE = IDENTIFIER($MY_WH_NAME)
    SCHEDULE = 'USING CRON 0 6 * * 1 America/Los_Angeles'  -- Monday 6am PT
    COMMENT = 'Weekly VBB model training and prediction run'
AS
    CALL VBB_RUN();

-- Enable error notifications (uncomment when notification integration is created)
-- ALTER TASK VBB_WEEKLY_TASK SET
--     ERROR_INTEGRATION = 'ML_LAYER_NOTIFICATIONS';

-- Resume the task (tasks are created in suspended state)
ALTER TASK VBB_WEEKLY_TASK RESUME;

-- Verify creation
SHOW TASKS LIKE 'VBB_WEEKLY_TASK';

-- =============================================================================
-- Useful Commands
-- =============================================================================
-- Suspend task:    ALTER TASK VBB_WEEKLY_TASK SUSPEND;
-- Resume task:     ALTER TASK VBB_WEEKLY_TASK RESUME;
-- Execute now:     EXECUTE TASK VBB_WEEKLY_TASK;
-- Manual call:     CALL VBB_RUN();
-- View history:
--   SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--       TASK_NAME => 'VBB_WEEKLY_TASK',
--       SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP())
--   )) ORDER BY SCHEDULED_TIME DESC;
