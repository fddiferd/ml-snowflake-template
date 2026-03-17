-- =============================================================================
-- ADWORDS GCLID UPLOAD Project - Stored Procedure and Task Setup
-- =============================================================================
-- This file creates the ADWORDS_GCLID_UPLOAD_RUN stored procedure and
-- ADWORDS_GCLID_UPLOAD_DAILY_TASK.
-- Run via: snow sql -f setup/adwords_gclid_upload/tasks.sql
-- =============================================================================

-- Configuration Variables
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_WH_NAME = 'ML_LAYER_WH';

-- Use the ML Layer role
USE ROLE IDENTIFIER($MY_ROLE_NAME);

-- Set database and schema
USE DATABASE ML_LAYER_ADWORDS_GCLID_UPLOAD_DB;
USE SCHEMA PROD;

-- =============================================================================
-- Stored Procedure
-- =============================================================================
CREATE OR REPLACE PROCEDURE ADWORDS_GCLID_UPLOAD_RUN()
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = (
        'snowflake-snowpark-python',
        'pandas',
        'numpy',
        'toml',
        'pydantic'
    )
    IMPORTS = ('@ML_LAYER_ADWORDS_GCLID_UPLOAD_DB.PROD.ML_LAYER_STAGE/adwords_gclid_upload/ml_layer.zip')
    HANDLER = 'projects.adwords_gclid_upload.sproc.run_sproc'
    EXECUTE AS OWNER;

-- =============================================================================
-- Scheduled Task (daily at 3pm PST)
-- =============================================================================
CREATE OR REPLACE TASK ADWORDS_GCLID_UPLOAD_DAILY_TASK
    WAREHOUSE = IDENTIFIER($MY_WH_NAME)
    SCHEDULE = 'USING CRON 0 15 * * * America/Los_Angeles'
    COMMENT = 'Daily GCLID conversion upload to GCS (3pm PST)'
AS
    CALL ADWORDS_GCLID_UPLOAD_RUN();

-- Resume the task (tasks are created in suspended state)
ALTER TASK ADWORDS_GCLID_UPLOAD_DAILY_TASK RESUME;

-- Verify creation
SHOW TASKS LIKE 'ADWORDS_GCLID_UPLOAD_DAILY_TASK';

-- =============================================================================
-- Useful Commands
-- =============================================================================
-- Suspend task:    ALTER TASK ADWORDS_GCLID_UPLOAD_DAILY_TASK SUSPEND;
-- Resume task:     ALTER TASK ADWORDS_GCLID_UPLOAD_DAILY_TASK RESUME;
-- Execute now:     EXECUTE TASK ADWORDS_GCLID_UPLOAD_DAILY_TASK;
-- Manual call:     CALL ADWORDS_GCLID_UPLOAD_RUN();
-- View history:
--   SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--       TASK_NAME => 'ADWORDS_GCLID_UPLOAD_DAILY_TASK',
--       SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP())
--   )) ORDER BY SCHEDULED_TIME DESC;
