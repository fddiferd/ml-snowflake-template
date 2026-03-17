-- =============================================================================
-- VBB Project - Stored Procedures and Task Setup
-- =============================================================================
-- Creates two stored procedures (train + predict) and two scheduled tasks.
-- Run via: snow sql -f setup/vbb/tasks.sql
-- =============================================================================

-- Configuration Variables
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_WH_NAME = 'ML_LAYER_WH';

USE ROLE IDENTIFIER($MY_ROLE_NAME);
USE DATABASE ML_LAYER_VBB_DB;
USE SCHEMA PROD;

-- =============================================================================
-- Stored Procedures
-- =============================================================================

-- Weekly training procedure (full diagnostics)
CREATE OR REPLACE PROCEDURE VBB_TRAIN()
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
        'scipy',
        'joblib',
        'toml',
        'pydantic'
    )
    IMPORTS = ('@ML_LAYER_VBB_DB.PROD.ML_LAYER_STAGE/vbb/ml_layer.zip')
    HANDLER = 'projects.vbb.sproc.run_train_sproc'
    EXECUTE AS OWNER;

-- Daily prediction procedure (train + predict + write to table/GCS)
CREATE OR REPLACE PROCEDURE VBB_PREDICT()
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
        'scipy',
        'joblib',
        'toml',
        'pydantic'
    )
    IMPORTS = ('@ML_LAYER_VBB_DB.PROD.ML_LAYER_STAGE/vbb/ml_layer.zip')
    HANDLER = 'projects.vbb.sproc.run_predict_sproc'
    EXECUTE AS OWNER;

-- =============================================================================
-- Scheduled Tasks
-- =============================================================================

-- Weekly training: Sunday 3am PT
CREATE OR REPLACE TASK VBB_WEEKLY_TRAIN_TASK
    WAREHOUSE = IDENTIFIER($MY_WH_NAME)
    SCHEDULE = 'USING CRON 0 3 * * 0 America/Los_Angeles'
    COMMENT = 'Weekly VBB model training with diagnostics'
AS
    CALL VBB_TRAIN();

-- Daily prediction: 3pm PT
CREATE OR REPLACE TASK VBB_DAILY_PREDICT_TASK
    WAREHOUSE = IDENTIFIER($MY_WH_NAME)
    SCHEDULE = 'USING CRON 0 15 * * * America/Los_Angeles'
    COMMENT = 'Daily VBB prediction export to table and GCS'
AS
    CALL VBB_PREDICT();

-- Resume tasks (created in suspended state)
ALTER TASK VBB_WEEKLY_TRAIN_TASK RESUME;
ALTER TASK VBB_DAILY_PREDICT_TASK RESUME;

-- Verify
SHOW TASKS LIKE 'VBB_%';

-- =============================================================================
-- Useful Commands
-- =============================================================================
-- Suspend tasks:   ALTER TASK VBB_WEEKLY_TRAIN_TASK SUSPEND;
--                  ALTER TASK VBB_DAILY_PREDICT_TASK SUSPEND;
-- Resume tasks:    ALTER TASK VBB_WEEKLY_TRAIN_TASK RESUME;
--                  ALTER TASK VBB_DAILY_PREDICT_TASK RESUME;
-- Execute now:     EXECUTE TASK VBB_WEEKLY_TRAIN_TASK;
--                  EXECUTE TASK VBB_DAILY_PREDICT_TASK;
-- Manual call:     CALL VBB_TRAIN();
--                  CALL VBB_PREDICT();
-- View history:
--   SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--       TASK_NAME => 'VBB_DAILY_PREDICT_TASK',
--       SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP())
--   )) ORDER BY SCHEDULED_TIME DESC;
