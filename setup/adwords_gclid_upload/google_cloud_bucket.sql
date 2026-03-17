-- =============================================================================
-- ADWORDS GCLID UPLOAD Project - GCS Storage Integration & External Stage
-- =============================================================================
-- This file creates the GCS storage integration, external stage, and file
-- format for uploading GCLID conversion data to Google Cloud Storage.
-- Run via: Snowflake VS Code Extension (ACCOUNTADMIN)
-- =============================================================================

-- Configuration Variables
SET MY_ROLE_NAME = 'ML_LAYER_ROLE';
SET MY_WH_NAME = 'ML_LAYER_WH';
SET GCS_BUCKET_URL = 'gcs://ml-layer-adwords-gclid-upload/';

-- =============================================================================
-- 1. Storage Integration (requires ACCOUNTADMIN)
-- =============================================================================
USE ROLE ACCOUNTADMIN;

CREATE STORAGE INTEGRATION IF NOT EXISTS ADWORDS_GCLID_UPLOAD_GCS_INTEGRATION
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'GCS'
    ENABLED = TRUE
    STORAGE_ALLOWED_LOCATIONS = ('gcs://ml-layer-adwords-gclid-upload/');

-- Grant usage to ML_LAYER_ROLE so stored procedures can use the stage
GRANT USAGE ON INTEGRATION ADWORDS_GCLID_UPLOAD_GCS_INTEGRATION
    TO ROLE IDENTIFIER($MY_ROLE_NAME);

-- Retrieve the Snowflake service account for GCS bucket permissions
-- IMPORTANT: Copy the STORAGE_GCP_SERVICE_ACCOUNT value from the output.
-- You will need to grant this service account access to the GCS bucket.
DESC STORAGE INTEGRATION ADWORDS_GCLID_UPLOAD_GCS_INTEGRATION;

-- =============================================================================
-- 2. GCS Bucket Setup (manual steps in GCP Console)
-- =============================================================================
-- After running DESC INTEGRATION above, complete these steps:
--
-- a) Create the GCS bucket (if it doesn't exist):
--    gsutil mb -l US gs://ml-layer-adwords-gclid-upload/
--
-- b) Grant the Snowflake service account access to the bucket:
--    gsutil iam ch serviceAccount:<STORAGE_GCP_SERVICE_ACCOUNT>:objectAdmin \
--        gs://ml-layer-adwords-gclid-upload/
--
--    Or via GCP Console:
--    1. Go to Cloud Storage > Buckets > ml-layer-adwords-gclid-upload
--    2. Permissions > Grant Access
--    3. Principal: <STORAGE_GCP_SERVICE_ACCOUNT> from DESC INTEGRATION output
--    4. Role: Storage Object Admin
-- =============================================================================

-- =============================================================================
-- 3. External Stage & File Format
-- =============================================================================
USE ROLE IDENTIFIER($MY_ROLE_NAME);
USE DATABASE ML_LAYER_ADWORDS_GCLID_UPLOAD_DB;
USE SCHEMA PROD;

CREATE FILE FORMAT IF NOT EXISTS GCLID_UPLOAD_CSV_FORMAT
    TYPE = CSV
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '')
    COMPRESSION = NONE;

CREATE STAGE IF NOT EXISTS GCLID_UPLOAD_GCS_STAGE
    STORAGE_INTEGRATION = ADWORDS_GCLID_UPLOAD_GCS_INTEGRATION
    URL = 'gcs://ml-layer-adwords-gclid-upload/'
    FILE_FORMAT = GCLID_UPLOAD_CSV_FORMAT
    COMMENT = 'External GCS stage for AdWords GCLID conversion uploads';

-- =============================================================================
-- 4. Verify
-- =============================================================================
LIST @GCLID_UPLOAD_GCS_STAGE;

-- =============================================================================
-- Useful Commands
-- =============================================================================
-- List files:           LIST @GCLID_UPLOAD_GCS_STAGE;
-- Upload from table:    COPY INTO @GCLID_UPLOAD_GCS_STAGE FROM (SELECT ...) FILE_FORMAT = GCLID_UPLOAD_CSV_FORMAT;
-- Check integration:    DESC INTEGRATION ADWORDS_GCLID_UPLOAD_GCS_INTEGRATION;
-- Remove files:         REMOVE @GCLID_UPLOAD_GCS_STAGE/pattern;
