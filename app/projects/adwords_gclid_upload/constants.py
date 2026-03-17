# MARK: - Table Names
SOURCE_DATABASE_NAME = "BI_LAYER_DB"
SOURCE_SCHEMA_NAME = "PROD"
SOURCE_TABLE_NAME = "MT_ADWORDS_LAST_CLICK_CONVERSIONS"

RESULTS_TABLE_NAME = "RAW_RESULTS"
GCS_STAGE_NAME = "GCLID_UPLOAD_GCS_STAGE"
GCS_FILE_NAME = "gclid_last_click_conversions" 
# GCP link - https://console.cloud.google.com/storage/browser/ml-layer-adwords-gclid-upload;tab=objects?hl=en&project=tcg-bi&prefix=&forceOnObjectsSortingFiltering=false
# PUBLIC BUCKET URL gs://ml-layer-adwords-gclid-upload/gclid_last_click_conversions/

# MARK: - Column Names
BRAND_COL = "BRAND"
GOOGLE_CLICK_ID_COL = "GOOGLE_CLICK_ID"
CONVERSION_NAME_COL = "CONVERSION_NAME"
CONVERSION_TIME_COL = "CONVERSION_TIME"
CONVERSION_VALUE_COL = "CONVERSION_VALUE"

LAST_RUN_DATE_COL = "LAST_RUN_DATE"