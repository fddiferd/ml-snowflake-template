if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import logging
    logging.basicConfig(level=logging.INFO)


from datetime import datetime, timedelta
from snowflake.snowpark.functions import col
from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.exceptions import SnowparkSQLException
import logging


from projects.adwords_gclid_upload.constants import (
    SOURCE_DATABASE_NAME,
    SOURCE_SCHEMA_NAME,
    SOURCE_TABLE_NAME,
    RESULTS_TABLE_NAME,
    GCS_STAGE_NAME,
    GCS_FILE_NAME,
    BRAND_COL,
    GOOGLE_CLICK_ID_COL,
    CONVERSION_TIME_COL,
    CONVERSION_NAME_COL,
    CONVERSION_VALUE_COL,
    LAST_RUN_DATE_COL,
)
from projects.adwords_gclid_upload.queries import UPLOAD_QUERY, LAST_RUN_DATE_QUERY


logger = logging.getLogger(__name__)


def _get_last_run_date(session: Session) -> str:
    """Get the last run date for the Adwords GCLID Upload project."""
    try:
        formatted_query = LAST_RUN_DATE_QUERY.format(
            source_database_name=session.get_current_database(),
            source_schema_name=session.get_current_schema(),
            source_table_name=RESULTS_TABLE_NAME,
            conversion_time_col=CONVERSION_TIME_COL,
            last_run_date_col=LAST_RUN_DATE_COL,
        )
        logger.info(f"Formatted query: {formatted_query}")
        value = session.sql(formatted_query).to_pandas()[LAST_RUN_DATE_COL].iloc[0]
        logger.info(f"Last run date: {value}")
        return value
    except SnowparkSQLException as e:
        if "does not exist or not authorized" in str(e):
            fallback = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            logger.warning(f"Results table not found, falling back to {fallback}")
            return fallback
        raise e


def _get_dataset(session: Session, last_run_date: str) -> DataFrame:
    """Get the dataset for the Adwords GCLID Upload project."""
    formatted_query = UPLOAD_QUERY.format(
        brand_col=BRAND_COL,
        source_database_name=SOURCE_DATABASE_NAME,
        source_schema_name=SOURCE_SCHEMA_NAME,
        source_table_name=SOURCE_TABLE_NAME,
        from_date=last_run_date,
        google_click_id_col=GOOGLE_CLICK_ID_COL,
        conversion_name_col=CONVERSION_NAME_COL,
        conversion_time_col=CONVERSION_TIME_COL,
        conversion_value_col=CONVERSION_VALUE_COL,
    )
    logger.info(f"Formatted query: {formatted_query}")
    return session.sql(formatted_query)


def _save_to_table(df: DataFrame) -> None:
    """Append the results to the results table."""
    df.write.save_as_table(
        RESULTS_TABLE_NAME,
        mode="append",
    )

def _overwrite_stage_file(df: DataFrame, brand: str) -> None:
    """Write the GCLID conversion data as CSV to the GCS stage."""
    file_name = f"@{GCS_STAGE_NAME}/{GCS_FILE_NAME}_{brand}.csv"
    logger.info(f"Writing CSV to GCS stage: {file_name}")
    df.write.copy_into_location(
        file_name,
        header=True,
        overwrite=True,  # type: ignore[arg-type]
        single=True,  # type: ignore[arg-type]
        file_format_type="CSV",
        format_type_options={
            "COMPRESSION": "NONE",
            "FIELD_OPTIONALLY_ENCLOSED_BY": '"',
        },
    )
    logger.info(f"Uploaded {file_name} to stage {GCS_STAGE_NAME}")


def _get_unique_brands(df: DataFrame) -> list[str]:
    """Get the unique brands from the dataset."""
    rows = df.select(BRAND_COL).distinct().collect()
    return [str(row[BRAND_COL]) for row in rows]


def _export_brand_files_to_gcs(df: DataFrame) -> None:
    """Export the brand files to GCS."""
    unique_brands = _get_unique_brands(df)
    for brand in unique_brands:
        logger.info(f"Exporting brand {brand} to GCS")
        filtered_df = df.filter(col(BRAND_COL) == brand)
        _overwrite_stage_file(filtered_df, brand)


class GclidUploadService:
    def __init__(self, session: Session):
        self.session = session


    def run(self) -> None:
        last_run_date = _get_last_run_date(self.session)
        logger.info(f"Last run date: {last_run_date}")
        df = _get_dataset(self.session, last_run_date)
        logger.info(f"Loaded {df.count()} rows")
        _save_to_table(df)
        _export_brand_files_to_gcs(df)


if __name__ == "__main__":
    from projects.adwords_gclid_upload import get_session
    session = get_session()
    service = GclidUploadService(session)
    service.run()