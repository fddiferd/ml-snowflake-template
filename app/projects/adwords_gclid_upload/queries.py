UPLOAD_QUERY = """--sql
select
    {brand_col},
    {google_click_id_col},
    {conversion_name_col},
    {conversion_time_col},
    {conversion_value_col}
from {source_database_name}.{source_schema_name}.{source_table_name}
where {conversion_time_col} > '{from_date}'
--endsql"""

LAST_RUN_DATE_QUERY = """--sql
select
    max({conversion_time_col}) as {last_run_date_col}
from {source_database_name}.{source_schema_name}.{source_table_name}
--endsql"""