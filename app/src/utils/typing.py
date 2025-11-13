from src.constants import (
    BI_LAYER_DB,
    BI_LAYER_SCHEMA
)

def get_sql_str(table_name: str, sub_tree_id: str | None = None) -> str:
    return f"""
        select
        *
        from {BI_LAYER_DB}.{BI_LAYER_SCHEMA}.{table_name}
        """ + (f"where sub_tree_id= '{sub_tree_id}'" if sub_tree_id else "")

def get_version(version_number: int) -> str:
    return f'V{version_number}'