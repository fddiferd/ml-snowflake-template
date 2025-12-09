SCHEMA_NAME = "PLTV"
VERSION_NUMBER: int = 1

TIME_STAMP_COL: str = "DATE_DAY"
JOIN_KEYS: list[str] = [TIME_STAMP_COL, "SUB_TREE_ID"]
FEATURE_TABLE_NAMES = [
    f'{SCHEMA_NAME}_REBILL_ACTIVITIES',
    f'{SCHEMA_NAME}_BILLING_ACTIVITIES',
    f'{SCHEMA_NAME}_SUB_ACTIVITIES',
]

"""
example of involuntary churn:
- sub_tree_id: 119385754
example of voluntary churn:
- sub_tree_id: 125568364
"""