SCHEMA_NAME = "PLTV"
VERSION_NUMBER: int = 1

TIMESTAMP_COL = 'date_month'
JOIN_KEYS: list[str] = [
    'date_month',
    'brand',
    'sku_type',
    'channel',
    'traffic_source',
    'campaign',
    'plan__offer_type', 
    'plan__promo_days',
    'plan__promo_price',
    'plan__recurring_days', 
    'plan__recurring_price'
]

"""
example of involuntary churn:
- sub_tree_id: 119385754
example of voluntary churn:
- sub_tree_id: 125568364
"""