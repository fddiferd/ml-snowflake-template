from enum import Enum

from projects.pltv.objects import Config, Level

# MARK: - Time Horizons
class TimeHorizon(Enum):
    DAYS_30 = "30"
    DAYS_60 = "60"
    DAYS_90 = "90"
    DAYS_180 = "180"
    DAYS_365 = "365"
    DAYS_730 = "730"

# MARK: - Column Utils
def get_gross_adds_created_over_days_ago_column(time_horizon: TimeHorizon) -> str:
    return f"gross_adds_created_over_{time_horizon.value}_days_ago".upper()

def get_net_billings_days_column(time_horizon: TimeHorizon) -> str:
    return f"net_billings_{time_horizon.value}_days".upper()

def get_avg_net_billings_column(time_horizon: TimeHorizon) -> str:
    return f"avg_net_billings_{time_horizon.value}_days".upper()

# MARK: - Config
config = Config(
    version_number=1,
    min_cohort_size=250,
    timestamp_col="start_date_month",
    partition_col="plan__is_promo",
    levels=[
        Level(
            group_bys=[
                'brand',
                'sku_type',
                'channel',
            ]
        )
    ],
    time_horizons=[t for t in TimeHorizon],
    cat_cols=[],
    num_cols=[
        'avg_promo_days',
        'avg_promo_price',
        'avg_recurring_days',
        'avg_recurring_price',
        'promo_to_recurring_days_ratio',
        'promo_to_recurring_price_ratio',
    ],
    boolean_cols=[],
    get_gross_adds_created_over_days_ago_column=get_gross_adds_created_over_days_ago_column,
    get_net_billings_days_column=get_net_billings_days_column,
    get_avg_net_billings_column=get_avg_net_billings_column,
)