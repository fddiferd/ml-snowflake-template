from projects.pltv.core.enums import (
    TimeHorizon, 
    partitions,
    time_horizons,
    model_steps,
)
from projects.pltv.core.base_models import (
    Config, 
    Level, 
    FeatureViewConfig,
    FeatureViewConfigs
)
from projects.pltv.data.queries.feature_views import RETENTION_METRICS_QUERY, BILLING_METRICS_QUERY

# MARK: - FV Configs
fv_configs: FeatureViewConfigs = [
    FeatureViewConfig(name="RETENTION_METRICS", query=RETENTION_METRICS_QUERY),
    FeatureViewConfig(name="BILLING_METRICS", query=BILLING_METRICS_QUERY),
]


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
    prediction_base_threshold=0.75,
    timestamp_col="start_date_month",
    partitions=partitions,
    levels=[
        Level(
            group_bys=[
                'brand',
                'sku_type',
                'channel',
            ]
        )
    ],
    time_horizons=time_horizons,
    model_steps=model_steps,
    # -- Common Model Step Features --
    cat_cols=[],
    num_cols=[
        'avg_recurring_days',
        'avg_recurring_price',
        'gross_adds_canceled_day_one_rate',
    ],
    boolean_cols=[],
    get_gross_adds_created_over_days_ago_column=get_gross_adds_created_over_days_ago_column,
    get_net_billings_days_column=get_net_billings_days_column,
    get_avg_net_billings_column=get_avg_net_billings_column,
)