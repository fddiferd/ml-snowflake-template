from projects.pltv.core.enums import (
    TimeHorizon, 
    ModelStep,
)
from projects.pltv.core.base_models import (
    Config, 
    Level, 
    Partition, 
    PartitionItem,
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
    timestamp_col="start_date_month",
    partition=Partition(
        name="plan__is_promo",
        items=[
            PartitionItem(
                value=True, additional_regressor_cols=[
                    'avg_promo_price',
                    'promo_to_recurring_days_ratio',
                    'promo_to_recurring_price_ratio',
                    # 'promo_activation_rate',
                ]
                ),
            PartitionItem(
                value=False, additional_regressor_cols=[]
            ),
        ]
    ),
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
    model_steps=[m for m in ModelStep],
    # -- Common Model Step Features --
    cat_cols=[],
    num_cols=[
        # plan features
        'avg_promo_days',
        'avg_recurring_days',
        'avg_recurring_price',
        # Cancelation features
        'gross_adds_canceled_day_one_rate',
        'gross_adds_canceled_day_three_rate',
        'gross_adds_canceled_day_seven_rate',
        # Retention features
        # 'first_rebill_rate',
    ],
    boolean_cols=[],
    get_gross_adds_created_over_days_ago_column=get_gross_adds_created_over_days_ago_column,
    get_net_billings_days_column=get_net_billings_days_column,
    get_avg_net_billings_column=get_avg_net_billings_column,
)