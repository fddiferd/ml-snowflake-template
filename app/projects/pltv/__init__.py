from enum import Enum
from typing import TypeAlias

from projects.pltv.objects import Config, Level, Partition, PartitionItem

# MARK: - Time Horizons
class TimeHorizon(Enum):
    DAYS_30 = "30"
    DAYS_60 = "60"
    DAYS_90 = "90"
    DAYS_180 = "180"
    DAYS_365 = "365"
    DAYS_730 = "730"

TimeHorizons: TypeAlias = list[TimeHorizon]


# MARK: - Model Steps
class ModelStep(Enum):
    AVG_NET_BILLINGS_30_DAYS = "avg_net_billings_30_days"
    AVG_NET_BILLINGS_60_DAYS = "avg_net_billings_60_days"
    AVG_NET_BILLINGS_90_DAYS = "avg_net_billings_90_days"
    AVG_NET_BILLINGS_180_DAYS = "avg_net_billings_180_days"
    AVG_NET_BILLINGS_365_DAYS = "avg_net_billings_365_days"
    AVG_NET_BILLINGS_730_DAYS = "avg_net_billings_730_days"

    @property
    def target_col(self) -> str:
        return self.value.upper()

    @property
    def previous_step_min_cohort_cols(self) -> list[str]:
        eligible_first_rebill_col = 'eligible_first_rebills'
        match self:
            case ModelStep.AVG_NET_BILLINGS_30_DAYS:
                days_ago_col = 'gross_adds_created_over_30_days_ago'
            case ModelStep.AVG_NET_BILLINGS_60_DAYS:
                days_ago_col = 'gross_adds_created_over_60_days_ago'
            case ModelStep.AVG_NET_BILLINGS_90_DAYS:
                days_ago_col = 'gross_adds_created_over_90_days_ago'
            case ModelStep.AVG_NET_BILLINGS_180_DAYS:
                days_ago_col = 'gross_adds_created_over_180_days_ago'
            case ModelStep.AVG_NET_BILLINGS_365_DAYS:
                days_ago_col = 'gross_adds_created_over_365_days_ago'
            case ModelStep.AVG_NET_BILLINGS_730_DAYS:
                days_ago_col = 'gross_adds_created_over_730_days_ago'

        return [x.upper() for x in [eligible_first_rebill_col, days_ago_col]]

    @property
    def additional_regressor_cols(self) -> list[str]:
        match self:
            case ModelStep.AVG_NET_BILLINGS_30_DAYS:
                return []
            case ModelStep.AVG_NET_BILLINGS_60_DAYS:
                additional_regressor_cols = ['avg_net_billings_30_days']
            case ModelStep.AVG_NET_BILLINGS_90_DAYS:
                additional_regressor_cols = ['avg_net_billings_60_days']
            case ModelStep.AVG_NET_BILLINGS_180_DAYS:
                additional_regressor_cols = ['avg_net_billings_90_days']
            case ModelStep.AVG_NET_BILLINGS_365_DAYS:
                additional_regressor_cols = ['avg_net_billings_180_days']
            case ModelStep.AVG_NET_BILLINGS_730_DAYS:
                additional_regressor_cols = ['avg_net_billings_365_days']

        return [x.upper() for x in additional_regressor_cols]

ModelSteps: TypeAlias = list[ModelStep]


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