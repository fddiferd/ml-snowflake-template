from enum import Enum
from typing import TypeAlias


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
    def pred_col(self) -> str:
        return f'PRED_{self.target_col}'

    @property
    def pred_lower_col(self) -> str:
        return f'PRED_{self.target_col}_LOWER'

    @property
    def pred_upper_col(self) -> str:
        return f'PRED_{self.target_col}_UPPER'

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