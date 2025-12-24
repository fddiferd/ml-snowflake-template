from enum import Enum
from typing import TypeAlias, Any


# MARK: Partition
class Partition(Enum):
    PLAN__IS_PROMO = "plan__is_promo"

    @property
    def values(self) -> list[Any]:
        match self:
            case Partition.PLAN__IS_PROMO:
                return [True, False]
            case _:
                raise ValueError(f"Invalid partition item: {self}")

    def get_additional_regressor_cols(self, value: Any) -> list[str]:
        match self:
            case Partition.PLAN__IS_PROMO:
                if value == True:
                    cols = [
                        'avg_promo_days',
                        'avg_promo_price', 
                        'promo_to_recurring_days_ratio', 
                        'promo_to_recurring_price_ratio'
                    ]
                else:
                    cols = []
                return [col.upper() for col in cols]
            case _:
                raise ValueError(f"Invalid partition item: {self}")

Partitions: TypeAlias = list[Partition]
partitions = [p for p in Partition]


# MARK: - Time Horizons
class TimeHorizon(Enum):
    DAYS_30 = "30"
    DAYS_60 = "60"
    DAYS_90 = "90"
    DAYS_180 = "180"
    DAYS_365 = "365"
    DAYS_730 = "730"

TimeHorizons: TypeAlias = list[TimeHorizon]
time_horizons = [t for t in TimeHorizon]


# MARK: - Model Steps
class ModelStep(Enum):
    # Cancelation Steps
    GROSS_ADDS_CANCELED_DAY_THREE_RATE = 1
    GROSS_ADDS_CANCELED_DAY_SEVEN_RATE = 2
    # Retention Steps
    PROMO_ACTIVATION_RATE = 3
    FIRST_REBILL_RATE = 4
    # Billing Steps
    AVG_NET_BILLINGS_30_DAYS = 5
    AVG_NET_BILLINGS_60_DAYS = 6
    AVG_NET_BILLINGS_90_DAYS = 7
    AVG_NET_BILLINGS_180_DAYS = 8
    AVG_NET_BILLINGS_365_DAYS = 9
    AVG_NET_BILLINGS_730_DAYS = 10

    @property
    def target_col(self) -> str:
        return self.name.upper()

    @property
    def pred_col(self) -> str:
        return f'PRED_{self.target_col}'

    @property
    def pred_lower_col(self) -> str:
        return f'PRED_{self.target_col}_LOWER'

    @property
    def pred_upper_col(self) -> str:
        return f'PRED_{self.target_col}_UPPER'

    def get_additional_regressor_cols(self, partition: Partition, partition_value: Any) -> list[str]:
        """Returns target_cols from all previous steps as additional regressors."""
        additional_regressor_cols = partition.get_additional_regressor_cols(partition_value)
        return [step.target_col for step in self._previous_steps] + additional_regressor_cols

    def is_step_in_partition(self, partition: Partition, partition_value: Any) -> bool:
        """Ensure that when PLAN__IS_PROMO is False, the step is skipped"""
        match self:
            case ModelStep.PROMO_ACTIVATION_RATE:
                if partition == Partition.PLAN__IS_PROMO and partition_value == False:
                    return False
                return True
            case _:
                return True

    @property
    def previous_step_min_cohort_col(self) -> str:
        """Returns the column name for the previous step min cohort."""
        match self:
            case ModelStep.PROMO_ACTIVATION_RATE:
                return 'eligible_promo_activations'.upper()
            case ModelStep.FIRST_REBILL_RATE:
                return 'eligible_first_rebills'.upper()
            case _:
                return f'gross_adds_created_over_{self._days_ago_col}_days_ago'.upper()

    @property
    def _days_ago_col(self) -> int:
        match self:
            case ModelStep.GROSS_ADDS_CANCELED_DAY_THREE_RATE:
                return 3
            case ModelStep.GROSS_ADDS_CANCELED_DAY_SEVEN_RATE:
                return 7
            case ModelStep.AVG_NET_BILLINGS_30_DAYS:
                return 30
            case ModelStep.AVG_NET_BILLINGS_60_DAYS:
                return 60
            case ModelStep.AVG_NET_BILLINGS_90_DAYS:
                return 90
            case ModelStep.AVG_NET_BILLINGS_180_DAYS:
                return 180
            case ModelStep.AVG_NET_BILLINGS_365_DAYS:
                return 365
            case ModelStep.AVG_NET_BILLINGS_730_DAYS:
                return 730
            case _:
                raise ValueError(f"Model Step {self.name} does not have a days ago column")

    @property
    def _previous_steps(self) -> list["ModelStep"]:
        """Returns all ModelStep enums with a lower value."""
        return [step for step in ModelStep if step.value < self.value]


ModelSteps: TypeAlias = list[ModelStep]
model_steps = [m for m in ModelStep]