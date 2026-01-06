from enum import Enum
from typing import TypeAlias, Any


# MARK: Level
class Level(Enum):
    BRAND = 1
    SKU_TYPE = 2
    CHANNEL = 3
    TRAFFIC_SOURCE = 4
    CAMPAIGN = 5

    @property
    def name(self) -> str:
        match self:
            case Level.BRAND:
                return "BRAND"
            case Level.SKU_TYPE:
                return "SKU_TYPE"
            case Level.CHANNEL:
                return "CHANNEL"
            case Level.TRAFFIC_SOURCE:
                return "TRAFFIC_SOURCE"
            case Level.CAMPAIGN:
                return "CAMPAIGN"
            case _:
                raise ValueError(f"Invalid level: {self}")

    @property
    def group_bys(self) -> list[str]:
        """Returns all ModelStep enums with a lower value."""
        return [l.name for l in Level if l.value <= self.value]

    @property
    def sql_fields(self) -> str:
        # Ensure there is a comma after every item, including the last one
        return ", ".join(self.group_bys) + ("," if self.group_bys else "")

    def get_key_fields(self) -> list[tuple[str, list[str]]]:
        """Returns list of tuples with level name and cumulative sublists of group_bys."""
        levels = [l for l in Level if l.value <= self.value]
        return [(l.name, self.group_bys[:i+1]) for i, l in enumerate(levels)]

Levels: TypeAlias = list[Level]
levels = [l for l in Level]

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
parition_fields = [p.name.upper() for p in partitions]
parition_sql_fields = ", ".join(parition_fields) + ("," if parition_fields else "")


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
    PROMO_ACTIVATION_RATE_EXCL_RETRIES = 3
    FIRST_REBILL_RATE_EXCL_RETRIES = 4
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
    def rate_target_survived_col(self) -> str | None:
        match self:
            case ModelStep.GROSS_ADDS_CANCELED_DAY_THREE_RATE:
                return 'gross_adds_canceled_day_three'.upper()
            case ModelStep.GROSS_ADDS_CANCELED_DAY_SEVEN_RATE:
                return 'gross_adds_canceled_day_seven'.upper()
            case ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
                return 'survived_promo_activations_excl_retries'.upper()
            case ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
                return 'survived_first_rebills_excl_retries'.upper()
            case _:
                if 'NET_BILLINGS' in self.name: # for billing steps, we don't have a rate target survived column
                    return None
                raise ValueError(f"Model Step {self.name} does not have a rate target survived column")

    @property
    def min_cohort_col(self) -> str:
        """Returns the column name for the previous step min cohort."""
        match self:
            case ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
                return 'eligible_promo_activations'.upper()
            case ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
                return 'eligible_first_rebills'.upper()
            case _:
                return f'gross_adds_created_over_{self._days_ago_col}_days_ago'.upper()

    def get_additional_regressor_cols(self, partition: Partition, partition_value: Any) -> list[str]:
        """Returns target_cols from all previous steps as additional regressors."""
        additional_regressor_cols = partition.get_additional_regressor_cols(partition_value)
        return [step.target_col for step in self._previous_steps] + additional_regressor_cols

    def is_step_in_partition(self, partition: Partition, partition_value: Any) -> bool:
        """Ensure that when PLAN__IS_PROMO is False, the step is skipped"""
        match self:
            case ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
                if partition == Partition.PLAN__IS_PROMO and partition_value == False:
                    return False
                return True
            case _:
                return True

    def get_prediction_base_col(self, partition: Partition, partition_value: Any) -> str:
        """For prediction, we need to compare the previous step col / base col to see if its greater than a certain value to determine if its BAKED"""
        gross_adds_col = 'gross_adds'.upper()
        match self:
            case ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
                if partition == Partition.PLAN__IS_PROMO and partition_value == True:
                    return 'survived_promo_activations_excl_retries'.upper()
                return gross_adds_col
            case _:
                return gross_adds_col

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