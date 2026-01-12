"""
PLTV Configuration
==================

Single source of truth for all configuration: enums, constants, column helpers.

Usage:
    from projects.pltv.config import (
        Level, Partition, TimeHorizon, ModelStep,
        GROSS_ADDS_COL, TIMESTAMP_COL,
        get_net_billings_col, get_total_net_billings_col,
        fv_configs,
    )
"""

from enum import Enum
from typing import TypeAlias, Any

from projects.pltv.data.queries.feature_views import RETENTION_METRICS_QUERY, BILLING_METRICS_QUERY, CROSS_SELL_METRICS_QUERY


# =============================================================================
# MARK: - Column Constants
# =============================================================================

GROSS_ADDS_COL = "GROSS_ADDS"
TIMESTAMP_COL = "COHORT_MONTH"
IS_PREDICTED_COL = "IS_PREDICTED"
PREDICTION_BASE_COL = "PREDICTION_BASE_COL"
TARGET_COL = "TARGET_COL"
GROSS_ADD_TYPE_COL = "GROSS_ADD__TYPE"

# Retention columns
ELIGIBLE_PROMO_ACTIVATIONS_COL = "ELIGIBLE_PROMO_ACTIVATIONS"
SURVIVED_PROMO_ACTIVATIONS_COL = "SURVIVED_PROMO_ACTIVATIONS_EXCL_RETRIES"
ELIGIBLE_FIRST_REBILLS_COL = "ELIGIBLE_FIRST_REBILLS"
SURVIVED_FIRST_REBILLS_COL = "SURVIVED_FIRST_REBILLS_EXCL_RETRIES"

# Cancellation columns
GROSS_ADDS_CANCELED_DAY_ONE_COL = "GROSS_ADDS_CANCELED_DAY_ONE"
GROSS_ADDS_CANCELED_DAY_THREE_COL = "GROSS_ADDS_CANCELED_DAY_THREE"
GROSS_ADDS_CANCELED_DAY_SEVEN_COL = "GROSS_ADDS_CANCELED_DAY_SEVEN"
GROSS_ADDS_CANCELED_DAY_ONE_RATE_COL = "GROSS_ADDS_CANCELED_DAY_ONE_RATE"
GROSS_ADDS_CANCELED_DAY_THREE_RATE_COL = "GROSS_ADDS_CANCELED_DAY_THREE_RATE"
GROSS_ADDS_CANCELED_DAY_SEVEN_RATE_COL = "GROSS_ADDS_CANCELED_DAY_SEVEN_RATE"

# Cross-sell columns (counts)
CROSS_SELL_ADDS_DAY_ONE_COL = "CROSS_SELL_ADDS_ONE_DAY_SINCE_GROSS_ADD"
CROSS_SELL_ADDS_DAY_THREE_COL = "CROSS_SELL_ADDS_THREE_DAYS_SINCE_GROSS_ADD"
CROSS_SELL_ADDS_DAY_SEVEN_COL = "CROSS_SELL_ADDS_SEVEN_DAYS_SINCE_GROSS_ADD"

# Cross-sell rate columns
CROSS_SELL_ADDS_DAY_ONE_RATE_COL = "CROSS_SELL_ADDS_DAY_ONE_RATE"
CROSS_SELL_ADDS_DAY_THREE_RATE_COL = "CROSS_SELL_ADDS_DAY_THREE_RATE"
CROSS_SELL_ADDS_DAY_SEVEN_RATE_COL = "CROSS_SELL_ADDS_DAY_SEVEN_RATE"

# Promo columns
AVG_PROMO_DAYS_COL = "AVG_PROMO_DAYS"
AVG_PROMO_PRICE_COL = "AVG_PROMO_PRICE"
PROMO_TO_RECURRING_DAYS_RATIO_COL = "PROMO_TO_RECURRING_DAYS_RATIO"
PROMO_TO_RECURRING_PRICE_RATIO_COL = "PROMO_TO_RECURRING_PRICE_RATIO"
AVG_RECURRING_DAYS_COL = "AVG_RECURRING_DAYS"
AVG_RECURRING_PRICE_COL = "AVG_RECURRING_PRICE"


# =============================================================================
# MARK: - Table Name Constants
# =============================================================================

TABLE_SPINE_DATA = "SPINE_DATA"
TABLE_RAW_RESULTS = "RAW_RESULTS"
TABLE_MODEL_METADATA = "MODEL_METADATA"
TABLE_TEST_TRAIN_SPLIT_METADATA = "TEST_TRAIN_SPLIT_METADATA"
TABLE_TEST_TRAIN_SPLIT_FEATURE_IMPORTANCES = "TEST_TRAIN_SPLIT_FEATURE_IMPORTANCES"
TABLE_TRAIN_PREDICT_METADATA = "TRAIN_PREDICT_METADATA"
TABLE_TRAIN_PREDICT_RESULTS = "TRAIN_PREDICT_RESULTS"
# TABLE_DATASET = "DATASET"

# Cache path for local development
CACHE_PATH = "app/projects/pltv/data/cache"


# =============================================================================
# MARK: - Status Constants
# =============================================================================

STATUS_TRAINED = "TRAINED"
STATUS_PREDICTED = "PREDICTED"
STATUS_BYPASSED = "BYPASSED"


# =============================================================================
# MARK: - Configuration Values
# =============================================================================

VERSION_NUMBER = 1
MIN_COHORT_SIZE = 400
PREDICTION_BASE_THRESHOLD = 0.50

# Additional join columns (also GROUP BY and cat features for the model)
ADDITIONAL_JOIN_COLS: list[str] = [
    GROSS_ADD_TYPE_COL,
]

# Feature columns for model training (cat features NOT used as join keys)
CAT_COLS: list[str] = []
NUM_COLS: list[str] = [
    AVG_RECURRING_DAYS_COL,
    AVG_RECURRING_PRICE_COL,
    GROSS_ADDS_CANCELED_DAY_ONE_RATE_COL,
    CROSS_SELL_ADDS_DAY_ONE_RATE_COL,
]
BOOLEAN_COLS: list[str] = []


# =============================================================================
# MARK: - Level Enum
# =============================================================================

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
        """Returns all Level names with a lower or equal value."""
        return [l.name for l in Level if l.value <= self.value]

    @property
    def sql_fields(self) -> str:
        """Returns comma-separated group_bys for SQL queries."""
        col_list = ADDITIONAL_JOIN_COLS + [col.upper() for col in self.group_bys]
        return ", ".join(col_list) + ("," if col_list else "")

Levels: TypeAlias = list[Level]
levels = [l for l in Level]


# =============================================================================
# MARK: - Partition Enum
# =============================================================================

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
                if value is True:
                    return [
                        AVG_PROMO_DAYS_COL,
                        AVG_PROMO_PRICE_COL,
                        PROMO_TO_RECURRING_DAYS_RATIO_COL,
                        PROMO_TO_RECURRING_PRICE_RATIO_COL,
                    ]
                return []
            case _:
                raise ValueError(f"Invalid partition item: {self}")


Partitions: TypeAlias = list[Partition]
partitions = [p for p in Partition]
partition_fields = [p.name.upper() for p in partitions]
partition_sql_fields = ", ".join(partition_fields) + ("," if partition_fields else "")


# =============================================================================
# MARK: - Time Horizon Enum
# =============================================================================

class TimeHorizon(Enum):
    DAYS_30 = "30"
    DAYS_60 = "60"
    DAYS_90 = "90"
    DAYS_180 = "180"
    DAYS_365 = "365"
    DAYS_730 = "730"


TimeHorizons: TypeAlias = list[TimeHorizon]
time_horizons = [t for t in TimeHorizon]


# =============================================================================
# MARK: - Model Step Enum
# =============================================================================

class ModelStep(Enum):
    # Multi-target steps (cancel + cross-sell together)
    DAY_3_METRICS = 1
    DAY_7_METRICS = 2
    # Retention Steps (single target)
    PROMO_ACTIVATION_RATE_EXCL_RETRIES = 3
    FIRST_REBILL_RATE_EXCL_RETRIES = 4
    # Billing Steps (single target)
    AVG_NET_BILLINGS_30_DAYS = 5
    AVG_NET_BILLINGS_60_DAYS = 6
    AVG_NET_BILLINGS_90_DAYS = 7
    AVG_NET_BILLINGS_180_DAYS = 8
    AVG_NET_BILLINGS_365_DAYS = 9
    AVG_NET_BILLINGS_730_DAYS = 10

    @property
    def target_cols(self) -> list[str]:
        """Return list of target columns for this step."""
        match self:
            case ModelStep.DAY_3_METRICS:
                return [GROSS_ADDS_CANCELED_DAY_THREE_RATE_COL, CROSS_SELL_ADDS_DAY_THREE_RATE_COL]
            case ModelStep.DAY_7_METRICS:
                return [GROSS_ADDS_CANCELED_DAY_SEVEN_RATE_COL, CROSS_SELL_ADDS_DAY_SEVEN_RATE_COL]
            case _:
                return [self.name.upper()]

    @property
    def is_multi_target(self) -> bool:
        """Return True if this step predicts multiple targets."""
        return len(self.target_cols) > 1

    @property
    def is_retention_target(self) -> bool:
        return self in (
            ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES,
            ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES,
        )

    @property
    def is_billing_step(self) -> bool:
        return "AVG_NET_BILLINGS" in self.name

    @property
    def rate_target_survived_cols(self) -> list[str]:
        """Returns the 'survived' columns for rate targets (used to back-calculate counts)."""
        match self:
            case ModelStep.DAY_3_METRICS:
                return [GROSS_ADDS_CANCELED_DAY_THREE_COL, CROSS_SELL_ADDS_DAY_THREE_COL]
            case ModelStep.DAY_7_METRICS:
                return [GROSS_ADDS_CANCELED_DAY_SEVEN_COL, CROSS_SELL_ADDS_DAY_SEVEN_COL]
            case ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
                return [SURVIVED_PROMO_ACTIVATIONS_COL]
            case ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
                return [SURVIVED_FIRST_REBILLS_COL]
            case _:
                return []

    @property
    def min_cohort_col(self) -> str:
        """Returns the column name for the previous step min cohort."""
        match self:
            case ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
                return ELIGIBLE_PROMO_ACTIVATIONS_COL
            case ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
                return ELIGIBLE_FIRST_REBILLS_COL
            case _:
                return get_gross_adds_created_over_days_ago_col(self._days_ago)

    def get_additional_regressor_cols(self, partition: "Partition", partition_value: Any) -> list[str]:
        """Returns target_cols from all previous steps as additional regressors."""
        additional_regressor_cols = partition.get_additional_regressor_cols(partition_value)
        previous_target_cols = []
        for step in self._previous_steps:
            previous_target_cols.extend(step.target_cols)
        return previous_target_cols + additional_regressor_cols

    def is_step_in_partition(self, partition: "Partition", partition_value: Any) -> bool:
        """Ensure that when PLAN__IS_PROMO is False, the promo step is skipped."""
        if self == ModelStep.PROMO_ACTIVATION_RATE_EXCL_RETRIES:
            if partition == Partition.PLAN__IS_PROMO and partition_value is False:
                return False
        return True

    def get_prediction_base_col(self, partition: "Partition", partition_value: Any) -> str:
        """For prediction, we need to compare the previous step col / base col."""
        if self == ModelStep.FIRST_REBILL_RATE_EXCL_RETRIES:
            if partition == Partition.PLAN__IS_PROMO and partition_value is True:
                return SURVIVED_PROMO_ACTIVATIONS_COL
        return GROSS_ADDS_COL

    @property
    def _days_ago(self) -> int:
        match self:
            case ModelStep.DAY_3_METRICS:
                return 3
            case ModelStep.DAY_7_METRICS:
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
                raise ValueError(f"Model Step {self.name} does not have a days ago value")

    @property
    def _previous_steps(self) -> list["ModelStep"]:
        """Returns all ModelStep enums with a lower value."""
        return [step for step in ModelStep if step.value < self.value]


ModelSteps: TypeAlias = list[ModelStep]
model_steps = [m for m in ModelStep]


# =============================================================================
# MARK: - Column Helper Functions
# =============================================================================

def get_gross_adds_created_over_days_ago_col(days: int | str) -> str:
    """Get column name for gross adds created over N days ago."""
    return f"GROSS_ADDS_CREATED_OVER_{days}_DAYS_AGO"


def get_net_billings_col(time_horizon: TimeHorizon) -> str:
    """Get column name for net billings at a time horizon."""
    return f"NET_BILLINGS_{time_horizon.value}_DAYS"


def get_avg_net_billings_col(time_horizon: TimeHorizon) -> str:
    """Get column name for average net billings at a time horizon."""
    return f"AVG_NET_BILLINGS_{time_horizon.value}_DAYS"


def get_total_net_billings_col(time_horizon: TimeHorizon) -> str:
    """Get column name for total net billings at a time horizon."""
    return f"TOTAL_NET_BILLINGS_{time_horizon.value}_DAYS"


def get_model_status_col(step: ModelStep) -> str:
    """Get the model status column name for a step."""
    return f"{step.name}_MODEL"


# =============================================================================
# MARK: - Key and SQL Field Helpers
# =============================================================================

def get_cat_cols(level: Level) -> list[str]:
    """Return all categorical columns for the model (CAT_COLS + ADDITIONAL_JOIN_COLS + level group_bys)."""
    return CAT_COLS + ADDITIONAL_JOIN_COLS + [col.upper() for col in level.group_bys]


def get_num_cols(partition: Partition, partition_value: Any, step: ModelStep) -> list[str]:
    """Return additional regressors based on the step and partition as well as the global num cols."""
    return NUM_COLS + step.get_additional_regressor_cols(partition, partition_value)


def get_join_keys(level: Level) -> list[str]:
    """Get join keys for a level."""
    return [TIMESTAMP_COL] + partition_fields + ADDITIONAL_JOIN_COLS + [col.upper() for col in level.group_bys]


# =============================================================================
# MARK: - Feature View Config
# =============================================================================

class FeatureViewConfig:
    """Simple data class for feature view configuration."""
    def __init__(self, name: str, query: str):
        self.name = name
        self.query = query


FeatureViewConfigs: TypeAlias = list[FeatureViewConfig]

fv_configs: FeatureViewConfigs = [
    FeatureViewConfig(name="RETENTION_METRICS", query=RETENTION_METRICS_QUERY),
    FeatureViewConfig(name="BILLING_METRICS", query=BILLING_METRICS_QUERY),
    FeatureViewConfig(name="CROSS_SELL_METRICS", query=CROSS_SELL_METRICS_QUERY),
]
