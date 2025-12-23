import logging
import pandas as pd

from projects.pltv import config
# from projects.pltv.session import get_session
# from projects.pltv.dataset import get_dataset
from projects.pltv.feature_engineering import clean_df
from projects.pltv.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# session = get_session()
# dataset = get_dataset(session, Level(group_bys=["brand", "sku_type", "channel"]))
# df = dataset.to_pandas()

df = pd.read_parquet("output_dataset.parquet")
clean_df(df)

# Filter for the partition we want to work with
test_df = df[df[config.partition_col] == True].copy()

# Prepare training set: has target and meets minimum cohort size
target_col_730 = config.get_avg_net_billings_column(config.time_horizons[-1])
target_col_365 = config.get_avg_net_billings_column(config.time_horizons[-2])

train_df: pd.DataFrame = test_df[
    (test_df[target_col_730].notna())  # type: ignore
    & (test_df['GROSS_ADDS'] >= config.min_cohort_size)  # type: ignore
].copy()

# Prepare prediction set: missing target but has shorter horizon data
predict_df: pd.DataFrame = test_df[
    (test_df[target_col_730].isna())  # type: ignore
    & (test_df[target_col_365].notna())  # type: ignore
].copy()

logger.info(f"Training samples: {len(train_df)}")
logger.info(f"Prediction samples: {len(predict_df)}")

# Run the pipeline with optional parameters
prophet_params = {
    'changepoint_prior_scale': 0.5,  # How flexible the trend is
    'seasonality_mode': 'multiplicative',  # 'additive' or 'multiplicative'
    'yearly_seasonality': 10,  # Fourier terms for yearly seasonality
    'weekly_seasonality': False,
    'daily_seasonality': False,
}

model, predictions_df = run_pipeline(
    train_df=train_df,
    predict_df=predict_df,
    max_categories=20,  # Limit categories in one-hot encoding
    prophet_params=prophet_params,
)

# Verify original dataframe structure is preserved
logger.info(f"\nOriginal predict_df shape: {predict_df.shape}")
logger.info(f"Predictions_df shape: {predictions_df.shape}")
logger.info(f"New columns added: pred_y, pred_y_lower, pred_y_upper")

# Display results with key columns
logger.info("\nSample predictions:")
print(predictions_df.head(10))
predictions_df.to_csv("predictions_output.csv", index=False)

# Show that all original columns are preserved
original_cols = set(predict_df.columns)
result_cols = set(predictions_df.columns)
new_cols = result_cols - original_cols
logger.info(f"\nAll {len(original_cols)} original columns preserved: {original_cols.issubset(result_cols)}")
logger.info(f"New columns added: {sorted(new_cols)}")

# Optionally save predictions
predictions_df.to_parquet("predictions_output.parquet", index=False)
logger.info("\nPredictions saved to predictions_output.parquet")