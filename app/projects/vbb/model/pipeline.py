import logging
import numpy as np
import pandas as pd

from src.pipeline.xgboost import run_pipeline, PRED_YHAT_COL, PRED_YHAT_LOWER_COL, PRED_YHAT_UPPER_COL, XGBoostRegressorWrapper
from src.utils.model import evaluate_model
from src.base_models.evaluation import EvaluationResult
from projects.vbb.constants import (
    TARGET_COL,
    TIME_COL,
    CAT_COLS,
    NUM_COLS,
    BOOLEAN_COLS,
    TARGET_ENCODE_COLS,
    XGBOOST_PARAMS,
)


logger = logging.getLogger(__name__)

MAX_CATEGORIES = 50

GENERATED_TIME_COLS = ['GROSS_ADD_MONTH', 'GROSS_ADD_DOW']


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, XGBoostRegressorWrapper, EvaluationResult, pd.DataFrame]:
    """Train on train_df, predict on test_df, evaluate, and return results.

    Pre-processing steps (VBB-specific, before shared pipeline):
      1. Target-encode high-cardinality categoricals
      2. Extract time features from GROSS_ADD__CREATED
      3. log1p-transform the target

    Post-processing:
      4. expm1 predictions back to dollar space
      5. Evaluate on original scale
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1 -- target encoding
    te_mappings = _fit_target_encoder(train_df)
    train_df = _apply_target_encoder(train_df, te_mappings)
    test_df = _apply_target_encoder(test_df, te_mappings)

    # 2 -- time features
    train_df = _add_time_features(train_df)
    test_df = _add_time_features(test_df)

    # 3 -- log-transform target
    train_df[TARGET_COL] = np.log1p(train_df[TARGET_COL])
    test_df[TARGET_COL] = np.log1p(test_df[TARGET_COL])

    num_cols_extended = NUM_COLS + [f'{c}_TE' for c in TARGET_ENCODE_COLS] + GENERATED_TIME_COLS

    result_df, model = run_pipeline(
        train_df=train_df,
        predict_df=test_df,
        target_col=TARGET_COL,
        cat_cols=CAT_COLS,
        num_cols=num_cols_extended,
        boolean_cols=BOOLEAN_COLS,
        max_categories=MAX_CATEGORIES,
        xgboost_params=XGBOOST_PARAMS,
    )

    # 4 -- inverse transform predictions and target back to dollar space
    result_df[PRED_YHAT_COL] = np.expm1(result_df[PRED_YHAT_COL].clip(lower=0))
    result_df[PRED_YHAT_LOWER_COL] = np.expm1(result_df[PRED_YHAT_LOWER_COL].clip(lower=0))
    result_df[PRED_YHAT_UPPER_COL] = np.expm1(result_df[PRED_YHAT_UPPER_COL].clip(lower=0))
    result_df[TARGET_COL] = np.expm1(result_df[TARGET_COL])

    # 5 -- evaluate on original dollar scale
    y_true = pd.Series(result_df[TARGET_COL])
    y_pred = pd.Series(result_df[PRED_YHAT_COL])
    evaluation = evaluate_model(y_true=y_true, y_pred=y_pred)
    logger.info(
        f"Evaluation -- RMSE: {evaluation.rmse:.2f}  MAE: {evaluation.mae:.2f}  "
        f"R2: {evaluation.r2:.4f}  Spearman: {evaluation.spearman:.4f}  "
        f"MAPE: {evaluation.mape:.1f}%"
    )

    feature_importance_df = model.get_feature_importance()
    logger.info(f"Top features: {feature_importance_df.head(10)['feature'].tolist()}")

    return result_df, model, evaluation, feature_importance_df


def predict_only(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> tuple[pd.DataFrame, XGBoostRegressorWrapper]:
    """Train on full train_df then predict on predict_df (no evaluation).

    Used by the daily prediction flow where predict_df has no target column.
    Returns predictions in dollar space with YHAT clipped at 0.
    """
    train_df = train_df.copy()
    predict_df = predict_df.copy()

    te_mappings = _fit_target_encoder(train_df)
    train_df = _apply_target_encoder(train_df, te_mappings)
    predict_df = _apply_target_encoder(predict_df, te_mappings)

    train_df = _add_time_features(train_df)
    predict_df = _add_time_features(predict_df)

    train_df[TARGET_COL] = np.log1p(train_df[TARGET_COL])

    num_cols_extended = NUM_COLS + [f'{c}_TE' for c in TARGET_ENCODE_COLS] + GENERATED_TIME_COLS

    result_df, model = run_pipeline(
        train_df=train_df,
        predict_df=predict_df,
        target_col=TARGET_COL,
        cat_cols=CAT_COLS,
        num_cols=num_cols_extended,
        boolean_cols=BOOLEAN_COLS,
        max_categories=MAX_CATEGORIES,
        xgboost_params=XGBOOST_PARAMS,
    )

    result_df[PRED_YHAT_COL] = np.expm1(result_df[PRED_YHAT_COL].clip(lower=0))
    result_df[PRED_YHAT_LOWER_COL] = np.expm1(result_df[PRED_YHAT_LOWER_COL].clip(lower=0))
    result_df[PRED_YHAT_UPPER_COL] = np.expm1(result_df[PRED_YHAT_UPPER_COL].clip(lower=0))
    logger.info(f"Predicted {len(result_df):,} rows (mean YHAT: ${result_df[PRED_YHAT_COL].mean():.2f})")

    return result_df, model


# ---------------------------------------------------------------------------
# Target encoding helpers
# ---------------------------------------------------------------------------

def _fit_target_encoder(train_df: pd.DataFrame) -> dict[str, dict]:
    """Compute mean target per category for each high-cardinality column."""
    mappings: dict[str, dict] = {}
    for col in TARGET_ENCODE_COLS:
        means: dict = train_df.groupby(col)[TARGET_COL].mean().to_dict()
        mappings[col] = means
    return mappings


def _apply_target_encoder(
    df: pd.DataFrame,
    mappings: dict[str, dict],
) -> pd.DataFrame:
    """Map each high-card column to its target-encoded value, filling unseen
    categories with the global training mean."""
    for col, means in mappings.items():
        global_mean = float(np.mean(list(means.values())))
        df[f'{col}_TE'] = df[col].map(means).fillna(global_mean).astype(float)  # type: ignore[arg-type]
    return df


# ---------------------------------------------------------------------------
# Time feature helpers
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month and day-of-week from the gross add timestamp."""
    ts = pd.to_datetime(df[TIME_COL], errors='coerce')
    df['GROSS_ADD_MONTH'] = ts.dt.month.fillna(0).astype(int)
    df['GROSS_ADD_DOW'] = ts.dt.dayofweek.fillna(0).astype(int)
    return df
