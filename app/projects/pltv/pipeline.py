import logging
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin

from projects.pltv import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Prophet's verbose output
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for Prophet.
    Allows Prophet to be used in sklearn Pipelines.
    """
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.5,
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: int = 10,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model_ = None
        self.regressor_cols_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProphetRegressor':
        """
        Fit the Prophet model.
        
        Args:
            X: DataFrame with 'ds' column (datetime) and regressor columns
            y: Target series
        """
        logger.info(f"Fitting ProphetRegressor with {len(X)} samples")
        
        # Initialize Prophet model
        self.model_ = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,  # type: ignore
            weekly_seasonality=self.weekly_seasonality,  # type: ignore
            daily_seasonality=self.daily_seasonality,  # type: ignore
        )
        
        # Get regressor columns (all except 'ds')
        self.regressor_cols_ = [col for col in X.columns if col != 'ds']
        
        # Add regressors to Prophet
        for regressor in self.regressor_cols_:
            logger.info(f"Adding regressor: {regressor}")
            self.model_.add_regressor(regressor)
        
        # Prepare training DataFrame
        train_df = X.copy()
        train_df['y'] = y.values
        
        # Fit the model
        logger.info("Fitting Prophet model...")
        self.model_.fit(train_df)
        logger.info("Prophet model fitted successfully!")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore
        """
        Make predictions using the fitted Prophet model.
        
        Args:
            X: DataFrame with 'ds' column and regressor columns
            
        Returns:
            Array of predictions
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        logger.info(f"Making predictions on {len(X)} samples")
        
        # Make predictions
        forecast = self.model_.predict(X)
        
        return forecast['yhat'].values  # type: ignore
    
    def predict_with_intervals(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """
        Make predictions with uncertainty intervals.
        
        Args:
            X: DataFrame with 'ds' column and regressor columns
            
        Returns:
            DataFrame with yhat, yhat_lower, yhat_upper
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        forecast = self.model_.predict(X)
        return forecast[['yhat', 'yhat_lower', 'yhat_upper']]  # type: ignore


def get_preprocessor(
    cat_cols: list[str],
    num_cols: list[str],
    max_categories: int | None = None,
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Args:
        cat_cols: Categorical columns to one-hot encode
        num_cols: Numerical columns to scale
        max_categories: Maximum categories for one-hot encoding
        
    Returns:
        Configured ColumnTransformer
    """
    transformers = []
    
    if num_cols:
        transformers.append((
            'num', 
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
            ]), 
            num_cols
        ))
    
    if cat_cols:
        transformers.append((
            'cat', 
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    max_categories=max_categories
                ))
            ]), 
            cat_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
    )
    
    preprocessor.set_output(transform="pandas")  # type: ignore
    
    return preprocessor


def get_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    max_categories: int | None = None,
    prophet_params: dict[str, Any] | None = None,
) -> tuple[ProphetRegressor, ColumnTransformer]:
    """
    Create a Prophet pipeline with preprocessing.
    
    Args:
        cat_cols: Categorical columns to one-hot encode
        num_cols: Numerical columns to scale
        max_categories: Maximum categories for one-hot encoding
        prophet_params: Prophet parameters (None for defaults)
        
    Returns:
        Tuple of (Pipeline, preprocessor) - preprocessor returned separately for feature name access
    """
    # Create preprocessor
    preprocessor = get_preprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        max_categories=max_categories,
    )
    
    # Default Prophet parameters
    if prophet_params is None:
        prophet_params = {
            'changepoint_prior_scale': 0.5,
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': 10,
            'weekly_seasonality': False,
            'daily_seasonality': False,
        }
    
    # Note: We don't include preprocessing in the pipeline because
    # Prophet needs special handling of the 'ds' column
    logger.info(f'Pipeline created for {len(cat_cols)} categorical + {len(num_cols)} numerical features')
    
    return ProphetRegressor(**prophet_params), preprocessor


def prepare_data_for_prophet(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str | None,
    preprocessor: ColumnTransformer,
    fit_preprocessor: bool = True,
) -> pd.DataFrame:
    """
    Prepare data for Prophet by applying preprocessing to features
    while preserving the timestamp column.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        timestamp_col: Timestamp column name
        preprocessor: Fitted or unfitted ColumnTransformer
        fit_preprocessor: Whether to fit the preprocessor (True for train, False for predict)
        
    Returns:
        DataFrame with 'ds' column, target (if present), and preprocessed features
    """
    result_df = pd.DataFrame()
    
    # Add timestamp as 'ds'
    if timestamp_col and timestamp_col in df.columns:
        result_df['ds'] = pd.to_datetime(df[timestamp_col])
    
    # Transform features
    if fit_preprocessor:
        transformed = preprocessor.fit_transform(df)
    else:
        transformed = preprocessor.transform(df)
    
    # Add transformed features
    if isinstance(transformed, pd.DataFrame):
        for col in transformed.columns:
            result_df[col] = transformed[col].values
    
    # Add target if present and not NaN
    if target_col in df.columns:
        result_df['y'] = df[target_col].values
    
    return result_df


def run_pipeline(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    max_categories: int | None = None,
    prophet_params: dict[str, Any] | None = None,
) -> tuple[ProphetRegressor, pd.DataFrame]:  # type: ignore
    """
    Main pipeline function that orchestrates the entire process.
    
    IMPORTANT: This function preserves the original predict_df structure completely.
    All original columns are kept intact, and only prediction columns are added:
    - pred_y: Point predictions
    - pred_y_lower: Lower bound of prediction interval
    - pred_y_upper: Upper bound of prediction interval
    
    Args:
        train_df: Training DataFrame (with target column populated)
        predict_df: Prediction DataFrame (original structure will be preserved)
        max_categories: Maximum categories for one-hot encoding (None for no limit)
        prophet_params: Prophet hyperparameters (None for defaults)
        
    Returns:
        Tuple of (trained_model, predictions_dataframe)
        - trained_model: Fitted ProphetRegressor instance
        - predictions_dataframe: Original predict_df with 3 new prediction columns added
    """
    logger.info("Starting PLTV Prophet pipeline...")
    
    # Get the target column (longest time horizon)
    longest_horizon = config.time_horizons[-1]  # DAYS_730
    target_col = config.get_avg_net_billings_column(longest_horizon)
    logger.info(f"Target column: {target_col}")
    
    # Get feature columns
    level = config.levels[0]
    cat_cols = level.get_all_cat_cols(config.cat_cols)
    num_cols = config.num_cols
    
    logger.info(f"Categorical columns: {cat_cols}")
    logger.info(f"Numerical columns: {num_cols}")
    
    # Create pipeline
    model, preprocessor = get_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        max_categories=max_categories,
        prophet_params=prophet_params,
    )
    
    # Prepare training data
    logger.info("Preparing training data...")
    train_prepared = prepare_data_for_prophet(
        df=train_df,
        target_col=target_col,
        timestamp_col=config.timestamp_col,
        preprocessor=preprocessor,
        fit_preprocessor=True,
    )
    
    # Extract X and y for training
    X_train = train_prepared.drop(columns=['y'])
    y_train: pd.Series = train_prepared['y']  # type: ignore
    
    # Fit the model
    logger.info(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Prepare prediction data
    logger.info("Preparing prediction data...")
    predict_prepared = prepare_data_for_prophet(
        df=predict_df,
        target_col=target_col,
        timestamp_col=config.timestamp_col,
        preprocessor=preprocessor,
        fit_preprocessor=False,
    )
    
    # Make predictions with intervals
    X_predict = predict_prepared.drop(columns=['y'], errors='ignore')
    predictions = model.predict_with_intervals(X_predict)  # type: ignore
    
    # CRITICAL: Preserve original dataframe completely
    # Strategy: Copy original predict_df, then add only prediction columns
    # This ensures all original columns, dtypes, and structure are maintained
    logger.info(f"Preserving original dataframe with {len(predict_df.columns)} columns")
    
    result_df = predict_df.reset_index(drop=True).copy()
    predictions_aligned = predictions.reset_index(drop=True)
    
    # Add only the prediction columns (no modifications to existing columns)
    result_df['pred_y'] = predictions_aligned['yhat'].values
    result_df['pred_y_lower'] = predictions_aligned['yhat_lower'].values
    result_df['pred_y_upper'] = predictions_aligned['yhat_upper'].values
    
    logger.info(f"Result dataframe now has {len(result_df.columns)} columns (added 3 prediction columns)")
    logger.info("Pipeline complete!")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean predicted value: {result_df['pred_y'].mean():.2f}")
    logger.info(f"  Median predicted value: {result_df['pred_y'].median():.2f}")
    logger.info(f"  Min predicted value: {result_df['pred_y'].min():.2f}")
    logger.info(f"  Max predicted value: {result_df['pred_y'].max():.2f}")
    
    return model, result_df
