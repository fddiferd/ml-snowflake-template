"""
XGBoost Pipeline Module

Provides a scikit-learn compatible wrapper for XGBoost with preprocessing
and prediction interval support.

Usage:
    from src.pipeline.xgboost import get_pipeline, prepare_data
    
    # Define your feature columns
    cat_cols = ['brand', 'channel', 'sku_type']
    num_cols = ['price', 'days_active', 'cancellation_rate']
    
    # Create pipeline
    model, preprocessor = get_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        xgboost_params={'n_estimators': 200, 'max_depth': 8}
    )
    
    # Prepare and train
    train_prepared = prepare_data(train_df, 'target_col', preprocessor, fit_preprocessor=True)
    X_train = train_prepared.drop(columns=['y'])
    y_train = train_prepared['y']
    model.fit(X_train, y_train)
    
    # Predict
    predict_prepared = prepare_data(test_df, 'target_col', preprocessor, fit_preprocessor=False)
    predictions = model.predict_with_intervals(predict_prepared.drop(columns=['y'], errors='ignore'))
"""
import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, RegressorMixin


logger = logging.getLogger(__name__)

PRED_YHAT_COL = 'YHAT'
PRED_YHAT_LOWER_COL = 'YHAT_LOWER'
PRED_YHAT_UPPER_COL = 'YHAT_UPPER'


class XGBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for XGBoost with prediction intervals.
    Provides similar interface to ProphetRegressor.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.y_std_ = None  # For prediction intervals
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostRegressorWrapper':
        """
        Fit the XGBoost model.
        
        Args:
            X: DataFrame with features
            y: Target series
        """
        logger.info(f"Fitting XGBoostRegressor with {len(X)} samples, {len(X.columns)} features")
        
        # Initialize XGBoost model
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit the model
        logger.info("Fitting XGBoost model...")
        self.model_.fit(X, y)
        
        # Calculate residual standard deviation for prediction intervals
        train_predictions = self.model_.predict(X)
        residuals = y - train_predictions
        self.y_std_ = np.std(residuals)
        
        logger.info("XGBoost model fitted successfully!")
        logger.info(f"Training RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore
        """
        Make predictions using the fitted XGBoost model.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        logger.info(f"Making predictions on {len(X)} samples")
        
        return self.model_.predict(X)
    
    def predict_with_intervals(
        self, 
        X: pd.DataFrame, 
        confidence_level: float = 0.95,
        pred_col: str = PRED_YHAT_COL,
        pred_lower_col: str = PRED_YHAT_LOWER_COL,
        pred_upper_col: str = PRED_YHAT_UPPER_COL,
    ) -> pd.DataFrame:
        """
        Make predictions with uncertainty intervals.
        Uses residual standard deviation from training to estimate intervals.
        
        Args:
            X: DataFrame with features
            confidence_level: Confidence level for intervals (default 0.95)
            
        Returns:
            DataFrame with pred_col, pred_lower_col, pred_upper_col
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        # Get point predictions
        predictions = self.model_.predict(X)
        
        # Calculate interval width (approximation using normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        interval_width = z_score * self.y_std_
        
        # Create result dataframe
        result = pd.DataFrame({
            pred_col: predictions,
            pred_lower_col: predictions - interval_width,
            pred_upper_col: predictions + interval_width
        })
        
        return result
    
    def get_feature_importance(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        importance_dict = self.model_.get_booster().get_score(importance_type='gain')
        
        # If feature names not provided, use from importance dict
        if feature_names is None:
            feature_names = list(importance_dict.keys())
        
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df


def get_preprocessor(
    cat_cols: list[str],
    num_cols: list[str],
    boolean_cols: list[str],
    max_categories: int | None = None,
    impute_strategy: str = 'median',
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Args:
        cat_cols: Categorical columns to one-hot encode
        num_cols: Numerical columns to scale
        max_categories: Maximum categories for one-hot encoding
        impute_strategy: Strategy for numerical imputation ('median', 'mean', or 'constant')
        
    Returns:
        Configured ColumnTransformer
    """
    transformers = []
    
    if num_cols:
        # Use median imputation by default (better than filling with 0)
        transformers.append((
            'num', 
            Pipeline([
                ('imputer', SimpleImputer(strategy=impute_strategy, fill_value=0)),
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
    if boolean_cols:
        transformers.append((
            'bool',
            Pipeline([
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ]),
            boolean_cols
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
    boolean_cols: list[str],
    max_categories: int | None = None,
    xgboost_params: dict[str, Any] | None = None,
    impute_strategy: str = 'median',
) -> tuple[XGBoostRegressorWrapper, ColumnTransformer]:
    """
    Create an XGBoost pipeline with preprocessing.
    
    Args:
        cat_cols: Categorical columns to one-hot encode
        num_cols: Numerical columns to scale
        boolean_cols: Boolean columns to convert to integers
        max_categories: Maximum categories for one-hot encoding
        xgboost_params: XGBoost parameters (None for defaults)
        impute_strategy: Strategy for numerical imputation
        
    Returns:
        Tuple of (XGBoostRegressorWrapper, preprocessor)
    """
    # Create preprocessor
    preprocessor = get_preprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        boolean_cols=boolean_cols,
        max_categories=max_categories,
        impute_strategy=impute_strategy,
    )
    
    # Default XGBoost parameters
    if xgboost_params is None:
        xgboost_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    
    logger.info(f'Pipeline created for {len(cat_cols)} categorical + {len(num_cols)} numerical features')
    
    return XGBoostRegressorWrapper(**xgboost_params), preprocessor


def prepare_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit_preprocessor: bool = True,
) -> pd.DataFrame:
    """
    Prepare data for XGBoost by applying preprocessing to features.
    
    Args:
        df: Input DataFrame
        preprocessor: Fitted or unfitted ColumnTransformer
        fit_preprocessor: Whether to fit the preprocessor (True for train, False for predict)
        
    Returns:
        DataFrame with preprocessed features and target (if present)
    """
    # Transform features
    if fit_preprocessor:
        transformed = preprocessor.fit_transform(df)
    else:
        transformed = preprocessor.transform(df)
    
    # Convert to DataFrame if needed
    if isinstance(transformed, pd.DataFrame):
        result_df = transformed.copy()
    else:
        result_df = pd.DataFrame(transformed)
    
    return result_df


def run_pipeline(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    boolean_cols: list[str],
    max_categories: int | None = None,
    xgboost_params: dict[str, Any] | None = None,
    impute_strategy: str = 'median',
    pred_col: str = PRED_YHAT_COL,
    pred_lower_col: str = PRED_YHAT_LOWER_COL,
    pred_upper_col: str = PRED_YHAT_UPPER_COL,
) -> tuple[pd.DataFrame, XGBoostRegressorWrapper]:
    """
    Run the XGBoost pipeline.
    """
    # Create pipeline
    model, preprocessor = get_pipeline(
        cat_cols,
        num_cols,
        boolean_cols,
        max_categories,
        xgboost_params,
        impute_strategy,
    )

    # Extract target before preprocessing (target is not in cat_cols/num_cols)
    y_train: pd.Series = train_df[target_col]  # type: ignore[assignment]
    
    # Prepare training data
    train_prepared = prepare_data(
        train_df,
        preprocessor,
        fit_preprocessor=True,
    )
    
    # Use the preprocessed features as X_train
    X_train = train_prepared

    # Fit the model
    logger.info(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Prepare prediction data
    logger.info("Preparing prediction data...")
    predict_prepared = prepare_data(
        df=predict_df,
        preprocessor=preprocessor,
        fit_preprocessor=False,
    )

    # Use the preprocessed features for prediction
    X_predict = predict_prepared
    predictions = model.predict_with_intervals(
        X_predict,
        pred_col=pred_col,
        pred_lower_col=pred_lower_col,
        pred_upper_col=pred_upper_col,
    ) 

    result_df = predict_df.reset_index(drop=True).copy()
    predictions_aligned = predictions.reset_index(drop=True)

    # Add only the prediction columns (no modifications to existing columns)
    result_df[pred_col] = predictions_aligned[PRED_YHAT_COL].values
    result_df[pred_lower_col] = predictions_aligned[PRED_YHAT_LOWER_COL].values
    result_df[pred_upper_col] = predictions_aligned[PRED_YHAT_UPPER_COL].values

    logger.info(f"Result dataframe now has {len(result_df.columns)} columns (added 3 prediction columns)")
    return result_df, model