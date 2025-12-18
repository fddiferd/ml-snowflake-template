from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ConfigDict
import numpy as np


logger = logging.getLogger(__name__)


class RegressionModelEvalOutput(BaseModel):
    """
    Definitions:
        - rmse: Root Mean Squared Error
            - Range: [0, ∞)
            - Lower is better
            - Penalizes large errors more than small ones (sensitive to outliers)
            - Good when large mistakes are especially costly

        - mae: Mean Absolute Error
            - Range: [0, ∞)
            - Lower is better
            - Treats all errors equally (linear penalty)
            - More robust to outliers than RMSE

        - mape: Mean Absolute Percentage Error
            - Range: [0, ∞)
            - Lower is better
            - Expresses error as a percentage of actual values
            - Can be misleading when actual values are near zero

        - r2: R-squared (Coefficient of Determination)
            - Range: (-∞, 1]
            - Higher is better
            - Measures proportion of variance in the target explained by the model
            - Negative values mean the model is worse than predicting the mean
    """
    rmse: float
    mae: float
    mape: float
    r2: float

class RegressionModelEvalOverfittingAnalysis(BaseModel):
    rmse_degradation: float
    r2_degradation: float

class RegressionTestTrainModelEvalOutput(BaseModel):
    train_metrics: RegressionModelEvalOutput
    test_metrics: RegressionModelEvalOutput
    overfitting_analysis: RegressionModelEvalOverfittingAnalysis

class XGBoostParamSearchOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    best_params: Dict[str, Any]
    best_score: float
    best_estimator: Pipeline
    cv_results: pd.DataFrame
    test_metrics: Optional[RegressionTestTrainModelEvalOutput]
    

def evaluate_model(y_true: pd.Series, y_pred: pd.Series):
    """Calculate the metrics for actual vs predicted"""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero and handle edge case of all zeros
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan  # All true values are zero
    
    return RegressionModelEvalOutput(
        rmse=rmse,
        mae=mae,
        mape=float(mape),
        r2=float(r2)
    )
    
def evaluate_test_train_models(
    y_train: pd.Series,
    y_train_pred: pd.Series,
    y_test: pd.Series,
    y_test_pred: pd.Series,
):
    
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)

    return RegressionTestTrainModelEvalOutput(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        overfitting_analysis=RegressionModelEvalOverfittingAnalysis(
            rmse_degradation=(test_metrics.rmse - train_metrics.rmse) / train_metrics.rmse * 100,
            r2_degradation=(train_metrics.r2 - test_metrics.r2) / train_metrics.r2 * 100,
        )
    )


# MARK: - Feature Importances
def get_feature_importances(pipe) -> pd.DataFrame:
    logger.info("Getting feature importances")
    # Get the fitted XGBoost model from the pipeline
    xgb_model = pipe.named_steps['xgboost']

    # Get feature names - handle feature selection if present
    if 'feature_selection' in pipe.named_steps:
        # Get original feature names from preprocessor
        all_feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        # Get selected feature mask from feature selector
        feature_selector = pipe.named_steps['feature_selection']
        selected_mask = feature_selector.get_support()
        # Get only the selected feature names
        feature_names = all_feature_names[selected_mask]
    else:
        # No feature selection, use all features
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance_df

# MARK: - Hyper Tuning
def xgb_param_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_distributions: Optional[Dict[str, List[Any]]] = None,
    n_iter: int = 25,
    cv: int = 3,
    scoring: str = 'neg_root_mean_squared_error',
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1
):
    if param_distributions is None:
        # Focus on a compact search that emphasizes regularization and shallow trees
        param_distributions = {
            'xgboost__n_estimators': [200, 300, 500, 700],
            'xgboost__learning_rate': [0.05, 0.075, 0.1, 0.15],
            'xgboost__max_depth': [3, 4, 5, 6],
            'xgboost__min_child_weight': [1, 2, 3, 5],
            'xgboost__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgboost__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgboost__reg_alpha': [0.0, 0.001, 0.01, 0.1, 1.0],
            'xgboost__reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
            'xgboost__gamma': [0, 0.1, 0.5, 1.0],
            
            # 'xgboost__n_estimators': [50, 100, 200],  # Reduce max estimators
            # 'xgboost__learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
            # 'xgboost__max_depth': [2, 3, 4],  # Shallower trees
            # 'xgboost__min_child_weight': [3, 5, 10],  # Higher min_child_weight
            # 'xgboost__subsample': [0.5, 0.6, 0.7],  # More aggressive subsampling
            # 'xgboost__colsample_bytree': [0.5, 0.6, 0.7],  # More feature subsampling
            # 'xgboost__reg_alpha': [0.1, 1.0, 5.0, 10.0],  # Stronger L1 regularization
            # 'xgboost__reg_lambda': [1.0, 5.0, 10.0, 20.0],  # Stronger L2 regularization
            # 'xgboost__gamma': [0.1, 0.5, 1.0, 2.0],  # Higher gamma for pruning
        }

    logger.info("Starting XGBoost hyperparameter search (RandomizedSearchCV)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    search.fit(X_train, y_train)
    
    y_train_pred = search.predict(X_train)
    y_test_pred = search.predict(X_test)
    # Prepare CV results
    results_df = pd.DataFrame(search.cv_results_).sort_values(
        by='rank_test_score', ascending=True
    )
    best_params = search.best_params_
    best_score = search.best_score_
    best_estimator = search.best_estimator_

    logger.info("Best CV params for XGBoost:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Best CV score ({scoring}): {best_score:.6f}")

    # Optional holdout evaluation to check generalization

    if X_test is not None and y_test is not None:
        y_train_pred = best_estimator.predict(X_train)
        y_test_pred = best_estimator.predict(X_test)
    
    return XGBoostParamSearchOutput(
        best_params=best_params,
        best_score=best_score,
        best_estimator=best_estimator,
        cv_results=results_df,
        test_metrics=evaluate_test_train_models(y_train, y_train_pred, y_test, y_test_pred),
    )

def get_pipeline(
    ohe_cols: list[str],
    num_cols: list[str],
    boolean_cols: list[str],
    top_n_features: int | None = None,
    max_categories: int = 30,
    xgboost_params: dict[str, Any] | None = None,
):
    """
    Create a machine learning pipeline with preprocessing and XGBoost.
    
    Args:
        ohe_cols: Columns to apply one-hot encoding
        num_cols: Numerical columns to scale and apply polynomial features
        boolean_cols: Boolean columns to convert to integers
        top_n_features: Number of top features to select (None for no feature selection)
        max_categories: Maximum categories for one-hot encoding (None for no limit)
        missing_value_threshold: Columns with missing values above this threshold (0-1) will be dropped
        xgboost_params: XGBoost parameters (None for default parameters)
        
    Returns:
        Configured sklearn Pipeline
    """    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(
                        handle_unknown='ignore', 
                        sparse_output=False,
                        max_categories=max_categories
                    )
                )
            ]), ohe_cols),
            ('bool', Pipeline([
                ('encoder', OrdinalEncoder()),
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ]), boolean_cols),
        ],
        remainder='drop',
    )

    preprocessor.set_output(transform="pandas")

    if xgboost_params is None:
        xgboost_params = {
            'n_estimators': 1000,  # Set high, early stopping will control
            'learning_rate': 0.05,  # Reduced from 0.1
            'max_depth': 3,  # Reduced from 5
            'min_child_weight': 5,  # Added regularization
            'subsample': 0.7,  # Added subsampling
            'colsample_bytree': 0.8,  # Added feature subsampling
            'reg_alpha': 1.0,  # L1 regularization
            'reg_lambda': 5.0,  # L2 regularization
            'gamma': 0.1,  # Minimum loss reduction for split
            'random_state': 42,
            'n_jobs': -1
        }
    # XGBoost pipeline
    steps = [
        ('preprocessor', preprocessor),
    ]
    if top_n_features is not None:
        steps.append(('feature_selection', SelectKBest(f_regression, k=top_n_features))) # type: ignore
        
    steps.append(('xgboost', XGBRegressor(**xgboost_params))) # type: ignore

    pipeline = Pipeline(steps)

    logger.info(f'Pipeline created for {ohe_cols} + {num_cols} + {boolean_cols}')
    return pipeline