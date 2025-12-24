from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


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
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
