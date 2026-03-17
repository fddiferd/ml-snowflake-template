from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from src.base_models.evaluation import EvaluationResult


def evaluate_model(y_true: pd.Series, y_pred: pd.Series):
    """Calculate the metrics for actual vs predicted"""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    spearman_result = spearmanr(y_true, y_pred)
    spearman_corr = float(spearman_result.statistic)  # type: ignore[union-attr]
    
    return EvaluationResult(
        rmse=rmse,
        mae=mae,
        mape=float(mape),
        r2=float(r2),
        spearman=spearman_corr,
    )
