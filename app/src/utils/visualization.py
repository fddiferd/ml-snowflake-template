import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import r2_score


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series) -> Figure:
    """Scatter plot of actual vs predicted with diagonal reference line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    r2 = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.1, s=4, color='steelblue')

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.92), xycoords='axes fraction', fontsize=12)

    fig.tight_layout()
    return fig


def plot_residual_distribution(y_true: pd.Series, y_pred: pd.Series) -> Figure:
    """Histogram of residuals (actual - predicted)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=80, color='steelblue', edgecolor='white', alpha=0.8)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))
    ax.axvline(mean_r, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.annotate(f'Mean = {mean_r:.2f}\nStd = {std_r:.2f}',
                xy=(0.75, 0.85), xycoords='axes fraction', fontsize=11)

    fig.tight_layout()
    return fig


def plot_feature_importances(importance_df: pd.DataFrame, top_n: int = 20) -> Figure:
    """Horizontal bar chart of top-N feature importances.

    Expects a DataFrame with 'feature' and 'importance' columns,
    as returned by XGBoostRegressorWrapper.get_feature_importance().
    """
    df = importance_df.head(top_n).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))

    ax.barh(df['feature'], df['importance'], color='steelblue')
    ax.set_xlabel('Importance (gain)')
    ax.set_title(f'Top {top_n} Feature Importances')

    fig.tight_layout()
    return fig


def plot_decile_lift(y_true: pd.Series, y_pred: pd.Series) -> Figure:
    """Bar chart of mean actual value per predicted-value decile.

    A monotonically increasing chart means the model ranks customers
    correctly, which is the primary requirement for Value Based Bidding.
    """
    df = pd.DataFrame({'actual': np.asarray(y_true), 'predicted': np.asarray(y_pred)})
    decile_labels = pd.qcut(df['predicted'], 10, labels=False, duplicates='drop')
    df['decile'] = np.asarray(decile_labels) + 1
    summary = df.groupby('decile')['actual'].mean()
    x_vals = np.asarray(summary.index)
    y_vals = np.asarray(summary.values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_vals, y_vals, color='steelblue', edgecolor='white')

    for i, val in zip(x_vals, y_vals):
        ax.text(i, val + y_vals.max() * 0.01, f'${val:.0f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Predicted Value Decile (1 = lowest)')
    ax.set_ylabel('Mean Actual NET_BILLINGS ($)')
    ax.set_title('Decile Lift Chart -- VBB Rank Quality')
    ax.set_xticks(range(1, len(x_vals) + 1))

    fig.tight_layout()
    return fig
