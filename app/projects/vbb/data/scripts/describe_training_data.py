from src.initialize import load
load()

import logging
from pandas import DataFrame
import numpy as np

from projects.vbb.data.training import get_training_df
from projects.vbb import get_session

from projects.vbb.constants import (
    ALL_COLS, TRAINING_COLS, TARGET_COL,
    CAT_COLS, NUM_COLS, BOOLEAN_COLS,
)


logger = logging.getLogger(__name__)

SEPARATOR: str = "=" * 70
FORCE_REFRESH: bool = True


def _ensure_all_cols_in_df(df: DataFrame, all_cols: list[str]) -> None:
    for col in all_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")


def _describe_target(df: DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  DATASET OVERVIEW")
    print(SEPARATOR)
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Training features: {len(TRAINING_COLS)}")

    target = df[TARGET_COL]
    print(f"\n{SEPARATOR}")
    print(f"  TARGET: {TARGET_COL}")
    print(SEPARATOR)
    print(f"  Mean:    {target.mean():.2f}")
    print(f"  Median:  {target.median():.2f}")
    print(f"  Std:     {target.std():.2f}")
    print(f"  Min:     {target.min():.2f}")
    print(f"  Max:     {target.max():.2f}")
    print(f"  P25:     {target.quantile(0.25):.2f}")
    print(f"  P75:     {target.quantile(0.75):.2f}")
    print(f"  P90:     {target.quantile(0.90):.2f}")
    n = len(target)
    print(f"  Null:    {target.isnull().sum()} ({target.isnull().mean() * 100:.1f}%)")
    print(f"  Zero:    {(target == 0).sum()} ({(target == 0).sum() / n * 100:.1f}%)")
    print(f"  Negative:{(target < 0).sum()} ({(target < 0).sum() / n * 100:.1f}%)")


def _describe_cat_col(df: DataFrame, col: str) -> None:
    series = df[col]
    n = len(series)
    null_count = series.isnull().sum()
    unique = series.nunique(dropna=True)

    print(f"\n  {col}  (categorical)")
    print(f"    Unique: {unique}  |  Null: {null_count} ({null_count / n * 100:.1f}%)")
    top = series.value_counts(dropna=True).head(5)
    for val, count in top.items():
        print(f"    {str(val):<40s} {count:>8,}  ({count / n * 100:.1f}%)")


def _describe_num_col(df: DataFrame, col: str) -> None:
    series = df[col]
    n = len(series)
    null_count = series.isnull().sum()

    print(f"\n  {col}  (numerical)")
    print(f"    Null: {null_count} ({null_count / n * 100:.1f}%)")
    if series.dropna().empty:
        print("    (all null)")
        return
    print(f"    Mean:   {series.mean():.2f}  |  Std: {series.std():.2f}  |  Skew: {series.skew():.2f}")
    print(f"    Min:    {series.min():.2f}  |  Median: {series.median():.2f}  |  Max: {series.max():.2f}")
    print(f"    P25:    {series.quantile(0.25):.2f}  |  P75: {series.quantile(0.75):.2f}")


def _describe_bool_col(df: DataFrame, col: str) -> None:
    series = df[col]
    n = len(series)
    null_count = series.isnull().sum()
    true_count = series.sum()
    false_count = n - null_count - true_count

    print(f"\n  {col}  (boolean)")
    print(f"    True:  {int(true_count):>8,}  ({true_count / n * 100:.1f}%)")
    print(f"    False: {int(false_count):>8,}  ({false_count / n * 100:.1f}%)")
    print(f"    Null:  {null_count:>8,}  ({null_count / n * 100:.1f}%)")


def _describe_training_col(df: DataFrame, col: str) -> None:
    if col in CAT_COLS:
        _describe_cat_col(df, col)
    elif col in NUM_COLS:
        _describe_num_col(df, col)
    elif col in BOOLEAN_COLS:
        _describe_bool_col(df, col)


def _describe_training_cols(df: DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  CATEGORICAL FEATURES ({len(CAT_COLS)})")
    print(SEPARATOR)
    for col in CAT_COLS:
        _describe_training_col(df, col)

    print(f"\n{SEPARATOR}")
    print(f"  NUMERICAL FEATURES ({len(NUM_COLS)})")
    print(SEPARATOR)
    for col in NUM_COLS:
        _describe_training_col(df, col)

    print(f"\n{SEPARATOR}")
    print(f"  BOOLEAN FEATURES ({len(BOOLEAN_COLS)})")
    print(SEPARATOR)
    for col in BOOLEAN_COLS:
        _describe_training_col(df, col)


def _describe_missing(df: DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  MISSING DATA SUMMARY")
    print(SEPARATOR)
    n = len(df)
    null_counts = df[ALL_COLS].isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    if null_counts.empty:
        print("  No missing values found.")
        return

    print(f"  {'Column':<45s} {'Null':>8s}  {'%':>6s}")
    print(f"  {'-' * 45}  {'-' * 8}  {'-' * 6}")
    for col, count in null_counts.items():
        print(f"  {col:<45s} {count:>8,}  {count / n * 100:>5.1f}%")


def main():
    df = get_training_df(
        get_session(),
        FORCE_REFRESH
    )
    _ensure_all_cols_in_df(df, ALL_COLS)
    _describe_target(df)
    _describe_training_cols(df)
    _describe_missing(df)


if __name__ == "__main__":
    main()
