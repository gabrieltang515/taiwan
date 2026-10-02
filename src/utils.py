"""
Utility helpers used across multiple modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_table(df: pd.DataFrame, path: Path, fmt: str = "csv") -> None:
    """Save a DataFrame as CSV (default) or LaTeX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "latex":
        path.write_text(df.to_latex(float_format="%.4f"))
    else:
        raise ValueError(f"Unknown format {fmt}")
    print(f"  ✓ saved {path.name}")


def pct_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with count and % of missing values per column."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    return (
        pd.DataFrame({"n_missing": missing, "pct_missing": pct})
        .sort_values("pct_missing", ascending=False)
    )


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int] = [1]) -> pd.DataFrame:
    """Add lagged columns. E.g. col_lag1 = col shifted by 1 period."""
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def time_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time index (no shuffle). Assumes df is sorted by date."""
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that get the sign correct."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return correct.mean()
