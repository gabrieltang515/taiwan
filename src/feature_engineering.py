"""
Feature engineering module.

Creates lagged predictors, rolling windows, and standardised features
from the master monthly dataset.

Principles
----------
* Lagged features use shift(1) so that month-t prediction uses
  month-(t-1) data → no lookahead bias.
* Z-scoring is optional; mainly used for Ridge/Lasso regularisation
  (fitted on training set only, applied to test set).
* Rolling correlations use 12-month windows by default.
"""

import pandas as pd
import numpy as np

from src.config import (
    DATA_PROC, CORE_FEATURES, RICH_FEATURES, PRED_FEATURES
)
from src.utils import add_lags


def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1-month lags of the features used for prediction.

    The lag naming convention is  <feature>_lag1.
    """
    base_cols = [c.replace("_lag1", "") for c in PRED_FEATURES]
    available = [c for c in base_cols if c in df.columns]
    df = add_lags(df, available, lags=[1])
    return df


def create_rolling_correlations(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    window: int = 12,
) -> pd.DataFrame:
    """12-month rolling correlation of each feature with the target."""
    roll_corr = pd.DataFrame(index=df.index)
    for feat in features:
        if feat in df.columns and target in df.columns:
            roll_corr[f"rollcorr_{feat}"] = (
                df[target].rolling(window).corr(df[feat])
            )
    return roll_corr


def create_regime_flags(df: pd.DataFrame, vix_threshold: float = 20,
                        dxy_ma_window: int = 12) -> pd.DataFrame:
    """
    Add binary regime columns:
      * regime_vix : 1 = risk-off (VIX >= threshold), 0 = risk-on
      * regime_dxy : 1 = strong USD (DXY above its 12m MA), 0 = weak USD
    """
    df = df.copy()

    if "vix_level" in df.columns:
        df["regime_vix"] = (df["vix_level"] >= vix_threshold).astype(int)

    if "dxy_ret" in df.columns:
        # Use cumulative DXY level proxy via cumsum of returns
        dxy_cum = df["dxy_ret"].cumsum()
        ma = dxy_cum.rolling(dxy_ma_window).mean()
        df["regime_dxy"] = (dxy_cum > ma).astype(int)
    elif "dxy_level" in df.columns:
        ma = df["dxy_level"].rolling(dxy_ma_window).mean()
        df["regime_dxy"] = (df["dxy_level"] > ma).astype(int)

    return df


def engineer_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    1. Create lagged predictors.
    2. Create regime flags.
    3. Save the analysis-ready dataset.
    """
    print("── Feature engineering ──")

    df = master.copy()

    # Lagged features for prediction
    df = create_lagged_features(df)

    # Regime flags
    df = create_regime_flags(df)

    # Save
    out = DATA_PROC / "features_monthly.csv"
    df.to_csv(out)
    print(f"  ✓ saved {out.name}  shape={df.shape}")

    # Report available feature coverage
    available_core = [c for c in CORE_FEATURES if c in df.columns]
    available_rich = [c for c in RICH_FEATURES if c in df.columns]
    available_pred = [c for c in PRED_FEATURES if c in df.columns]
    print(f"  Core features available: {len(available_core)}/{len(CORE_FEATURES)}")
    print(f"  Rich features available: {len(available_rich)}/{len(RICH_FEATURES)}")
    print(f"  Pred features available: {len(available_pred)}/{len(PRED_FEATURES)}")

    return df


if __name__ == "__main__":
    master = pd.read_csv(DATA_PROC / "master_monthly.csv",
                         index_col=0, parse_dates=True)
    df = engineer_features(master)
    print(df.tail())
