"""
Data cleaning and merging module.

Takes raw daily market data + monthly FRED macro data and produces a single
month-end aligned master DataFrame saved to data/processed/.

Key decisions
-------------
* Market prices are resampled to month-end (last available observation).
* Monthly returns are computed as simple % changes of month-end prices.
* Macro series that are already monthly are aligned to month-end dates.
* Forward-fill is used for small gaps; remaining NaN rows are documented
  but NOT dropped automatically (models handle missing data per target).
"""

import pandas as pd
import numpy as np

from src.config import DATA_PROC


def resample_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily market data to month-end (last observation)."""
    monthly = daily.resample("ME").last()
    return monthly


def compute_returns(prices: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Simple monthly % returns for given price columns."""
    if cols is None:
        cols = prices.columns.tolist()
    rets = prices[cols].pct_change() * 100  # in percentage points
    rets.columns = [f"{c}_ret" for c in cols]
    return rets


def compute_yoy(series: pd.Series, name: str) -> pd.Series:
    """Year-over-year % change (12-month lag)."""
    return series.pct_change(12).mul(100).rename(f"{name}_yoy")


def align_and_merge(
    mkt_monthly: pd.DataFrame,
    macro_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge market and macro data on month-end index.

    * Market data is already month-end resampled.
    * FRED macro dates are shifted to month-end to align.
    """
    macro = macro_monthly.copy()
    macro.index = macro.index + pd.offsets.MonthEnd(0)

    # Drop duplicate month-ends (e.g. if FRED has multiple obs per month)
    macro = macro[~macro.index.duplicated(keep="last")]
    mkt_monthly = mkt_monthly[~mkt_monthly.index.duplicated(keep="last")]

    merged = mkt_monthly.join(macro, how="outer")
    merged.sort_index(inplace=True)
    return merged


def build_master(
    market_daily: pd.DataFrame,
    macro_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full cleaning pipeline → master monthly DataFrame.

    Steps
    -----
    1. Resample market prices to month-end.
    2. Compute monthly returns for market series.
    3. Compute YoY changes for key macro series.
    4. Compute trade balance from exports − imports.
    5. Merge everything.
    6. Forward-fill up to 3 months for small gaps.
    7. Save to data/processed/master_monthly.csv.
    """
    print("── Cleaning & merging ──")

    # 1) Month-end prices
    mkt_m = resample_to_monthly(market_daily)

    # 2) Returns for tradeable series
    ret_cols = ["usdtwd", "ewt", "tsmc", "dxy", "nasdaq", "sox", "oil"]
    ret_cols = [c for c in ret_cols if c in mkt_m.columns]
    rets = compute_returns(mkt_m, ret_cols)

    # Keep VIX level (not return), and yield levels
    levels = pd.DataFrame()
    if "vix" in mkt_m.columns:
        levels["vix_level"] = mkt_m["vix"]
    if "us10y" in mkt_m.columns:
        levels["us10y_level"] = mkt_m["us10y"]
    if "us2y" in mkt_m.columns:
        levels["us2y_level"] = mkt_m["us2y"]

    mkt_features = pd.concat([rets, levels], axis=1)

    # 3) Macro YoY changes
    macro_feats = pd.DataFrame(index=macro_monthly.index)

    yoy_map = {
        "tw_cpi":     "tw_cpi",
        "tw_ip":      "tw_ip",
        "tw_exports": "tw_exports",
        "tw_imports": "tw_imports",
    }
    for col, label in yoy_map.items():
        if col in macro_monthly.columns:
            macro_feats[f"{label}_yoy"] = compute_yoy(macro_monthly[col], label)

    # 4) Trade balance (level, in same units as exports/imports)
    if "tw_exports" in macro_monthly.columns and "tw_imports" in macro_monthly.columns:
        macro_feats["tw_trade_balance"] = (
            macro_monthly["tw_exports"] - macro_monthly["tw_imports"]
        )

    # Policy rate level
    if "tw_policy_rate" in macro_monthly.columns:
        macro_feats["tw_policy_rate_level"] = macro_monthly["tw_policy_rate"]

    # Rate change (month-over-month in bps)
    if "tw_policy_rate" in macro_monthly.columns:
        macro_feats["tw_rate_chg"] = macro_monthly["tw_policy_rate"].diff() * 100

    # 5) Merge
    master = align_and_merge(mkt_features, macro_feats)

    # Also use FRED yield backups if yfinance yields are mostly missing
    if "us10y_level" not in master.columns or master["us10y_level"].isna().mean() > 0.5:
        if "us10y_fred" in macro_monthly.columns:
            macro_monthly.index = macro_monthly.index + pd.offsets.MonthEnd(0)
            backup = macro_monthly["us10y_fred"]
            backup = backup[~backup.index.duplicated(keep="last")]
            master["us10y_level"] = master["us10y_level"].fillna(backup)

    if "us2y_level" not in master.columns or master.get("us2y_level", pd.Series()).isna().mean() > 0.5:
        if "us2y_fred" in macro_monthly.columns:
            backup = macro_monthly["us2y_fred"]
            backup = backup[~backup.index.duplicated(keep="last")]
            if "us2y_level" in master.columns:
                master["us2y_level"] = master["us2y_level"].fillna(backup)
            else:
                master["us2y_level"] = backup.reindex(master.index)

    # 6) Forward-fill small gaps (max 3 months)
    master = master.ffill(limit=3)

    # 7) Save
    out_path = DATA_PROC / "master_monthly.csv"
    master.to_csv(out_path)
    print(f"  ✓ saved {out_path.name}  shape={master.shape}")
    print(f"  Date range: {master.index.min()} → {master.index.max()}")

    return master


if __name__ == "__main__":
    from src.data_ingest import ingest_all
    mkt, macro = ingest_all()
    master = build_master(mkt, macro)
    print(master.tail())
