"""
Simple regime analysis.

Two regime definitions:
  1. VIX-based:  risk-on (VIX < 20) vs risk-off (VIX ≥ 20)
  2. DXY-based:  weak USD (DXY below 12m MA) vs strong USD

For each regime split we:
  - Compare average target returns / changes.
  - Compare correlations of targets with key features.
  - Optionally re-run the core OLS in each sub-sample.

Kept deliberately lightweight for a student presentation.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.config import TARGETS, CORE_FEATURES, OUT_TABLES
from src.utils import save_table


def _regime_stats(df: pd.DataFrame, regime_col: str, target_col: str) -> pd.DataFrame:
    """Mean, std, count, Sharpe-like ratio by regime."""
    grouped = df.groupby(regime_col)[target_col]
    stats = grouped.agg(["mean", "std", "count"])
    stats["sharpe_proxy"] = stats["mean"] / stats["std"]
    stats.index.name = "regime"
    return stats.round(4)


def _regime_correlations(
    df: pd.DataFrame, regime_col: str, target_col: str, features: list[str]
) -> pd.DataFrame:
    """Pairwise correlation of target with features, split by regime."""
    avail = [f for f in features if f in df.columns and f != target_col]
    rows = []
    for regime_val in sorted(df[regime_col].dropna().unique()):
        sub = df[df[regime_col] == regime_val]
        for feat in avail:
            pair = sub[[target_col, feat]].dropna()
            if len(pair) > 5:
                corr = pair[target_col].corr(pair[feat])
            else:
                corr = np.nan
            rows.append({
                "regime": regime_val,
                "feature": feat,
                "correlation": round(corr, 4) if not np.isnan(corr) else np.nan,
            })
    return pd.DataFrame(rows)


def _regime_ols(
    df: pd.DataFrame, regime_col: str, target_col: str, features: list[str]
) -> pd.DataFrame:
    """Run the core OLS separately in each regime sub-sample."""
    avail = [f for f in features if f in df.columns]
    rows = []
    for regime_val in sorted(df[regime_col].dropna().unique()):
        sub = df[df[regime_col] == regime_val][[target_col] + avail].dropna()
        if len(sub) < len(avail) + 5:
            continue
        X = sm.add_constant(sub[avail])
        y = sub[target_col]
        res = sm.OLS(y, X).fit()
        for feat in avail:
            if feat in res.params.index:
                rows.append({
                    "regime": regime_val,
                    "feature": feat,
                    "coef": round(res.params[feat], 4),
                    "t_stat": round(res.tvalues[feat], 2),
                    "p_value": round(res.pvalues[feat], 4),
                })
        rows.append({
            "regime": regime_val,
            "feature": "_R²_adj",
            "coef": round(res.rsquared_adj, 4),
            "t_stat": np.nan,
            "p_value": np.nan,
        })
    return pd.DataFrame(rows)


def run_regime_analysis(df: pd.DataFrame) -> dict:
    """
    Run regime comparisons for both VIX and DXY regimes.

    Returns
    -------
    regime_results : dict with DataFrames for stats, correlations, and OLS
    """
    print("── Regime analysis ──")
    results = {}
    regime_cols = []

    if "regime_vix" in df.columns:
        regime_cols.append(("regime_vix", "VIX"))
    if "regime_dxy" in df.columns:
        regime_cols.append(("regime_dxy", "DXY"))

    if not regime_cols:
        print("  ⚠ no regime flags found – skipping")
        return results

    all_stats = []
    all_corrs = []
    all_ols = []

    for regime_col, regime_name in regime_cols:
        for target_label, target_col in TARGETS.items():
            if target_col not in df.columns:
                continue

            # Average return/change by regime
            stats = _regime_stats(df, regime_col, target_col)
            stats["target"] = target_label
            stats["regime_type"] = regime_name
            all_stats.append(stats.reset_index())

            # Correlations by regime
            corr = _regime_correlations(df, regime_col, target_col, CORE_FEATURES)
            corr["target"] = target_label
            corr["regime_type"] = regime_name
            all_corrs.append(corr)

            # OLS by regime
            ols = _regime_ols(df, regime_col, target_col, CORE_FEATURES)
            ols["target"] = target_label
            ols["regime_type"] = regime_name
            all_ols.append(ols)

    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        save_table(stats_df, OUT_TABLES / "regime_stats.csv")
        results["stats"] = stats_df
        print("  ✓ regime average stats saved")

    if all_corrs:
        corrs_df = pd.concat(all_corrs, ignore_index=True)
        save_table(corrs_df, OUT_TABLES / "regime_correlations.csv")
        results["correlations"] = corrs_df
        print("  ✓ regime correlations saved")

    if all_ols:
        ols_df = pd.concat(all_ols, ignore_index=True)
        save_table(ols_df, OUT_TABLES / "regime_ols.csv")
        results["ols"] = ols_df
        print("  ✓ regime OLS saved")

    return results


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_csv(DATA_PROC / "features_monthly.csv",
                     index_col=0, parse_dates=True)
    run_regime_analysis(df)
