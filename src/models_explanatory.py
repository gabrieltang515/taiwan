"""
Explanatory (contemporaneous) models.

For each target (FX returns, equity returns, rate changes) we run:
  1. Baseline OLS with a small core feature set.
  2. Rich OLS adding Taiwan macro + global controls.
  3. Summarise coefficients, t-stats, R².

Uses statsmodels OLS for rich regression output.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    TARGETS, CORE_FEATURES, RICH_FEATURES, OUT_TABLES
)
from src.utils import save_table


def run_ols(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    label: str = "",
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """
    Run OLS with constant on the available subset of features.
    Returns the fitted result (or None if insufficient data).
    """
    avail = [f for f in features if f in df.columns]
    if not avail or target not in df.columns:
        print(f"  ⚠ skipping {label}: missing target or all features")
        return None

    sub = df[[target] + avail].dropna()
    if len(sub) < len(avail) + 5:
        print(f"  ⚠ skipping {label}: too few observations ({len(sub)})")
        return None

    X = sm.add_constant(sub[avail])
    y = sub[target]
    model = sm.OLS(y, X).fit()
    return model


def ols_summary_table(result, label: str) -> pd.DataFrame:
    """Extract a compact summary from a statsmodels OLS result."""
    if result is None:
        return pd.DataFrame()
    params = result.params.rename("coef")
    tvals = result.tvalues.rename("t_stat")
    pvals = result.pvalues.rename("p_value")
    tbl = pd.concat([params, tvals, pvals], axis=1)
    tbl["label"] = label
    tbl["r_squared"] = result.rsquared
    tbl["adj_r_squared"] = result.rsquared_adj
    tbl["n_obs"] = int(result.nobs)
    return tbl


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Variance Inflation Factors for multicollinearity inspection."""
    avail = [f for f in features if f in df.columns]
    sub = df[avail].dropna()
    if len(sub) < len(avail) + 2:
        return pd.DataFrame({"feature": avail, "VIF": [np.nan] * len(avail)})

    X = sm.add_constant(sub)
    vifs = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vifs.append({"feature": col, "VIF": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(vifs)


def run_all_explanatory(df: pd.DataFrame) -> dict:
    """
    Run core + rich OLS for each target.

    Returns
    -------
    results : dict  –  {target_label: {model_label: statsmodels result}}
    """
    print("── Explanatory models ──")
    all_summaries = []
    results = {}

    for target_label, target_col in TARGETS.items():
        results[target_label] = {}
        for feat_set, feat_label in [
            (CORE_FEATURES, "core"),
            (RICH_FEATURES, "rich"),
        ]:
            label = f"{target_label}_{feat_label}"
            res = run_ols(df, target_col, feat_set, label=label)
            results[target_label][feat_label] = res
            tbl = ols_summary_table(res, label)
            if not tbl.empty:
                all_summaries.append(tbl)
                r2 = res.rsquared_adj if res else np.nan
                print(f"  {label:25s}  adj-R²={r2:.3f}  n={int(res.nobs)}")

    if all_summaries:
        full = pd.concat(all_summaries)
        save_table(full, OUT_TABLES / "explanatory_regressions.csv")

    # VIF for rich feature set
    vif_df = compute_vif(df, RICH_FEATURES)
    if not vif_df.empty:
        save_table(vif_df, OUT_TABLES / "vif_rich_features.csv")
        print("  VIF table saved")

    return results


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_csv(DATA_PROC / "features_monthly.csv",
                     index_col=0, parse_dates=True)
    run_all_explanatory(df)
