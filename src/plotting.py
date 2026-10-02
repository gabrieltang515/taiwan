"""
Plotting module.

All charts are saved to outputs/charts/ as PNG files.
Functions return the matplotlib Figure so the notebook can display inline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import OUT_CHARTS, TARGETS, CORE_FEATURES

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")

FIGSIZE = (12, 5)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)


def _save(fig, name: str) -> None:
    path = OUT_CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ saved {path.name}")


# ── Time series plots ───────────────────────────────────────────────────────

def plot_timeseries(df: pd.DataFrame, col: str, title: str,
                    ylabel: str = "", name: str | None = None) -> plt.Figure:
    """Simple time-series line plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df.index, df[col], linewidth=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    if name:
        _save(fig, name)
    return fig


def plot_multi_ts(df: pd.DataFrame, cols: list[str], title: str,
                  ylabel: str = "", name: str | None = None) -> plt.Figure:
    """Overlay multiple time series."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for col in cols:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.1)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    if name:
        _save(fig, name)
    return fig


# ── Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None,
                             title: str = "Correlation Matrix",
                             name: str | None = None) -> plt.Figure:
    """Heatmap of pairwise correlations."""
    if cols:
        avail = [c for c in cols if c in df.columns]
    else:
        avail = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    if name:
        _save(fig, name)
    return fig


# ── Rolling correlation ─────────────────────────────────────────────────────

def plot_rolling_corr(df: pd.DataFrame, target: str, features: list[str],
                      window: int = 12, name: str | None = None) -> plt.Figure:
    """Rolling correlation of target with each feature."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for feat in features:
        if feat in df.columns and target in df.columns:
            rc = df[target].rolling(window).corr(df[feat])
            ax.plot(df.index, rc, label=feat, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Month Correlation with {target}", fontsize=13)
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.set_ylim(-1, 1)
    fig.autofmt_xdate()
    if name:
        _save(fig, name)
    return fig


# ── Actual vs Predicted ─────────────────────────────────────────────────────

def plot_actual_vs_predicted(preds: dict, target_label: str,
                             name: str | None = None) -> plt.Figure:
    """Plot actual vs predicted for the best model."""
    actual = preds.get("actual")
    if actual is None:
        return plt.figure()

    models = [k for k in preds if k != "actual"]
    n = len(models)
    fig, axes = plt.subplots(1, min(n, 3), figsize=(5 * min(n, 3), 4),
                             squeeze=False)
    for i, model in enumerate(models[:3]):
        ax = axes[0, i]
        ax.plot(actual.index, actual.values, label="Actual", linewidth=1.2)
        ax.plot(actual.index, preds[model].values, label=model,
                linewidth=1.0, linestyle="--")
        ax.set_title(f"{target_label} – {model}", fontsize=11)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Actual vs Predicted: {target_label}", fontsize=13, y=1.02)
    fig.tight_layout()
    if name:
        _save(fig, name)
    return fig


# ── Coefficient bar plot ────────────────────────────────────────────────────

def plot_coefficients(result, title: str = "OLS Coefficients",
                      name: str | None = None) -> plt.Figure:
    """Horizontal bar chart of regression coefficients (excl. constant)."""
    if result is None:
        return plt.figure()
    coefs = result.params.drop("const", errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(coefs) * 0.4)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Coefficient")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    if name:
        _save(fig, name)
    return fig


# ── T-statistic significance plot ───────────────────────────────────────────

def plot_significance(reg_table: pd.DataFrame, 
                      targets: list[str] = ["fx_ret_rich", "eq_ret_rich"],
                      name: str | None = None) -> plt.Figure:
    """
    Bar chart of t-statistics showing which factors are statistically significant.
    
    Bars crossing ±1.96 (dashed lines) are significant at 95% confidence.
    Significant factors are shown in bold colors, others are faded.
    """
    # Filter to selected targets, exclude constant
    df = reg_table[
        (reg_table["label"].isin(targets)) & 
        (reg_table.index != "const")
    ].copy()
    
    if df.empty:
        return plt.figure()
    
    # Pivot for side-by-side comparison
    pivot_t = df.pivot_table(index=df.index, columns="label", values="t_stat")
    pivot_p = df.pivot_table(index=df.index, columns="label", values="p_value")
    
    # Sort by average absolute t-stat
    order = pivot_t.abs().mean(axis=1).sort_values(ascending=True).index
    pivot_t = pivot_t.loc[order]
    pivot_p = pivot_p.loc[order]
    
    # Create figure
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 8), sharey=True)
    if n_targets == 1:
        axes = [axes]
    
    # Nice labels for targets
    target_labels = {
        "fx_ret_rich": "FX (USD/TWD)",
        "eq_ret_rich": "Equities (EWT)", 
        "rate_chg_rich": "Policy Rate"
    }
    
    for i, target in enumerate(targets):
        ax = axes[i]
        if target not in pivot_t.columns:
            continue
            
        t_vals = pivot_t[target]
        p_vals = pivot_p[target]
        
        # Color by significance: significant = bold, not significant = faded
        colors = []
        for idx in t_vals.index:
            p = p_vals.loc[idx]
            t = t_vals.loc[idx]
            if p < 0.05:  # Significant
                colors.append("#2ca02c" if t > 0 else "#d62728")  # Green/Red
            else:  # Not significant
                colors.append("#c7e9c0" if t > 0 else "#fcbba1")  # Faded green/red
        
        # Plot bars
        bars = ax.barh(t_vals.index, t_vals.values, color=colors, edgecolor="white", height=0.7)
        
        # Add significance threshold lines
        ax.axvline(1.96, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="p=0.05")
        ax.axvline(-1.96, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        
        # Add p-value annotations for significant factors
        for idx, (t, p) in enumerate(zip(t_vals.values, p_vals.values)):
            if p < 0.05:
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                x_pos = t + 0.3 if t > 0 else t - 0.3
                ha = "left" if t > 0 else "right"
                ax.annotate(stars, (x_pos, idx), fontsize=12, fontweight="bold", 
                           ha=ha, va="center")
        
        # Formatting
        label = target_labels.get(target, target)
        ax.set_title(f"{label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("t-statistic", fontsize=11)
        ax.set_xlim(-6, 8)
        
        # Add legend only to first plot
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)
    
    # Add significance guide
    fig.text(0.5, 0.02, 
             "Significance: *** p<0.001  ** p<0.01  * p<0.05  |  Dashed lines = 95% significance threshold (|t| > 1.96)",
             ha="center", fontsize=10, style="italic")
    
    fig.suptitle("Factor Significance: Which Variables Drive Taiwan Assets?", 
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if name:
        _save(fig, name)
    return fig


# ── Regime comparison bar chart ─────────────────────────────────────────────

def plot_regime_comparison(stats_df: pd.DataFrame,
                           name: str | None = None) -> plt.Figure:
    """Grouped bar chart of mean returns by regime and target."""
    if stats_df.empty:
        return plt.figure()

    regime_types = stats_df["regime_type"].unique()
    n = len(regime_types)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, rtype in enumerate(regime_types):
        ax = axes[0, i]
        sub = stats_df[stats_df["regime_type"] == rtype]
        pivot = sub.pivot_table(index="target", columns="regime", values="mean")
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title(f"Mean Return/Change by {rtype} Regime", fontsize=11)
        ax.set_ylabel("Mean (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Regime")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    if name:
        _save(fig, name)
    return fig


# ── Master plotting function (called from notebook) ────────────────────────

def generate_all_charts(df: pd.DataFrame, explanatory_results: dict,
                        all_preds: dict, regime_results: dict) -> None:
    """Generate and save all charts for the project."""
    print("── Generating charts ──")

    # Time series: key levels / raw data
    if "tw_exports_yoy" in df.columns:
        plot_timeseries(df, "tw_exports_yoy",
                        "Taiwan Exports YoY Growth (%)",
                        ylabel="%", name="tw_exports_yoy")

    if "usdtwd_ret" in df.columns:
        plot_timeseries(df, "usdtwd_ret",
                        "USD/TWD Monthly Return (%)",
                        ylabel="%", name="usdtwd_ret")

    if "ewt_ret" in df.columns:
        plot_timeseries(df, "ewt_ret",
                        "Taiwan Equity (EWT) Monthly Return (%)",
                        ylabel="%", name="ewt_ret")

    if "tw_rate_chg" in df.columns:
        plot_timeseries(df, "tw_rate_chg",
                        "Taiwan Policy Rate MoM Change (bps)",
                        ylabel="bps", name="tw_rate_chg")

    if "tw_trade_balance" in df.columns:
        plot_timeseries(df, "tw_trade_balance",
                        "Taiwan Trade Balance",
                        ylabel="USD", name="tw_trade_balance")

    # Correlation heatmap
    target_cols = [v for v in TARGETS.values() if v in df.columns]
    heatmap_cols = target_cols + [c for c in CORE_FEATURES if c in df.columns]
    if len(heatmap_cols) > 2:
        plot_correlation_heatmap(df, heatmap_cols,
                                 "Key Variable Correlations",
                                 name="correlation_heatmap")

    # Rolling correlations for FX
    if "usdtwd_ret" in df.columns:
        feats = [f for f in CORE_FEATURES if f in df.columns]
        if feats:
            plot_rolling_corr(df, "usdtwd_ret", feats,
                              name="rolling_corr_fx")

    # Coefficient plots for rich OLS
    for target_label, models in explanatory_results.items():
        res = models.get("rich")
        if res is not None:
            plot_coefficients(res, f"OLS Coefficients – {target_label} (rich)",
                              name=f"coef_{target_label}_rich")

    # Actual vs predicted
    for target_label, preds in all_preds.items():
        if preds:
            plot_actual_vs_predicted(preds, target_label,
                                     name=f"pred_{target_label}")

    # Regime comparison
    if "stats" in regime_results:
        plot_regime_comparison(regime_results["stats"],
                               name="regime_comparison")

    print("  ✓ all charts generated")
