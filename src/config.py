"""
Central configuration for the Taiwan Cross-Asset Macro Playbook project.
All paths, tickers, date ranges, and model parameters in one place.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUT_CHARTS   = PROJECT_ROOT / "outputs" / "charts"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"

for d in [DATA_RAW, DATA_PROC, OUT_CHARTS, OUT_TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# ── Date range ──────────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
END_DATE   = "2025-12-31"

# ── yfinance tickers ───────────────────────────────────────────────────────
TICKERS = {
    "usdtwd": "TWD=X",       # USD/TWD exchange rate
    "ewt":    "EWT",          # iShares MSCI Taiwan ETF (TAIEX proxy)
    "tsmc":   "TSM",          # TSMC ADR (optional)
    "dxy":    "DX-Y.NYB",     # US Dollar Index
    "nasdaq": "^IXIC",        # Nasdaq Composite
    "sox":    "^SOX",         # PHLX Semiconductor Index
    "vix":    "^VIX",         # CBOE Volatility Index
    "oil":    "CL=F",         # WTI Crude Oil futures
    "us10y":  "^TNX",         # US 10-Year Treasury yield
    "us2y":   "2YY=F",        # US 2-Year yield (futures proxy)
}

# ── FRED series ─────────────────────────────────────────────────────────────
# Taiwan-specific series available on FRED
FRED_SERIES = {
    "tw_cpi":           "TWNCPIALLMINMEI",   # Taiwan CPI
    "tw_ip":            "TWNIPMANSMINMEI",   # Taiwan industrial production (manufacturing)
    "tw_policy_rate":   "IRSTCI01TWM156N",   # Taiwan short-term interest rate
    "us_fed_funds":     "FEDFUNDS",          # US Fed Funds rate (control)
    "us10y_fred":       "GS10",              # US 10Y yield (backup)
    "us2y_fred":        "GS2",               # US 2Y yield (backup)
}

# ── Taiwan customs / trade data from FRED ──────────────────────────────────
FRED_TRADE = {
    "tw_exports":  "XTEXVA01TWM667S",  # Taiwan exports value
    "tw_imports":  "XTIMVA01TWM667S",  # Taiwan imports value
}

# ── Modeling ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.75  # 75 % train, 25 % test (time-ordered)

TARGETS = {
    "fx_ret":     "usdtwd_ret",
    "eq_ret":     "ewt_ret",
    "rate_chg":   "tw_rate_chg",
}

# Core (parsimonious) feature set for baseline explanatory regressions
CORE_FEATURES = [
    "dxy_ret",
    "nasdaq_ret",
    "vix_level",
    "us10y_level",
    "tw_exports_yoy",
]

# Rich feature set (adds Taiwan macro + more global controls)
RICH_FEATURES = CORE_FEATURES + [
    "sox_ret",
    "oil_ret",
    "us2y_level",
    "tw_cpi_yoy",
    "tw_ip_yoy",
    "tw_trade_balance",
    "tw_policy_rate_level",
]

# Features used for prediction (all lagged by 1 month)
PRED_FEATURES = [
    "dxy_ret_lag1",
    "nasdaq_ret_lag1",
    "sox_ret_lag1",
    "vix_level_lag1",
    "us10y_level_lag1",
    "oil_ret_lag1",
    "tw_exports_yoy_lag1",
    "tw_cpi_yoy_lag1",
    "tw_ip_yoy_lag1",
    "tw_trade_balance_lag1",
]

# ── Regime definitions ──────────────────────────────────────────────────────
VIX_THRESHOLD = 20       # VIX >= 20 → risk-off
DXY_MA_WINDOW = 12       # 12-month MA for DXY trend; above MA → strong USD
