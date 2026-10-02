# Taiwan Cross-Asset Macro Playbook

**Research question:** Which macro, external, and global variables best explain and predict Taiwan's FX (USD/TWD), equities, and rates?

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline from the notebook
cd notebooks
jupyter notebook main_analysis.ipynb
```

## Project Structure

```
taiwan/
├── data/
│   ├── raw/              # Cached downloads (yfinance, FRED)
│   └── processed/        # Cleaned master dataset
├── notebooks/
│   └── main_analysis.ipynb   # Main workflow notebook
├── src/
│   ├── config.py             # Paths, tickers, parameters
│   ├── data_ingest.py        # Download market + macro data
│   ├── data_cleaning.py      # Resample, compute returns, merge
│   ├── feature_engineering.py # Lags, regimes, rolling features
│   ├── models_explanatory.py  # OLS regressions (core + rich)
│   ├── models_predictive.py   # Ridge, Lasso, Logistic
│   ├── regime_analysis.py     # VIX / DXY regime splits
│   ├── plotting.py            # All chart generation
│   └── utils.py               # Helpers (save, split, metrics)
├── outputs/
│   ├── charts/           # PNG charts
│   └── tables/           # CSV result tables
├── requirements.txt
└── README.md
```

## Workflow

The notebook runs these steps in order:

1. **Data ingestion** – downloads market prices (yfinance), US rates (FRED), and Taiwan macro data (DGBAS/MOEA/Customs/CBC). Cached to `data/raw/`.
2. **Cleaning** – resamples to monthly, computes returns and YoY changes, merges into a master dataset.
3. **Feature engineering** – adds 1-month lagged predictors and binary regime flags.
4. **Exploratory analysis** – summary stats, missing data report, correlation heatmap, time series plots, rolling correlations.
5. **Explanatory models** – core and rich OLS regressions for each target, VIF check for multicollinearity.
6. **Predictive models** – OLS, Ridge, Lasso, and Logistic (directional) with time-ordered train/test split. Benchmarked against a naïve (no-change) forecast.
7. **Regime analysis** – VIX risk-on/off and DXY strong/weak USD regime splits with average return comparison, correlation comparison, and sub-sample OLS.
8. **Robustness** – Taiwan-only vs global-only feature set R² comparison.

## Data Sources

| Series | Source | Notes |
|--------|--------|-------|
| USD/TWD | yfinance (`TWD=X`) | Daily, resampled to month-end |
| Taiwan equity | yfinance (`EWT`) | iShares MSCI Taiwan ETF as TAIEX proxy |
| DXY, Nasdaq, SOX, VIX, Oil | yfinance | Global market controls |
| US 10Y / 2Y yields | yfinance + FRED backup | `^TNX`, `GS10`, `GS2` |
| US Fed Funds | FRED (`FEDFUNDS`) | Monthly |
| Taiwan CPI | DGBAS XML open data | Headline CPI index (base 2021=100), monthly since 1981 |
| Taiwan industrial production | MOEA CSV open data | Overall industry production index (base 2021=100), monthly since 1996 |
| Taiwan exports / imports | Taiwan Customs CSV open data | NTD thousands, monthly (starts 2014 in current dataset) |
| Taiwan policy rate | CBC CSV open data | Avg 1-year fixed deposit rate across banks, monthly since 2001 |

### Taiwan Macro Data Sources

All Taiwan macro data is downloaded automatically from official government open data APIs (Open Government Data License v1.0, no API key required):

| Series | Agency | Endpoint |
|--------|--------|----------|
| CPI | DGBAS (主計總處) | `ws.dgbas.gov.tw` XML |
| Industrial Production | MOEA (經濟部) | `service.moea.gov.tw` CSV |
| Exports / Imports | Customs Administration (關務署) | `opendata.customs.gov.tw` CSV |
| Policy Rate | CBC (中央銀行) | `www.cbc.gov.tw` CSV |

Data is cached to `data/raw/tw_*.csv` after first download. Delete these files to force re-download.

## Assumptions & Simplifications

- **TAIEX proxy:** EWT ETF used because TAIEX is not available via yfinance. EWT is heavily weighted toward TSMC and Taiwan tech.
- **Rates proxy:** Average 1-year fixed deposit rate across Taiwan banks (from CBC). This is a deposit rate, not the official rediscount rate, but it tracks CBC policy closely and has monthly granularity.
- **Trade data coverage:** Taiwan Customs open data currently provides only ~12 years of monthly history (2014+). Trade YoY features will have fewer usable observations than CPI or IP.
- **Monthly alignment:** Macro data release dates are simplified to month-end. No exact publication-date alignment.
- **Forward-fill:** Gaps ≤3 months are forward-filled in macro series.
- **Lookahead prevention:** All predictive features use 1-month lags. Train/test split is strictly time-ordered.
- **No HMMs / Markov-switching / clustering:** Regime analysis uses simple threshold splits only.

## Output Files

### Charts (`outputs/charts/`)
- `tw_exports_yoy.png` – Taiwan exports growth
- `usdtwd_ret.png` – USD/TWD monthly returns
- `ewt_ret.png` – Taiwan equity returns
- `tw_rate_chg.png` – Taiwan rate changes
- `tw_trade_balance.png` – Trade balance
- `correlation_heatmap.png` – Key variable correlations
- `rolling_corr_fx.png` / `rolling_corr_eq.png` – Rolling correlations
- `coef_*.png` – OLS coefficient bar charts
- `pred_*.png` – Actual vs predicted
- `regime_comparison.png` – Regime average returns

### Tables (`outputs/tables/`)
- `summary_stats.csv` – Descriptive statistics
- `missing_data.csv` – Missing data report
- `data_dictionary.csv` – Column metadata
- `explanatory_regressions.csv` – OLS results
- `vif_rich_features.csv` – Multicollinearity check
- `predictive_performance.csv` – Out-of-sample metrics
- `regime_stats.csv` / `regime_correlations.csv` / `regime_ols.csv`
- `robustness_tw_vs_global.csv` – Feature set comparison
