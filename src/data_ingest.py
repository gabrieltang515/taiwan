"""
Data ingestion module.

Downloads market data (yfinance) and macro data (FRED + Taiwan gov APIs)
for the Taiwan Cross-Asset Macro Playbook.  All downloads cached to data/raw/.

Data sources
------------
* yfinance  – FX, equity, and global market tickers (free, no API key).
* FRED      – US macro series (Fed Funds, US yields) via public CSV endpoint.
* DGBAS     – Taiwan CPI (XML from ws.dgbas.gov.tw).
* MOEA      – Taiwan Industrial Production Index (CSV open data).
* Customs   – Taiwan monthly exports/imports (CSV from opendata.customs.gov.tw).
* CBC       – Taiwan bank deposit rates as policy-rate proxy (CSV open data).

All Taiwan government sources are official open data under the
Open Government Data License v1.0 and require no API key.
"""

import warnings
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

from src.config import (
    START_DATE, END_DATE,
    TICKERS, FRED_SERIES, FRED_TRADE,
    DATA_RAW,
)

warnings.filterwarnings("ignore", category=FutureWarning)
# Taiwan gov sites have non-standard SSL certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── helpers ─────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    return str(DATA_RAW / f"{name}.csv")


def _try_load_cache(name: str) -> pd.DataFrame | None:
    p = DATA_RAW / f"{name}.csv"
    if p.exists():
        print(f"  ↻ loading cached {p.name}")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    return None


# ── yfinance downloads ──────────────────────────────────────────────────────

def download_yf_series(ticker: str, name: str) -> pd.Series:
    """Download *Close* from yfinance, cache as CSV."""
    cached = _try_load_cache(name)
    if cached is not None:
        return cached.squeeze()

    print(f"  ⬇ downloading {name} ({ticker}) from yfinance …")
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"Empty result for {ticker}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        series = raw["Close"].rename(name)
        series.to_csv(_cache_path(name))
        return series
    except Exception as e:
        print(f"  ✗ failed to download {name}: {e}")
        return pd.Series(dtype=float, name=name)


def download_all_yf() -> pd.DataFrame:
    """Download all yfinance tickers and return a wide DataFrame (daily)."""
    frames = {}
    for name, ticker in TICKERS.items():
        s = download_yf_series(ticker, name)
        if not s.empty:
            frames[name] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ── FRED downloads (public CSV endpoint, no API key) ───────────────────────

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_fred_series(series_id: str, name: str) -> pd.Series:
    """Download a single FRED series via the public CSV endpoint."""
    cached = _try_load_cache(name)
    if cached is not None:
        return cached.squeeze()

    print(f"  ⬇ downloading {name} ({series_id}) from FRED …")
    try:
        params = {"id": series_id, "cosd": START_DATE, "coed": END_DATE}
        resp = requests.get(FRED_CSV_URL, params=params, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[name] = pd.to_numeric(df[name], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df.to_csv(_cache_path(name))
        return df[name]
    except Exception as e:
        print(f"  ✗ FRED download failed for {name}: {e}")
        return pd.Series(dtype=float, name=name)


def download_all_fred() -> pd.DataFrame:
    """Download US macro series from FRED (Taiwan series loaded separately)."""
    # Only attempt FRED for US series (Taiwan OECD series no longer available)
    us_only = {k: v for k, v in FRED_SERIES.items()
               if not k.startswith("tw_")}
    frames = {}
    for name, sid in us_only.items():
        s = download_fred_series(sid, name)
        if not s.empty:
            frames[name] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ── Taiwan macro from government open data APIs ────────────────────────────

_REQ_KW = dict(timeout=30, verify=False,
               headers={"User-Agent": "Mozilla/5.0"})


def _roc_ym_to_date(ym_int: int) -> pd.Timestamp:
    """Convert ROC year-month integer (e.g. 11412) → 2025-12-01."""
    s = str(int(ym_int))
    month = int(s[-2:])
    year_ad = int(s[:-2]) + 1911
    return pd.Timestamp(year=year_ad, month=month, day=1)


def _download_tw_cpi() -> pd.Series:
    """CPI headline index from DGBAS (XML open data)."""
    url = ("https://ws.dgbas.gov.tw/001/Upload/461/"
           "relfile/11525/230557/pr0105a1m.xml")
    r = requests.get(url, **_REQ_KW)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for obs in root.findall("Obs"):
        item  = obs.findtext("Item", "")
        period = obs.findtext("TIME_PERIOD", "")
        dtype  = obs.findtext("TYPE", "")
        value  = obs.findtext("Item_VALUE", "")
        if "總指數" in item and "原始值" in dtype:
            rows.append({"period": period, "value": value})
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(
        df["period"].str.replace("M", "-"), format="%Y-%m", errors="coerce"
    )
    df = df.dropna(subset=["date", "value"]).sort_values("date").set_index("date")
    return df["value"].rename("tw_cpi")


def _download_tw_ip() -> pd.Series:
    """Industrial production index from MOEA (CSV open data)."""
    url = "https://service.moea.gov.tw/EE520/opendata/d.csv"
    r = requests.get(url, **_REQ_KW)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    # Filter: 生產指數 + overall industry (行業別 = 工業)
    mask = (raw.iloc[:, 0] == "生產指數") & (raw.iloc[:, 2].str.strip() == "工業")
    sub = raw.loc[mask].copy()
    sub["date"]  = sub.iloc[:, 3].apply(_roc_ym_to_date)
    sub["tw_ip"] = pd.to_numeric(sub.iloc[:, 4], errors="coerce")
    sub = sub.dropna(subset=["date", "tw_ip"]).sort_values("date").set_index("date")
    return sub["tw_ip"]


def _download_tw_trade() -> tuple[pd.Series, pd.Series]:
    """Monthly exports & imports from Taiwan Customs (CSV open data)."""
    url = "https://opendata.customs.gov.tw/data/6053/csv.csv"
    r = requests.get(url, **_REQ_KW)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    yr_col, mo_col = raw.columns[0], raw.columns[1]
    exp_col, imp_col = raw.columns[2], raw.columns[5]
    raw["year_ad"] = (
        raw[yr_col].astype(str).str.strip().str.replace('"', "").astype(int) + 1911
    )
    raw["date"] = pd.to_datetime(
        raw["year_ad"].astype(str) + "-"
        + raw[mo_col].astype(int).astype(str).str.zfill(2) + "-01"
    )
    raw["tw_exports"] = pd.to_numeric(
        raw[exp_col].astype(str).str.replace(",", ""), errors="coerce"
    )
    raw["tw_imports"] = pd.to_numeric(
        raw[imp_col].astype(str).str.replace(",", ""), errors="coerce"
    )
    clean = raw[["date", "tw_exports", "tw_imports"]].dropna().sort_values("date")
    clean = clean.set_index("date")
    return clean["tw_exports"], clean["tw_imports"]


def _download_tw_rate() -> pd.Series:
    """Bank deposit rates from CBC (proxy for policy rate)."""
    url = "https://www.cbc.gov.tw/public/data/OpenData/A13Rate.csv"
    r = requests.get(url, **_REQ_KW)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    rate_col = "定存利率-一年期-固定"
    raw[rate_col] = pd.to_numeric(raw[rate_col], errors="coerce")
    ym = raw.iloc[:, 1].astype(str).str.strip()
    raw["month"]    = ym.str[-2:].astype(int)
    raw["year_ad"]  = ym.str[:-2].astype(int) + 1911
    raw["date"] = pd.to_datetime(
        raw["year_ad"].astype(str) + "-"
        + raw["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    monthly = raw.groupby("date")[rate_col].mean().dropna()
    monthly = monthly[monthly.index <= "2026-12-31"]
    return monthly.rename("tw_policy_rate")


def download_tw_macro() -> pd.DataFrame:
    """
    Download Taiwan macro from government open data, cache to CSV.

    Series: CPI (DGBAS), IP (MOEA), exports/imports (Customs), rate (CBC).
    Falls back to cached CSV if the download fails.
    """
    print("\n── Taiwan macro (government open data) ──")
    frames: dict[str, pd.Series] = {}

    downloaders = {
        "tw_cpi":         _download_tw_cpi,
        "tw_ip":          _download_tw_ip,
        "tw_policy_rate": _download_tw_rate,
    }

    for name, func in downloaders.items():
        cached = _try_load_cache(name)
        if cached is not None:
            frames[name] = cached.squeeze()
            continue
        try:
            print(f"  ⬇ downloading {name} …")
            s = func()
            s = s[s.index >= START_DATE]
            s.to_csv(_cache_path(name))
            frames[name] = s
            print(f"  ✓ {name}: {len(s)} obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Trade comes as a pair
    if _try_load_cache("tw_exports") is not None:
        frames["tw_exports"] = _try_load_cache("tw_exports").squeeze()
        frames["tw_imports"] = _try_load_cache("tw_imports").squeeze()
    else:
        try:
            print("  ⬇ downloading tw_exports / tw_imports …")
            exp, imp = _download_tw_trade()
            exp, imp = exp[exp.index >= START_DATE], imp[imp.index >= START_DATE]
            exp.to_csv(_cache_path("tw_exports"))
            imp.to_csv(_cache_path("tw_imports"))
            frames["tw_exports"] = exp
            frames["tw_imports"] = imp
            print(f"  ✓ tw_exports: {len(exp)} obs / tw_imports: {len(imp)} obs")
        except Exception as e:
            print(f"  ✗ tw_exports/imports: {e}")

    if not frames:
        return pd.DataFrame()
    tw = pd.DataFrame(frames)
    tw.index = pd.to_datetime(tw.index)
    tw.sort_index(inplace=True)
    return tw


# ── public entry point ──────────────────────────────────────────────────────

def ingest_all(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (or load cached) all data.

    Returns
    -------
    market_daily : pd.DataFrame   – daily market prices
    macro_monthly : pd.DataFrame  – monthly macro series (FRED + Taiwan local)
    """
    if force_refresh:
        # Only clear yfinance / FRED caches, preserve tw_ CSVs
        for f in DATA_RAW.glob("*.csv"):
            if not f.name.startswith("tw_"):
                f.unlink()

    print("── Market data (yfinance) ──")
    market_daily = download_all_yf()

    print("\n── US macro data (FRED) ──")
    fred_monthly = download_all_fred()

    tw_monthly = download_tw_macro()

    # Merge FRED + Taiwan macro
    if not tw_monthly.empty and not fred_monthly.empty:
        macro_monthly = fred_monthly.join(tw_monthly, how="outer")
    elif not tw_monthly.empty:
        macro_monthly = tw_monthly
    else:
        macro_monthly = fred_monthly

    macro_monthly.sort_index(inplace=True)

    print(f"\n  Market shape: {market_daily.shape}")
    print(f"  Macro  shape: {macro_monthly.shape}")
    return market_daily, macro_monthly


if __name__ == "__main__":
    mkt, macro = ingest_all()
    print("\nMarket tail:\n", mkt.tail())
    print("\nMacro tail:\n", macro.tail())
