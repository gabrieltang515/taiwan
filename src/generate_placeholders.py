"""
Generate placeholder Taiwan macro CSV files for pipeline testing.

These use SIMULATED data with realistic ranges and patterns.
Replace with real data from DGBAS / CBC / Taiwan Customs for actual analysis.
See README.md for download sources.
"""

import pandas as pd
import numpy as np
from src.config import DATA_RAW


def generate_all():
    np.random.seed(42)
    dates = pd.date_range("2005-01-01", "2025-09-01", freq="MS")

    # Taiwan CPI index (base ~85, trending up ~2% per year)
    cpi = 85 * np.cumprod(1 + np.random.normal(0.0015, 0.003, len(dates)))
    pd.DataFrame({"tw_cpi": cpi.round(2)}, index=dates).rename_axis("date").to_csv(
        DATA_RAW / "tw_cpi.csv"
    )

    # Taiwan industrial production (trending up with business cycles)
    cycle = 15 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
    trend = np.linspace(0, 45, len(dates))
    ip = 75 + trend + cycle + np.random.normal(0, 3, len(dates))
    pd.DataFrame({"tw_ip": ip.round(2)}, index=dates).rename_axis("date").to_csv(
        DATA_RAW / "tw_ip.csv"
    )

    # Taiwan policy rate (stepped, CBC-style discrete moves)
    rate = 2.0
    rates = []
    for i in range(len(dates)):
        if i > 0 and np.random.random() < 0.05:
            rate += np.random.choice([-0.125, 0.125, -0.25, 0.25])
            rate = np.clip(rate, 1.0, 3.75)
        rates.append(round(rate, 3))
    pd.DataFrame({"tw_policy_rate": rates}, index=dates).rename_axis("date").to_csv(
        DATA_RAW / "tw_policy_rate.csv"
    )

    # Taiwan exports (billions USD, trending up, COVID dip)
    exp_trend = np.linspace(0, 25, len(dates))
    exp_cycle = 3 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))
    exports = 15 + exp_trend + exp_cycle + np.random.normal(0, 1.5, len(dates))
    covid_mask = (dates >= "2020-02-01") & (dates <= "2020-06-01")
    exports[covid_mask] *= 0.85
    pd.DataFrame({"tw_exports": exports.round(2)}, index=dates).rename_axis("date").to_csv(
        DATA_RAW / "tw_exports.csv"
    )

    # Taiwan imports (surplus economy → imports < exports)
    imports = exports * np.random.uniform(0.8, 0.95, len(dates))
    pd.DataFrame({"tw_imports": imports.round(2)}, index=dates).rename_axis("date").to_csv(
        DATA_RAW / "tw_imports.csv"
    )

    print("✓ Placeholder CSVs generated in data/raw/")
    print("  ⚠ These are SIMULATED — replace with real data for analysis")


if __name__ == "__main__":
    generate_all()
