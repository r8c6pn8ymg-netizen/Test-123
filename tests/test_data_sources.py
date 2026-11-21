"""Datenqualität der Schlusskurs-Datei, Übereinstimmung der Kursquellen und feste
Zufalls-Seeds (übernommen aus claude/gracious-hamilton-qspy15).

Ergänzt tests/test_survivorship.py (prüft nur die OHLC-Dateien) und
tests/test_reference_tradingview.py (Determinismus der Backtests).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def close_only():
    return ds.load_daily_close("data/CAPITALCOM_SPX500_1D_close.csv")


def test_close_only_file_quality(close_only):
    d = close_only
    assert not d.close.isna().any()
    assert (d.close > 0).all()
    assert d.day.is_monotonic_increasing and not d.day.duplicated().any()
    assert (d.day.dt.dayofweek < 5).all(), "Wochenend-Kerzen in Tagesdaten"
    assert d.day.diff().max() <= pd.Timedelta(days=7)
    assert d.close.pct_change().abs().max() < 0.25


def test_sources_agree_daily_ohlc_vs_close_only(close_only, daily_df):
    o = daily_df.assign(day=daily_df.trading_day.dt.tz_localize(None))
    m = close_only.merge(o, on="day", suffixes=("_c", "_o"))
    assert len(m) > 900
    assert np.allclose(m.close_c, m.close_o)


def test_sources_agree_weekly_close_is_last_daily_close(close_only, weekly_df):
    wk = weekly_df.assign(monday=weekly_df.week_start.dt.tz_localize(None)).set_index("monday")
    last = close_only.groupby("monday").close.last()
    j = wk[["close"]].join(last.rename("daily_last"), how="inner")
    assert len(j) > 800
    assert np.allclose(j.close, j.daily_last)


def test_all_random_generators_are_seeded():
    """Monte-Carlo, Bootstrap und Permutationstests müssen mit festem Seed laufen."""
    for p in sorted(ROOT.glob("*.py")):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"default_rng\(\s*\)", src), f"{p.name}: Zufallsgenerator ohne Seed"
        assert not re.search(r"np\.random\.(rand|randn|randint|choice|normal|shuffle|permutation)\(", src), \
            f"{p.name}: globaler (nicht geseedeter) NumPy-Zufall"
        assert "import random" not in src, f"{p.name}: Modul random ohne Seed-Kontrolle"
