"""Reproduzierbarkeit: gleiche Eingaben -> gleiche Ergebnisse, feste Zufalls-Seeds,
festgeschriebene Bibliotheksversionen."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pandas as pd
import pytest

import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(ROOT.glob("*.py"))


def _trade_tuples(trades):
    return [(t.entry_time, t.entry_price, t.qty, t.exit_time, t.exit_price, t.exit_signal, t.pnl) for t in trades]


def test_weekly_backtest_is_deterministic_and_does_not_mutate_input(weekly_df):
    before = weekly_df.copy()
    a, ca = weekly.run_backtest(weekly_df, sl_pct=6, tp_pct=5)
    b, cb = weekly.run_backtest(weekly_df, sl_pct=6, tp_pct=5)
    assert _trade_tuples(a) == _trade_tuples(b)
    pd.testing.assert_series_equal(ca, cb)
    pd.testing.assert_frame_equal(weekly_df, before)


def test_daily_backtest_is_deterministic_and_does_not_mutate_input(daily_df):
    before = daily_df.copy()
    a, ca = daily.run_backtest(daily_df, daily.Settings())
    b, cb = daily.run_backtest(daily_df, daily.Settings())
    assert _trade_tuples(a) == _trade_tuples(b)
    pd.testing.assert_series_equal(ca, cb)
    pd.testing.assert_frame_equal(daily_df, before)


def test_all_random_generators_are_seeded():
    """Monte-Carlo, Bootstrap und Permutationstests müssen mit festem Seed laufen."""
    for p in SCRIPTS:
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"default_rng\(\s*\)", src), f"{p.name}: Zufallsgenerator ohne Seed"
        assert not re.search(r"np\.random\.(rand|randn|randint|choice|normal|shuffle|permutation)\(", src), \
            f"{p.name}: globaler (nicht geseedeter) NumPy-Zufall"
        assert "import random" not in src, f"{p.name}: Modul random ohne Seed-Kontrolle"


def _requirements():
    req = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*(==|>=)\s*([^\s#]+)", line)
        if m:
            req[m.group(1).lower()] = (m.group(2), m.group(3))
    return req


def test_excel_reader_is_in_requirements():
    """load_tv liest die TradingView-Exporte mit pandas.read_excel -> benötigt openpyxl."""
    assert "openpyxl" in _requirements()


@pytest.mark.parametrize("dist", sorted(k for k, (op, _) in _requirements().items() if op == "=="))
def test_installed_versions_match_pinned_requirements(dist):
    try:
        installed = metadata.version(dist)
    except metadata.PackageNotFoundError:
        pytest.skip(f"{dist} ist in dieser Umgebung nicht installiert")
    assert installed == _requirements()[dist][1], f"{dist}: installiert {installed}"


def test_pytest_runs_in_project_interpreter():
    """pytest muss dieselbe Umgebung nutzen wie die Skripte (python -m pytest)."""
    import numpy
    import openpyxl
    import scipy
    import statsmodels
    assert all(m.__version__ for m in (pd, numpy, openpyxl, scipy, statsmodels))
