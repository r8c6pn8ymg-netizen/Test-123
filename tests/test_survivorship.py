"""Survivorship Bias und Datenqualität.

Ausgangslage: Alle Backtests laufen auf einem einzigen Instrument, dem S&P-500-Index
(CAPITALCOM:SPX500). Der Indexstand selbst ist frei von klassischem Survivorship Bias,
weil der Index seine ausgeschiedenen Mitglieder historisch enthalten hat. Es bleibt aber
ein Auswahl-Bias: Der S&P 500 wurde ausgewählt, obwohl man heute weiß, dass er einer der
erfolgreichsten Märkte der letzten Jahrzehnte war. Würde die Strategie später auf
Einzelaktien übertragen, müssten auch delistete Titel in den Daten sein.

Geprüft wird hier:
  * die Datenreihen sind vollständig, lückenlos und plausibel
  * die Testzeiträume enthalten echte Krisen (nicht nur "überlebende" Phasen)
  * die Out-of-Sample-Daten stimmen mit den In-Sample-Daten überein (gleiche Quelle)
  * ein Hinweis auf den verbleibenden Auswahl-Bias wird als Warnung ausgegeben
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest_checks as bc

PRICE_FILES = {
    "1W": ("data/CAPITALCOM_SPX500_1W.csv", 8),
    "1W_long": ("data/CAPITALCOM_SPX500_1W_long.csv", 8),
    "1D": ("data/CAPITALCOM_SPX500_1D.csv", 5),
}


class SelectionBiasWarning(UserWarning):
    pass


@pytest.fixture(scope="module", params=list(PRICE_FILES))
def price_data(request):
    path, max_gap_days = PRICE_FILES[request.param]
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return request.param, df, max_gap_days


def test_no_missing_or_duplicate_bars(price_data):
    name, df, max_gap = price_data
    assert not df[["open", "high", "low", "close"]].isna().any().any(), f"{name}: fehlende Kurse"
    assert not df.time.duplicated().any(), f"{name}: doppelte Zeitstempel"
    assert df.time.is_monotonic_increasing, f"{name}: Zeitstempel nicht sortiert"
    gap = df.time.diff().max()
    assert gap <= pd.Timedelta(days=max_gap), f"{name}: Datenlücke von {gap}"


def test_ohlc_is_consistent(price_data):
    name, df, _ = price_data
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df.high >= df[["open", "close"]].max(axis=1) - 1e-9).all(), f"{name}: High < Open/Close"
    assert (df.low <= df[["open", "close"]].min(axis=1) + 1e-9).all(), f"{name}: Low > Open/Close"


def test_no_extreme_price_jumps(price_data):
    """Sprünge > 25 % von einem Balken zum nächsten deuten auf fehlerhafte Daten hin."""
    name, df, _ = price_data
    jumps = df.close.pct_change().abs()
    assert jumps.max() < 0.25, f"{name}: verdächtiger Kurssprung {jumps.max():.1%}"


def test_weekly_sources_agree_on_overlap(weekly_df, weekly_long_df):
    m = weekly_df.merge(weekly_long_df, on="time", suffixes=("_a", "_b"))
    assert len(m) == len(weekly_df), "Kurze Wochenreihe muss vollständig in der langen enthalten sein"
    for c in ("open", "high", "low", "close"):
        assert np.allclose(m[f"{c}_a"], m[f"{c}_b"]), f"Abweichung in Spalte {c}"


def test_samples_include_crises(weekly_long_df, weekly_df):
    """Out-of-Sample muss Dotcom-Crash und Finanzkrise enthalten, In-Sample den Corona-Crash
    und 2022. Sonst wäre die Stichprobe auf 'überlebende' Phasen beschränkt."""
    oos = weekly_long_df[weekly_long_df.time < pd.Timestamp("2010-01-01", tz="UTC")]
    _, dd_oos = bc.max_drawdown(oos.close.values)
    assert dd_oos > 0.45, f"OOS-Drawdown des Index nur {dd_oos:.0%}"
    ins = weekly_df[weekly_df.time >= pd.Timestamp("2010-01-01", tz="UTC")]
    _, dd_is = bc.max_drawdown(ins.close.values)
    assert dd_is > 0.25, f"IS-Drawdown des Index nur {dd_is:.0%}"


def test_price_files_are_raw_prices_not_adjusted_returns():
    """Kontrolle, dass alle Kursdateien Kerzen mit Zeitstempel sind (keine bereits
    gefilterten Trade-Listen o. Ä.)."""
    for path, _ in PRICE_FILES.values():
        cols = pd.read_csv(path, nrows=1).columns.str.lower().tolist()
        assert cols == ["time", "open", "high", "low", "close"], path


def test_single_instrument_selection_bias_is_documented():
    instruments = {Path(p).name.split("_")[1] for p, _ in PRICE_FILES.values()}
    assert instruments == {"SPX500"}
    warnings.warn(
        "Survivorship/Auswahl-Bias: Alle Backtests nutzen nur den S&P 500 – einen Markt, der "
        "rückblickend außergewöhnlich gut lief. Ergebnisse sind nicht auf andere Märkte oder "
        "Einzelaktien übertragbar, ohne dort erneut (inkl. delisteter Titel) zu testen.",
        SelectionBiasWarning,
        stacklevel=2,
    )
