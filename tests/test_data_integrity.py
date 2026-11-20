"""Prüft, dass die Kursdaten und gespeicherten Ergebnisse vollständig und plausibel sind."""

from __future__ import annotations

import pandas as pd
import pytest

PRICE_FILES = ["CAPITALCOM_SPX500_1D.csv", "CAPITALCOM_SPX500_1D_close.csv", "CAPITALCOM_SPX500_1W.csv"]
TV_EXPORTS = ["tradingview_trades.xlsx", "tradingview_trades_weekly.xlsx", "tradingview_trades_weekly_sl6_tp5.xlsx"]


@pytest.mark.parametrize("name", PRICE_FILES)
def test_price_file_is_clean(data_dir, name):
    d = pd.read_csv(data_dir / name)
    assert {"time", "close"} <= set(d.columns)
    assert len(d) > 500
    assert not d.isna().any().any(), "fehlende Werte"
    assert d["time"].is_unique and d["time"].is_monotonic_increasing, "Zeitstempel doppelt/unsortiert"
    assert (d["close"] > 0).all()


@pytest.mark.parametrize("name", ["CAPITALCOM_SPX500_1D.csv", "CAPITALCOM_SPX500_1W.csv"])
def test_ohlc_is_consistent(data_dir, name):
    d = pd.read_csv(data_dir / name)
    hi = d[["open", "close"]].max(axis=1)
    lo = d[["open", "close"]].min(axis=1)
    assert ((d["high"] >= hi) & (d["low"] <= lo)).all(), "High/Low passen nicht zu Open/Close"


def test_weekly_bars_are_one_week_apart(weekly_df):
    gaps = weekly_df["week_start"].diff().dropna().dt.days
    assert gaps.isin([7]).mean() > 0.99


@pytest.mark.parametrize("name", TV_EXPORTS)
def test_tradingview_exports_are_readable(data_dir, name):
    import friday_rush_backtest as fr

    tv = fr.load_tv(str(data_dir / name))
    assert len(tv) > 100
    assert tv["Einstieg"].notna().all() and tv["Ausstieg"].notna().all()
    assert (pd.Series(tv["Ausstieg"]) >= pd.Series(tv["Einstieg"])).all()


def test_stored_grid_is_complete(data_dir):
    g = pd.read_csv(data_dir / "daily_stop_grid_full.csv")
    assert len(g) == 20 * 30  # 20 SL-Werte × 30 TP-Werte
    assert not g.duplicated(["SL %", "TP %"]).any()
