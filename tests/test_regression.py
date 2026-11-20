"""Regressionstests: Die Python-Backtests müssen Trade für Trade mit TradingView übereinstimmen
und die gespeicherten Ergebnisse reproduzieren. Schlägt einer fehl, hat eine Code-Änderung
die Logik verändert."""

from __future__ import annotations

import pandas as pd
import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as fr
import monthly_mid_weekly_backtest as mw

pytestmark = pytest.mark.regression


def _assert_identical(py: pd.DataFrame, tv: pd.DataFrame, n_expected: int) -> None:
    cmp = fr.compare(py, tv)
    bad = cmp[cmp["Abweichung"] != ""]
    assert int((cmp["_merge"] == "both").sum()) == n_expected
    assert bad.empty, "Abweichungen zu TradingView:\n" + bad[["Einstieg", "Abweichung"]].to_string(index=False)


def test_friday_rush_matches_tradingview(daily_df, data_dir):
    tv = fr.load_tv(str(data_dir / "tradingview_trades.xlsx"))
    first_day = daily_df.trading_day.min().date()
    before = tv[tv["Einstieg"] < first_day]
    capital = 10_000.0 + (before["Kum G&V"].iloc[-1] if len(before) else 0.0)  # wie main()
    trades, _ = fr.run_backtest(daily_df, fr.Settings(initial_capital=capital))
    _assert_identical(fr.trades_df(trades), tv[tv["Einstieg"] >= first_day], 197)


@pytest.mark.parametrize(
    ("export", "sl", "tp"),
    [("tradingview_trades_weekly.xlsx", 0.0, 0.0), ("tradingview_trades_weekly_sl6_tp5.xlsx", 6.0, 5.0)],
    ids=["ohne SL/TP", "SL6 TP5"],
)
def test_weekly_matches_tradingview(weekly_df, data_dir, export, sl, tp):
    trades, curve = mw.run_backtest(weekly_df, sl_pct=sl, tp_pct=tp)
    _assert_identical(fr.trades_df(trades), fr.load_tv(str(data_dir / export)), 202)
    assert len(curve) == len(weekly_df)


def test_daily_control_equals_weekly_result(daily_close_df):
    # Ohne SL/TP muss der Tages-Backtest das Weekly/TradingView-Ergebnis reproduzieren
    base = ds.simulate(daily_close_df.close.values, ds.trade_windows(daily_close_df))
    assert len(base) == 202
    assert base.pnl.sum() == pytest.approx(11212.27, abs=0.01)


@pytest.mark.parametrize(("sl", "tp"), [(0.0, 0.0), (6.0, 5.0), (3.0, 0.0), (0.0, 7.5), (10.0, 15.0)])
def test_daily_grid_reproduces_stored_results(daily_close_df, data_dir, sl, tp):
    grid = pd.read_csv(data_dir / "daily_stop_grid_full.csv")
    stored = grid[(grid["SL %"] == sl) & (grid["TP %"] == tp)].iloc[0]
    years = (pd.Timestamp("2026-10-30") - pd.Timestamp("2010-01-01")).days / 365.25
    m = ds.metrics(ds.simulate(daily_close_df.close.values, ds.trade_windows(daily_close_df), sl, tp), years)
    for key, val in m.items():
        assert val == pytest.approx(stored[key], abs=0.011), key
