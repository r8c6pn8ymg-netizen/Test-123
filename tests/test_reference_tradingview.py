"""Referenzwerte: Die Python-Nachbauten müssen die TradingView-Exporte Trade für Trade
reproduzieren, und die gespeicherten Ergebnisdateien in data/ müssen sich aus dem
aktuellen Code exakt neu berechnen lassen (Reproduzierbarkeit).

Toleranz für Preise/G&V: 0,011 USD (wie friday_rush_backtest.compare), d. h. Rundung
auf Cent in den TradingView-Exporten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly
from conftest import CAPITAL, IS_END, IS_START, OOS_END, OOS_START

YEARS_IS = (pd.Timestamp(IS_END) - pd.Timestamp(IS_START)).days / 365.25

# Datei, Datenreihe, Parameter, erwartete Trades, erwarteter Nettogewinn (aus TV-Export)
WEEKLY_REFERENCES = {
    "Basis 2010–2026": ("tradingview_trades_weekly.xlsx", "1W",
                        dict(), 202, 11212.27),
    "SL 6 / TP 5 2010–2026": ("tradingview_trades_weekly_sl6_tp5.xlsx", "1W",
                              dict(sl_pct=6, tp_pct=5), 202, 12865.34),
    "SL 6 / TP 5 2000–2009": ("tradingview_trades_weekly_2000_2009.xlsx", "1W_long",
                              dict(sl_pct=6, tp_pct=5, start=OOS_START, end=OOS_END), 118, None),
}


def _deviations(py: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    cmp = daily.compare(py, tv)
    return cmp[cmp["Abweichung"] != ""]


@pytest.mark.parametrize("name", WEEKLY_REFERENCES)
def test_weekly_matches_tradingview_trade_by_trade(name, weekly_df, weekly_long_df):
    xlsx, series, kw, n_trades, net = WEEKLY_REFERENCES[name]
    df = weekly_df if series == "1W" else weekly_long_df
    trades, _ = weekly.run_backtest(df, capital=CAPITAL, **kw)
    py = daily.trades_df(trades)
    tv = daily.load_tv(f"data/{xlsx}")
    bad = _deviations(py, tv)
    assert bad.empty, bad[["Einstieg", "Abweichung"]].head(10).to_string()
    assert len(py) == len(tv) == n_trades
    if net is not None:
        assert py["G&V USD"].sum() == pytest.approx(net, abs=0.01)


@pytest.mark.parametrize("name", WEEKLY_REFERENCES)
def test_weekly_summary_matches_tradingview(name, weekly_df, weekly_long_df):
    """Kennzahlen (Trefferquote, Profitfaktor, Max. Drawdown ...) aus Python-Trades
    und TV-Trades sind identisch."""
    xlsx, series, kw, *_ = WEEKLY_REFERENCES[name]
    df = weekly_df if series == "1W" else weekly_long_df
    trades, _ = weekly.run_backtest(df, capital=CAPITAL, **kw)
    s_py = daily.summary(daily.trades_df(trades), None, CAPITAL)
    s_tv = daily.summary(daily.load_tv(f"data/{xlsx}"), None, CAPITAL)
    for k in s_py:
        if k != "Exit Ersatztag":
            assert s_py[k] == pytest.approx(s_tv[k], abs=0.02), k


def test_daily_matches_tradingview_trade_by_trade(daily_df):
    """Die Tages-CSV beginnt erst 2023. Python startet deshalb mit dem TV-Kapital vor dem
    ersten gemeinsamen Trade (wie friday_rush_backtest.main)."""
    tv = daily.load_tv("data/tradingview_trades.xlsx")
    first = daily_df.trading_day.min().date()
    before = tv[tv["Einstieg"] < first]
    capital = CAPITAL + before["Kum G&V"].iloc[-1]
    trades, _ = daily.run_backtest(daily_df, daily.Settings(initial_capital=capital))
    py = daily.trades_df(trades)
    tv_win = tv[tv["Einstieg"] >= first]
    bad = _deviations(py, tv_win)
    assert bad.empty, bad[["Einstieg", "Abweichung"]].head(10).to_string()
    assert len(py) == len(tv_win) == 197


# ───────────────────────── Reproduzierbarkeit ─────────────────────────
@pytest.mark.parametrize("csv,series,kw", [
    ("python_trades_weekly.csv", "1W", dict()),
    ("python_trades_weekly_sl6_tp5.csv", "1W", dict(sl_pct=6, tp_pct=5)),
    ("python_trades_weekly_2000_2009.csv", "1W_long", dict(sl_pct=6, tp_pct=5, start=OOS_START, end=OOS_END)),
])
def test_stored_trade_lists_are_reproducible(csv, series, kw, weekly_df, weekly_long_df):
    df = weekly_df if series == "1W" else weekly_long_df
    new = daily.trades_df(weekly.run_backtest(df, capital=CAPITAL, **kw)[0])
    old = pd.read_csv(f"data/{csv}")
    assert len(new) == len(old)
    for c in ("Preis rein", "Preis raus", "Menge", "G&V USD", "Balken"):
        assert np.allclose(new[c].astype(float), old[c].astype(float), atol=1e-9), c
    assert (new["Signal"].values == old["Signal"].values).all()


def test_backtest_is_deterministic(weekly_df):
    a, ca = weekly.run_backtest(weekly_df, sl_pct=6, tp_pct=5)
    b, cb = weekly.run_backtest(weekly_df, sl_pct=6, tp_pct=5)
    assert [(t.entry_time, t.exit_time, t.pnl) for t in a] == [(t.entry_time, t.exit_time, t.pnl) for t in b]
    assert ca.equals(cb)


@pytest.mark.parametrize("sl,tp", [(0.0, 0.0), (6.0, 5.0), (3.0, 10.0)])
def test_stored_sltp_grid_is_reproducible(weekly_df, sl, tp):
    """data/sltp_grid_full.csv (sltp_optimization.py) lässt sich neu berechnen."""
    from sma_filter_test import stats
    g = pd.read_csv("data/sltp_grid_full.csv")
    row = g[(g.SL == sl) & (g.TP == tp)].iloc[0]
    tr, _ = weekly.run_backtest(weekly_df, start=IS_START, end=IS_END, capital=CAPITAL, sl_pct=sl, tp_pct=tp)
    st = stats(tr, CAPITAL, YEARS_IS)
    for k in ("Trades", "Netto USD", "CAGR %", "Max DD %", "Netto/MaxDD"):
        assert st[k] == pytest.approx(row[k], abs=1e-6), k


# ───────────────────────── daily_stop_backtest ─────────────────────────
@pytest.fixture(scope="module")
def daily_close():
    d = ds.load_daily_close("data/CAPITALCOM_SPX500_1D_close.csv")
    return d, ds.trade_windows(d)


def test_daily_stop_control_equals_weekly_backtest(daily_close, weekly_df):
    """Ohne SL/TP muss die Tages-Simulation exakt den Weekly-Backtest ergeben
    (Kontrolle, die das Skript selbst nur ausdruckt)."""
    d, win = daily_close
    base = ds.simulate(d.close.values, win)
    trades, _ = weekly.run_backtest(weekly_df, capital=CAPITAL)
    assert len(base) == len(trades) == 202
    assert base.pnl.sum() == pytest.approx(11212.27, abs=0.01)
    assert np.allclose(base["in"].values, [t.entry_price for t in trades])
    assert np.allclose(base["out"].values, [t.exit_price for t in trades])


@pytest.mark.parametrize("sl,tp", [(0.0, 0.0), (6.0, 5.0), (2.0, 4.0)])
def test_stored_daily_stop_grid_is_reproducible(daily_close, sl, tp):
    d, win = daily_close
    g = pd.read_csv("data/daily_stop_grid_full.csv")
    row = g[(g["SL %"] == sl) & (g["TP %"] == tp)].iloc[0]
    m = ds.metrics(ds.simulate(d.close.values, win, sl, tp), YEARS_IS)
    for k in ("Trades", "Netto USD", "CAGR %", "Max DD %", "SL", "TP"):
        assert m[k] == pytest.approx(row[k], abs=1e-6), k
