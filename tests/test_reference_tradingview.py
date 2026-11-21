"""Referenzwerte: Die Python-Nachbauten müssen die TradingView-Exporte (data/*.xlsx)
Trade für Trade reproduzieren – gleiche Ein-/Ausstiegstage, Preise, Mengen, G&V,
Exkursionen, Haltedauer und Ausstiegssignal (Toleranz 0,011 USD je Wert).

Diese Exporte stammen direkt aus dem Pine-Strategietester und sind damit die
unabhängige Referenz für die mathematische Korrektheit der Python-Berechnung.
"""

from __future__ import annotations

import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly
from conftest import CAPITAL

WEEKLY_EXPORTS = {
    "2010–2026 ohne SL/TP": ("data/CAPITALCOM_SPX500_1W.csv", "data/tradingview_trades_weekly.xlsx",
                             {}, 202, 11_212.27),
    "2010–2026 SL 6 / TP 5": ("data/CAPITALCOM_SPX500_1W.csv", "data/tradingview_trades_weekly_sl6_tp5.xlsx",
                              {"sl_pct": 6, "tp_pct": 5}, 202, 12_865.34),
    "2000–2009 SL 6 / TP 5": ("data/CAPITALCOM_SPX500_1W_long.csv",
                              "data/tradingview_trades_weekly_2000_2009.xlsx",
                              {"sl_pct": 6, "tp_pct": 5, "start": "2000-01-03", "end": "2009-10-30"},
                              118, -1_454.50),
}


def _assert_identical(py, tv):
    cmp = daily.compare(py, tv)
    bad = cmp[cmp.Abweichung != ""]
    assert bad.empty, "Abweichungen zu TradingView:\n" + bad[["Einstieg", "Abweichung"]].to_string()


@pytest.mark.parametrize("name", WEEKLY_EXPORTS)
def test_weekly_matches_tradingview_trade_by_trade(name):
    csv, xlsx, kw, n, net = WEEKLY_EXPORTS[name]
    trades, _ = weekly.run_backtest(weekly.load_weekly(csv), capital=CAPITAL, **kw)
    py, tv = daily.trades_df(trades), daily.load_tv(xlsx)
    assert len(py) == len(tv) == n
    _assert_identical(py, tv)
    assert py["G&V USD"].sum() == pytest.approx(net, abs=0.01)
    assert daily.summary(py, None, CAPITAL) == daily.summary(tv, None, CAPITAL)


def test_daily_matches_tradingview_trade_by_trade(daily_df):
    """Die Tages-CSV beginnt 2023, der TV-Export 2010. Python startet deshalb mit dem
    TV-Kapital vor dem ersten gemeinsamen Trade (wie friday_rush_backtest.main)."""
    tv = daily.load_tv("data/tradingview_trades.xlsx")
    first = daily_df.trading_day.min().date()
    before = tv[tv.Einstieg < first]
    capital = CAPITAL + before["Kum G&V"].iloc[-1]
    trades, _ = daily.run_backtest(daily_df, daily.Settings(initial_capital=capital))
    py, tv_win = daily.trades_df(trades), tv[tv.Einstieg >= first]
    assert len(py) == len(tv_win) > 150
    _assert_identical(py, tv_win)
    assert (py.Signal == "Exit Ersatztag").sum() > 0            # Feiertagslogik wird genutzt


def test_daily_stop_backtest_without_stops_equals_weekly_reference():
    """daily_stop_backtest.py (Tages-Schlusskurse) muss ohne SL/TP exakt dieselben Trades
    wie der Weekly-Nachbau/TradingView liefern (202 Trades, 11.212,27 USD)."""
    d = ds.load_daily_close("data/CAPITALCOM_SPX500_1D_close.csv")
    base = ds.simulate(d.close.values, ds.trade_windows(d))
    tv = daily.load_tv("data/tradingview_trades_weekly.xlsx")
    assert len(base) == len(tv) == 202
    assert base.pnl.sum() == pytest.approx(11_212.27, abs=0.01)
    assert base["in"].values == pytest.approx(tv["Preis rein"].values)
    assert base["out"].values == pytest.approx(tv["Preis raus"].values)
    assert base.qty.values == pytest.approx(tv["Menge"].values)
