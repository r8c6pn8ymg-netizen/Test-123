"""Datenverarbeitung: Einlesen, Zeitzonen, Rundung, Auswertungs- und Vergleichstabellen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly


# ───────────────────────── Einlesen / Handelstag ─────────────────────────
def test_load_csv_maps_evening_bar_to_next_trading_day(tmp_path):
    """Capital.com-Tageskerzen öffnen am Vorabend (22:00 UTC Winter, 21:00/23:00 je nach
    Sommerzeit). Alle müssen auf den Folgetag fallen (time_tradingday)."""
    p = tmp_path / "d.csv"
    times = [pd.Timestamp("2024-01-03 22:00", tz="UTC"),     # Mi-Abend -> Do 4.1.
             pd.Timestamp("2024-07-03 21:00", tz="UTC"),     # Sommer: Mi-Abend -> Do 4.7.
             pd.Timestamp("2024-01-04 23:00", tz="UTC")]     # Do-Abend -> Fr 5.1.
    pd.DataFrame({" Time ": [int(t.timestamp()) for t in times], "OPEN": 1, "High": 2,
                  "low": 0.5, "Close": 1.5}).to_csv(p, index=False)
    df = daily.load_csv(str(p))
    assert list(df.columns[:5]) == ["time", "open", "high", "low", "close"]   # bereinigt
    assert [d.strftime("%Y-%m-%d %a") for d in df.trading_day] == [
        "2024-01-04 Thu", "2024-07-04 Thu", "2024-01-05 Fri"]


def test_load_weekly_derives_monday_and_friday(tmp_path):
    p = tmp_path / "w.csv"
    sunday = pd.Timestamp("2024-03-31 22:00", tz="UTC")       # Sommerzeit-Umstellung
    pd.DataFrame({"time": [int(sunday.timestamp())], "open": 1, "high": 1, "low": 1,
                  "close": 1}).to_csv(p, index=False)
    w = weekly.load_weekly(str(p)).iloc[0]
    assert w.week_start.strftime("%Y-%m-%d %a") == "2024-04-01 Mon"
    assert w.friday.strftime("%Y-%m-%d %a") == "2024-04-05 Fri"


def test_load_daily_close_week_keys(tmp_path):
    p = tmp_path / "c.csv"
    t = [pd.Timestamp("2024-01-03 22:00", tz="UTC"), pd.Timestamp("2024-01-07 22:00", tz="UTC")]
    pd.DataFrame({"time": [int(x.timestamp()) for x in t], "close": [1.0, 2.0]}).to_csv(p, index=False)
    d = ds.load_daily_close(str(p))
    assert [str(x.date()) for x in d.day] == ["2024-01-04", "2024-01-08"]
    assert [str(x.date()) for x in d.monday] == ["2024-01-01", "2024-01-08"]
    assert [str(x.date()) for x in d.friday] == ["2024-01-05", "2024-01-12"]


# ───────────────────────── Rundung ─────────────────────────
@pytest.mark.parametrize("x,step,expected", [
    (97.087, 0.1, 97.0),
    (0.3, 0.1, 0.3),            # 0,3 / 0,1 = 2,9999999999999996 -> ohne Epsilon 0,2
    (12.99, 0.1, 12.9),
    (5.0, 1.0, 5.0),
    (0.09, 0.1, 0.0),
])
def test_floor_step(x, step, expected):
    assert daily.floor_step(x, step) == pytest.approx(expected)


@pytest.mark.parametrize("x,expected", [
    (0.25, 0.3),                # kaufmännisch (Pine round_to_mintick), Python round() ergäbe 0,2
    (105.05, 105.1),
    (94.04, 94.0),
    (100 * 0.94, 94.0),
    (4123.456 * 1.05, 4329.6),  # 4329,6288 -> 4329,6
])
def test_round_tick_half_up(x, expected):
    assert weekly.round_tick(x, 0.1) == pytest.approx(expected, abs=1e-9)


# ───────────────────────── Auswertung ─────────────────────────
def _tdf(pnls, signals=None):
    n = len(pnls)
    return pd.DataFrame({"G&V USD": pnls, "Signal": signals or ["Exit Freitag"] * n})


def test_summary_hand_calculation():
    s = daily.summary(_tdf([100, -50, 200, -100, 0], ["Exit Freitag"] * 4 + ["Exit Ersatztag"]),
                      None, 10_000)
    assert (s["Trades"], s["Gewinner"], s["Verlierer"], s["Ausgeglichen"]) == (5, 2, 2, 1)
    assert s["Trefferquote %"] == 40.0
    assert s["Nettogewinn USD"] == 150 and s["Nettogewinn %"] == 1.5
    assert s["Bruttogewinn USD"] == 300 and s["Bruttoverlust USD"] == 150
    assert s["Profitfaktor"] == 2.0
    assert s["Durchschn. Gewinn USD"] == 150 and s["Durchschn. Verlust USD"] == 75
    assert s["Größter Gewinn USD"] == 200 and s["Größter Verlust USD"] == 100
    # realisierte Equity 0, 100, 50, 250, 150, 150 -> Hoch 250, Tief 150
    assert s["Max. Drawdown (Close zu Close) USD"] == 100
    assert s["Exit Ersatztag"] == 1


def test_summary_drawdown_counts_loss_from_start_capital():
    """Startkapital ist der erste Hochpunkt: ein erster Verlust ist bereits Drawdown."""
    assert daily.summary(_tdf([-80, 30]), None, 1_000)["Max. Drawdown (Close zu Close) USD"] == 80


def test_summary_without_losses_has_infinite_profit_factor():
    assert daily.summary(_tdf([10, 20]), None, 1_000)["Profitfaktor"] == float("inf")


def test_daily_stop_metrics_hand_calculation():
    t = pd.DataFrame({"pnl": [100.0, -50.0, 200.0],
                      "signal": ["Exit Ziel", "Stop-Loss", "Take-Profit"]})
    m = ds.metrics(t, years=1.0, capital=1_000)
    # Equity 1000, 1100, 1050, 1250 -> CAGR 25 %, Max DD 50 / 1100 = 4,5 %
    assert m["Netto USD"] == 250 and m["CAGR %"] == 25.0
    assert m["Max DD %"] == 4.5 and m["Netto/MaxDD"] == 5.0
    assert m["PF"] == 6.0 and m["Treffer %"] == 66.7
    assert (m["SL"], m["TP"]) == (1, 1)


# ───────────────────────── Vergleich Python <-> TradingView ─────────────────────────
def _trades_frame():
    return pd.DataFrame({
        "Einstieg": pd.to_datetime(["2024-01-04", "2024-01-11"]).date,
        "Ausstieg": pd.to_datetime(["2024-01-05", "2024-01-12"]).date,
        "Preis rein": [100.0, 101.0], "Preis raus": [102.0, 100.0], "Signal": ["Exit Freitag"] * 2,
        "Menge": [10.0, 10.0], "G&V USD": [20.0, -10.0], "Run-up USD": [20.0, 0.0],
        "Drawdown USD": [0.0, -10.0], "Balken": [1, 1],
    })


def test_compare_identical_has_no_deviation():
    a = _trades_frame()
    assert (daily.compare(a, a.copy()).Abweichung == "").all()


def test_compare_flags_differences_and_missing_trades():
    a, b = _trades_frame(), _trades_frame()
    b.loc[0, "G&V USD"] = 20.02                    # > 0,011 Toleranz
    b.loc[1, "Signal"] = "Exit Ersatztag"
    c = daily.compare(a, b).set_index("Einstieg").Abweichung
    assert "G&V USD" in c.iloc[0] and "Signal" in c.iloc[1]
    c2 = daily.compare(a, b.iloc[:1])
    assert (c2.Abweichung == "nur in Python").sum() == 1


def test_trades_df_columns_and_rounding():
    t = daily.Trade(1, "Long", pd.Timestamp("2024-01-03 22:00", tz="UTC"),
                    pd.Timestamp("2024-01-04", tz="UTC"), 100.0, 9.99999)
    t.exit_day, t.exit_price, t.pnl, t.ret_pct = pd.Timestamp("2024-01-05", tz="UTC"), 101.0, 9.999994, 1.0
    row = daily.trades_df([t]).iloc[0]
    assert row["Menge"] == 10.0 and row["G&V USD"] == 10.0
    assert str(row["Einstieg"]) == "2024-01-04"


# ───────────────────────── Datenqualität der Schlusskurs-Datei ─────────────────────────
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
