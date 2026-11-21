"""Signalgenerierung, Stop-Loss/Take-Profit-Ausführung und Datenaufbereitung.

Teil 1: synthetische Kurse, bei denen das richtige Ergebnis von Hand feststeht.
Teil 2: Regeln, die auf den echten Daten für jeden einzelnen Trade gelten müssen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly
from conftest import CAPITAL

# Wochenkerzen öffnen sonntags 22:00 UTC. Freitage: 5., 12., 19., 26. März, 2. April 2021
WEEK0 = pd.Timestamp("2021-02-28 22:00", tz="UTC")
RANGE_2021 = dict(start="2021-01-01", end="2021-12-31", capital=1000.0)


def make_weekly(tmp_path, bars) -> pd.DataFrame:
    """bars: Liste von (open, high, low, close). Geht über weekly.load_weekly (CSV)."""
    times = [WEEK0 + pd.Timedelta(weeks=i) for i in range(len(bars))]
    raw = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    raw.insert(0, "time", [int(t.timestamp()) for t in times])
    path = tmp_path / "weekly.csv"
    raw.to_csv(path, index=False)
    return weekly.load_weekly(str(path))


def make_daily(tmp_path, days, bars) -> pd.DataFrame:
    """days: Handelstage 'YYYY-MM-DD'; Kerze öffnet am Vorabend 22:00 UTC."""
    times = [pd.Timestamp(d, tz="UTC") - pd.Timedelta(hours=2) for d in days]
    raw = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    raw.insert(0, "time", [int(t.timestamp()) for t in times])
    path = tmp_path / "daily.csv"
    raw.to_csv(path, index=False)
    return daily.load_csv(str(path))


def flat(c):
    return (c, c + 1, c - 1, c)


# ───────────────────────── Hilfsfunktionen ─────────────────────────
def test_floor_step():
    assert daily.floor_step(2.56, 0.1) == pytest.approx(2.5)
    assert daily.floor_step(2.5, 0.1) == pytest.approx(2.5)        # 2.5/0.1 = 24.999… darf nicht auf 2.4 fallen
    assert daily.floor_step(0.3, 0.1) == pytest.approx(0.3)
    assert daily.floor_step(0.09, 0.1) == 0


def test_round_tick_rounds_half_up_like_pine():
    assert weekly.round_tick(4123.45, 0.1) == pytest.approx(4123.5)
    assert weekly.round_tick(4123.44, 0.1) == pytest.approx(4123.4)
    assert weekly.round_tick(0.25, 0.1) == pytest.approx(0.3)      # Python round() gäbe 0.2


def test_weekly_loader_derives_monday_and_friday(tmp_path):
    df = make_weekly(tmp_path, [flat(100)] * 5)
    assert (df.week_start.dt.dayofweek == 0).all()
    assert (df.friday.dt.dayofweek == 4).all()
    assert list(df.friday.dt.day) == [5, 12, 19, 26, 2]


def test_daily_loader_assigns_trading_day(tmp_path):
    df = make_daily(tmp_path, ["2024-01-04"], [flat(100)])
    assert df.time.iloc[0] == pd.Timestamp("2024-01-03 22:00", tz="UTC")
    assert df.trading_day.iloc[0] == pd.Timestamp("2024-01-04", tz="UTC")


# ───────────────────────── Weekly: Signale ─────────────────────────
def test_weekly_entry_and_target_exit(tmp_path):
    """Einstieg Freitag 12. (8.–14.), Ausstieg Freitag 26. (>= 24.) zum Close."""
    df = make_weekly(tmp_path, [flat(90), flat(100), flat(103), flat(105), flat(104)])
    (t,), curve = weekly.run_backtest(df, **RANGE_2021)
    assert t.entry_day == df.week_start.iloc[1] and t.entry_price == 100
    assert t.exit_day == df.week_start.iloc[3] and t.exit_price == 105
    assert t.exit_signal == "Exit Ziel" and t.bars == 2
    assert t.qty == pytest.approx(10) and t.pnl == pytest.approx(50)
    assert list(curve.values) == pytest.approx([1000, 1000, 1030, 1050, 1050])


def test_weekly_replacement_exit_in_new_month(tmp_path):
    """Ausstiegstag 30 -> Zielwoche ab 27. nie erreicht -> Ausstieg in der ersten Aprilwoche."""
    df = make_weekly(tmp_path, [flat(90), flat(100), flat(103), flat(105), flat(102)])
    (t,), _ = weekly.run_backtest(df, exit_day=30, **RANGE_2021)
    assert t.exit_signal == "Exit Ersatz" and t.exit_price == 102 and t.bars == 3


def test_weekly_no_entry_outside_date_range(tmp_path):
    df = make_weekly(tmp_path, [flat(90), flat(100), flat(103), flat(105)])
    trades, _ = weekly.run_backtest(df, start="2021-04-01", end="2021-12-31")
    assert trades == []


def test_weekly_sma_filter_blocks_entry_below_average(tmp_path):
    df = make_weekly(tmp_path, [flat(120), flat(110), flat(100), flat(105)])
    # Einstiegswoche ist Zeile 1: SMA(2) = 115 > 110 -> kein Einstieg; SMA(5) = na -> kein Einstieg
    assert weekly.run_backtest(df, sma_len=2, **RANGE_2021)[0] == []
    assert weekly.run_backtest(df, sma_len=5, **RANGE_2021)[0] == []
    df2 = make_weekly(tmp_path, [flat(90), flat(100), flat(103), flat(105)])
    assert len(weekly.run_backtest(df2, sma_len=2, **RANGE_2021)[0]) == 1    # 100 > 95


# ───────────────────────── Weekly: Stop-Loss / Take-Profit ─────────────────────────
# Einstieg 100 in Zeile 1, die Kerze in Zeile 2 entscheidet.
SLTP_CASES = {
    # Name: (Kerze o, h, l, c,  Richtung, SL %, TP %,  erwarteter Preis, Signal)
    "SL innerhalb der Kerze": ((99, 100, 94, 96), "Long", 5, 0, 95.0, "Stop-Loss"),
    "SL Gap unter Marke -> Open": ((90, 92, 88, 91), "Long", 5, 0, 90.0, "Stop-Loss"),
    "TP innerhalb der Kerze": ((101, 106, 100, 104), "Long", 0, 5, 105.0, "Take-Profit"),
    "TP Gap über Marke -> Open": ((107, 108, 106, 107), "Long", 0, 5, 107.0, "Take-Profit"),
    "beide, Low näher am Open": ((100, 106, 98, 101), "Long", 2, 5, 98.0, "Stop-Loss"),
    "beide, gleich weit -> Low zuerst": ((100, 106, 94, 101), "Long", 5, 5, 95.0, "Stop-Loss"),
    "beide, High näher am Open": ((100, 105.5, 94, 101), "Long", 5, 5, 105.0, "Take-Profit"),
    "Short SL oberhalb": ((101, 106, 100.5, 104), "Short", 5, 0, 105.0, "Stop-Loss"),
    "Short TP unterhalb": ((99, 99.5, 94, 96), "Short", 0, 5, 95.0, "Take-Profit"),
    "keine Marke erreicht": ((100, 103, 97, 102), "Long", 5, 5, None, None),
}


@pytest.mark.parametrize("case", SLTP_CASES)
def test_weekly_sl_tp_fill(tmp_path, case):
    bar, direction, sl, tp, px, signal = SLTP_CASES[case]
    df = make_weekly(tmp_path, [flat(90), flat(100), bar, flat(103)])
    (t,), _ = weekly.run_backtest(df, direction=direction, sl_pct=sl, tp_pct=tp, **RANGE_2021)
    if px is None:
        assert t.exit_signal == "Exit Ziel" and t.exit_price == 103
    else:
        assert t.exit_signal == signal
        assert t.exit_price == pytest.approx(px)
        assert t.exit_day == df.week_start.iloc[2]
        sign = 1 if direction == "Long" else -1
        assert t.pnl == pytest.approx(sign * (px - 100) * t.qty)


# ───────────────────────── Daily: Signale ─────────────────────────
DAILY_RANGE = dict(start="2024-01-01", end="2024-12-31", initial_capital=1000.0)


def test_daily_thursday_in_friday_out(tmp_path):
    df = make_daily(tmp_path, ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
                    [flat(98), flat(100), flat(104), flat(90)])
    (t,), _ = daily.run_backtest(df, daily.Settings(**DAILY_RANGE))
    assert t.entry_day.dayofweek == 3 and t.entry_price == 100
    assert t.exit_signal == "Exit Freitag" and t.exit_price == 104 and t.bars == 1


def test_daily_flat_friday_is_holiday(tmp_path):
    """Flache Freitagskerze (High == Low) gilt als Feiertag -> Ausstieg Montag."""
    days = ["2024-01-04", "2024-01-05", "2024-01-08"]
    bars = [flat(100), (101, 101, 101, 101), flat(103)]
    df = make_daily(tmp_path, days, bars)
    (t,), _ = daily.run_backtest(df, daily.Settings(**DAILY_RANGE))
    assert t.exit_signal == "Exit Ersatztag" and t.exit_price == 103 and t.bars == 2
    (t2,), _ = daily.run_backtest(df, daily.Settings(skip_flat=False, **DAILY_RANGE))
    assert t2.exit_signal == "Exit Freitag" and t2.exit_price == 101


def test_daily_flat_thursday_no_entry(tmp_path):
    df = make_daily(tmp_path, ["2024-01-04", "2024-01-05"], [(100, 100, 100, 100), flat(104)])
    assert daily.run_backtest(df, daily.Settings(**DAILY_RANGE))[0] == []


def test_daily_sma_na_blocks_entry(tmp_path):
    df = make_daily(tmp_path, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                    [flat(90), flat(95), flat(100), flat(104)])
    assert daily.run_backtest(df, daily.Settings(sma_len=5, **DAILY_RANGE))[0] == []
    assert len(daily.run_backtest(df, daily.Settings(sma_len=3, **DAILY_RANGE))[0]) == 1   # 100 > 95


# ───────────────────────── Kennzahlen der Skripte (Handrechnung) ─────────────────────────
def test_friday_rush_summary_hand_calculation():
    tdf = pd.DataFrame({"G&V USD": [100.0, -50.0, 200.0, -300.0],
                        "Signal": ["Exit Freitag"] * 3 + ["Exit Ersatztag"]})
    s = daily.summary(tdf, None, 1000.0)
    assert s["Nettogewinn USD"] == -50 and s["Nettogewinn %"] == -5
    assert s["Trefferquote %"] == 50
    assert s["Profitfaktor"] == pytest.approx(300 / 350, abs=1e-3)
    assert s["Max. Drawdown (Close zu Close) USD"] == 300      # Hoch +250, Tief -50
    assert s["Exit Ersatztag"] == 1


def test_daily_stop_simulate_hand_calculation():
    closes = np.array([100.0, 97.0, 94.9, 99.0, 101.0])
    win = [(0, 4, "Exit Ziel")]
    t = ds.simulate(closes, win, sl=5)
    assert (t.xi[0], t.out[0], t.signal[0]) == (2, 94.9, "Stop-Loss")    # Close 94,9 <= 95
    assert t.qty[0] == pytest.approx(100) and t.pnl[0] == pytest.approx(-510)
    t = ds.simulate(closes, win, tp=0.5)
    assert (t.xi[0], t.out[0], t.signal[0]) == (4, 101.0, "Take-Profit")
    t = ds.simulate(closes, win)
    assert (t.xi[0], t.signal[0]) == (4, "Exit Ziel")


def test_daily_stop_metrics_hand_calculation():
    t = pd.DataFrame({"pnl": [100.0, -50.0, 150.0], "signal": ["Exit Ziel", "Stop-Loss", "Take-Profit"]})
    m = ds.metrics(t, years=1.0, capital=1000.0)
    assert m["Netto USD"] == 200 and m["CAGR %"] == 20
    assert m["PF"] == 5 and m["Treffer %"] == pytest.approx(66.7)
    assert m["Max DD %"] == pytest.approx(round(50 / 1100 * 100, 1))
    assert (m["SL"], m["TP"]) == (1, 1)


def test_sma_filter_stats_hand_calculation():
    from friday_rush_backtest import Trade
    from sma_filter_test import stats
    ts = pd.Timestamp("2024-01-04", tz="UTC")
    trades = []
    for i, (p_in, p_out) in enumerate([(100, 110), (110, 99)]):
        t = Trade(i + 1, "Long", ts, ts, p_in, 10, exit_time=ts, exit_day=ts, exit_price=p_out)
        t.pnl, t.ret_pct = (p_out - p_in) * 10, (p_out / p_in - 1) * 100
        trades.append(t)
    s = stats(trades, 1000.0, years=2.0)
    assert s["Netto USD"] == -10 and s["Netto %"] == -1
    assert s["CAGR %"] == pytest.approx(round((0.99 ** 0.5 - 1) * 100, 2))
    assert s["Max DD USD"] == 110 and s["Max DD %"] == pytest.approx(10.0)    # 1100 -> 990


# ───────────────────────── Teil 2: Regeln auf echten Daten ─────────────────────────
def test_weekly_real_signals_follow_calendar(weekly_df):
    trades, _ = weekly.run_backtest(weekly_df, capital=CAPITAL)
    fri = weekly_df.set_index("time").friday
    months = set()
    for t in trades:
        fe, fx = fri[t.entry_time], fri[t.exit_time]
        assert 8 <= fe.day <= 14
        if t.exit_signal == "Exit Ziel":
            assert fx.day >= 24 and (fx.year, fx.month) == (fe.year, fe.month)
        else:
            assert t.exit_signal == "Exit Ersatz" and (fx.year, fx.month) != (fe.year, fe.month)
        key = (fe.year, fe.month)
        assert key not in months, f"zwei Einstiege im Monat {key}"
        months.add(key)


def test_daily_real_exit_is_next_valid_day(daily_df):
    trades, _ = daily.run_backtest(daily_df, daily.Settings())
    pos = {t: i for i, t in enumerate(daily_df.time)}
    valid = (daily_df.high != daily_df.low).values
    for t in trades:
        i, j = pos[t.entry_time], pos[t.exit_time]
        assert valid[j] and not valid[i + 1:j].any(), "Ausstieg nicht am nächsten gültigen Tag"


def test_sltp_variants_share_entries(weekly_df):
    """quant_analysis.py und robustness_sl6_tp5.py ordnen Trades verschiedener SL/TP-Varianten
    über die Position in der Liste einander zu. Das ist nur korrekt, wenn alle Varianten
    dieselben Einstiege haben."""
    ref = [t.entry_time for t in weekly.run_backtest(weekly_df)[0]]
    for sl, tp in [(1, 1), (6, 5), (10, 0), (0, 3), (4, 15)]:
        assert [t.entry_time for t in weekly.run_backtest(weekly_df, sl_pct=sl, tp_pct=tp)[0]] == ref
