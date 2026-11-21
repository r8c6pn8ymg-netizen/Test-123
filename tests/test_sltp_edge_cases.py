"""Grenzfälle der Stop-Loss/Take-Profit-Logik (übernommen aus claude/gracious-hamilton-qspy15).

Ergänzt tests/test_signals.py um: Kurslücke über den Stop, beide Marken in einer Kerze
(High zuerst / Gleichstand), Short-Seite, Kalenderausstieg ohne Treffer sowie die
Stop-Regeln und Kalenderfenster von daily_stop_backtest.py. Alle erwarteten Werte
sind im Kommentar von Hand hergeleitet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synth import write_weekly_csv

import daily_stop_backtest as ds
import monthly_mid_weekly_backtest as weekly


# ───────────────────────── Weekly: SL/TP innerhalb der Kerze ─────────────────────────
def _sltp_trade(tmp_path, bar, direction="Long", sl=6, tp=5):
    """Einstieg 8.1.2024 zum Close 100 (Freitag 12.), danach eine Prüfkerze (open, high, low, close)."""
    o, h, l, c = bar
    bars = [("2024-01-01", 100), ("2024-01-08", 100), ("2024-01-15", c, (o, h, l)), ("2024-01-22", 100)]
    df = weekly.load_weekly(str(write_weekly_csv(tmp_path / "w.csv", bars)))
    trades, _ = weekly.run_backtest(df, direction=direction, start="2024-01-01", end="2024-12-31",
                                    sl_pct=sl, tp_pct=tp)
    return trades[0]


def test_weekly_take_profit_inside_bar(tmp_path):
    # Low näher am Open (2 < 5) -> Pfad O -> L(99) -> H(106): SL 94 nicht, TP 105 erreicht
    t = _sltp_trade(tmp_path, (101, 106, 99, 104))
    assert (t.exit_signal, t.exit_price) == ("Take-Profit", 105)
    assert t.pnl == pytest.approx(5 * 100)
    assert t.run_up == pytest.approx(5 * 100)        # Exkursion endet am TP-Fill
    assert t.drawdown == pytest.approx(-1 * 100)


def test_weekly_gap_through_stop_fills_at_open(tmp_path):
    t = _sltp_trade(tmp_path, (93, 95, 90, 92))
    assert (t.exit_signal, t.exit_price) == ("Stop-Loss", 93)


def test_weekly_both_levels_in_bar_high_first(tmp_path):
    # |H - O| = 6 < |O - L| = 7 -> High zuerst -> TP
    t = _sltp_trade(tmp_path, (100, 106, 93, 99))
    assert (t.exit_signal, t.exit_price) == ("Take-Profit", 105)


def test_weekly_both_levels_in_bar_low_first_on_tie(tmp_path):
    # |H - O| = 7 = |O - L| -> nicht "näher" -> Low zuerst -> SL
    t = _sltp_trade(tmp_path, (100, 107, 93, 99))
    assert (t.exit_signal, t.exit_price) == ("Stop-Loss", 94)
    assert t.pnl == pytest.approx(-6 * 100)


def test_weekly_short_take_profit(tmp_path):
    # Short: SL 106, TP 95. Low näher (6 < 7) -> O -> L(94): TP 95
    t = _sltp_trade(tmp_path, (100, 107, 94, 99), direction="Short")
    assert (t.exit_signal, t.exit_price) == ("Take-Profit", 95)
    assert t.pnl == pytest.approx(5 * 100)


def test_weekly_no_level_hit_exits_by_calendar(tmp_path):
    t = _sltp_trade(tmp_path, (100, 104, 96, 102))
    assert t.exit_signal == "Exit Ziel" and t.exit_price == 100


# ───────────────────────── daily_stop_backtest: Stops auf Tagesschluss ─────────────────────────
def _sim(closes, sl=0.0, tp=0.0, capital=1_000.0):
    return ds.simulate(np.asarray(closes, float), [(0, len(closes) - 1, "Exit Ziel")], sl, tp, capital).iloc[0]


def test_daily_stop_executes_at_close_not_at_level():
    r = _sim([100, 101, 93, 99], sl=6)          # SL 94 -> Tag 2 schließt bei 93
    assert (r.out, r.signal, r.xi) == (93, "Stop-Loss", 2)
    assert r.pnl == pytest.approx(-70)


def test_daily_stop_level_touched_exactly_triggers():
    assert _sim([100, 94, 99], sl=6).signal == "Stop-Loss"
    assert _sim([100, 105, 99], tp=5).signal == "Take-Profit"


def test_daily_stop_entry_day_close_is_not_checked():
    """Nur Tage NACH dem Einstieg zählen (kein Ausstieg am Einstiegstag)."""
    r = ds.simulate(np.array([100.0, 50.0, 100.0]), [(0, 2, "Exit Ziel")], sl=1)
    assert r.iloc[0].xi == 1


def test_daily_stop_compounding_across_trades():
    closes = np.array([100.0, 110.0, 100.0, 120.0])
    out = ds.simulate(closes, [(0, 1, "Exit Ziel"), (2, 3, "Exit Ziel")], capital=1_000)
    # 10 Kontrakte +100 -> 1.100 USD -> 11 Kontrakte * 20 = +220
    assert list(out.qty) == pytest.approx([10, 11]) and list(out.pnl) == pytest.approx([100, 220])


def test_daily_stop_trade_windows_calendar(tmp_path):
    """Tagesdaten Jan–Feb 2024: Einstieg am Wochenschluss der Woche mit Freitag 8.–14.,
    Ausstieg in der Woche mit Freitag >= 24. bzw. im neuen Monat."""
    days = pd.bdate_range("2024-01-01", "2024-03-01")
    evening = pd.DatetimeIndex(days).tz_localize("UTC") - pd.Timedelta(hours=2)   # Vorabend 22:00 UTC
    t = [int(x.timestamp()) for x in evening]          # unabhängig von der Zeitauflösung (pandas 3: µs)
    p = tmp_path / "c.csv"
    pd.DataFrame({"time": t, "close": np.arange(len(days)) + 100.0}).to_csv(p, index=False)
    d = ds.load_daily_close(str(p))
    w = ds.trade_windows(d, start="2024-01-01", end="2024-12-31")
    got = [(str(d.day[e].date()), str(d.day[x].date()), s) for e, x, s in w]
    assert got == [("2024-01-12", "2024-01-26", "Exit Ziel"),
                   ("2024-02-09", "2024-03-01", "Exit Ersatz")]
