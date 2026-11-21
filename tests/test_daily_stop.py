"""daily_stop_backtest.py: Stop-Loss / Take-Profit auf Tages-Schlusskursen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds


def _sim(closes, sl=0.0, tp=0.0, capital=1_000.0):
    return ds.simulate(np.asarray(closes, float), [(0, len(closes) - 1, "Exit Ziel")], sl, tp, capital).iloc[0]


def test_no_stops_exits_at_planned_close():
    r = _sim([100, 101, 93, 99])
    assert (r.out, r.signal, r.qty) == (99, "Exit Ziel", 10.0)
    assert r.pnl == pytest.approx(-10)


def test_stop_loss_triggers_on_close_below_level():
    r = _sim([100, 101, 93, 99], sl=6)          # SL 94 -> Tag 2 schließt bei 93
    assert (r.out, r.signal, r.xi) == (93, "Stop-Loss", 2)
    assert r.pnl == pytest.approx(-70)           # Ausführung zum Close, nicht zum Stopkurs


def test_take_profit_triggers_on_close_above_level():
    r = _sim([100, 104, 106, 90], tp=5)          # TP 105 -> Tag 2
    assert (r.out, r.signal) == (106, "Take-Profit")


def test_level_touched_exactly_triggers():
    assert _sim([100, 94, 99], sl=6).signal == "Stop-Loss"
    assert _sim([100, 105, 99], tp=5).signal == "Take-Profit"


def test_entry_day_close_is_not_checked_against_stops():
    """Nur Tage NACH dem Einstieg zählen (kein Ausstieg am Einstiegstag)."""
    r = ds.simulate(np.array([100.0, 50.0, 100.0]), [(0, 2, "Exit Ziel")], sl=1)
    assert r.iloc[0].xi == 1


def test_compounding_across_trades():
    closes = np.array([100.0, 110.0, 100.0, 120.0])
    out = ds.simulate(closes, [(0, 1, "Exit Ziel"), (2, 3, "Exit Ziel")], capital=1_000)
    # 10 Kontrakte +100 -> 1.100 USD -> 11 Kontrakte * 20 = +220
    assert list(out.qty) == pytest.approx([10, 11]) and list(out.pnl) == pytest.approx([100, 220])


def test_trade_windows_calendar(tmp_path):
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
