"""Signalgenerierung und Trade-Abwicklung auf künstlichen Daten mit bekanntem Ergebnis.

Jeder erwartete Wert ist im Kommentar von Hand hergeleitet.
"""

from __future__ import annotations

import pytest
from synth import write_daily_csv, write_weekly_csv

import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly

# Januar 2024: Mo 1., Di 2., Mi 3., Do 4., Fr 5., Mo 8. …
WEEK_JAN = [("2024-01-01", 100), ("2024-01-02", 101), ("2024-01-03", 102),
            ("2024-01-04", 103), ("2024-01-05", 105), ("2024-01-08", 104)]


def _daily(tmp_path, bars, **settings):
    df = daily.load_csv(str(write_daily_csv(tmp_path / "d.csv", bars)))
    s = daily.Settings(start="2024-01-01", end="2024-12-31", **settings)
    return daily.run_backtest(df, s)


# ───────────────────────── Donnerstag rein / Freitag raus ─────────────────────────
def test_daily_thursday_entry_friday_exit(tmp_path):
    trades, _ = _daily(tmp_path, WEEK_JAN)
    assert len(trades) == 1
    t = trades[0]
    assert str(t.entry_day.date()) == "2024-01-04" and t.entry_price == 103
    assert str(t.exit_day.date()) == "2024-01-05" and t.exit_price == 105
    assert t.qty == pytest.approx(97.0)              # floor(10000 / 103 = 97,087 auf 0,1)
    assert t.pnl == pytest.approx(2 * 97)            # (105 - 103) * 97 = 194
    assert t.ret_pct == pytest.approx((105 / 103 - 1) * 100)
    assert t.exit_signal == "Exit Freitag" and t.bars == 1
    # Exkursion nur auf dem Ausstiegsbalken (High 106, Low 104)
    assert t.run_up == pytest.approx((106 - 103) * 97)
    assert t.drawdown == 0.0


def test_daily_flat_friday_is_holiday_exit_next_valid_day(tmp_path):
    bars = [*WEEK_JAN[:4], ("2024-01-05", 105, "flat"), WEEK_JAN[5]]
    t = _daily(tmp_path, bars)[0][0]
    assert str(t.exit_day.date()) == "2024-01-08" and t.exit_price == 104
    assert t.exit_signal == "Exit Ersatztag" and t.bars == 2
    assert t.pnl == pytest.approx(1 * 97)


def test_daily_flat_friday_counts_when_skip_flat_off(tmp_path):
    bars = [*WEEK_JAN[:4], ("2024-01-05", 105, "flat"), WEEK_JAN[5]]
    t = _daily(tmp_path, bars, skip_flat=False)[0][0]
    assert str(t.exit_day.date()) == "2024-01-05" and t.exit_signal == "Exit Freitag"


def test_daily_flat_thursday_gives_no_entry(tmp_path):
    bars = [*WEEK_JAN[:3], ("2024-01-04", 103, "flat"), *WEEK_JAN[4:]]
    assert _daily(tmp_path, bars)[0] == []


def test_daily_short_direction(tmp_path):
    t = _daily(tmp_path, WEEK_JAN, direction="Short")[0][0]
    assert t.pnl == pytest.approx(-2 * 97)           # (103 - 105) * 97
    assert t.run_up == 0.0                           # Low 104 liegt über dem Einstieg
    assert t.drawdown == pytest.approx((103 - 106) * 97)


def test_daily_date_range_limits_entries(tmp_path):
    df = daily.load_csv(str(write_daily_csv(tmp_path / "d.csv", WEEK_JAN)))
    trades, _ = daily.run_backtest(df, daily.Settings(start="2024-01-05", end="2024-12-31"))
    assert trades == []


def test_daily_position_size_compounds(tmp_path):
    bars = [*WEEK_JAN, ("2024-01-11", 110), ("2024-01-12", 99)]
    trades, curve = _daily(tmp_path, bars)
    assert len(trades) == 2
    # nach Trade 1: 10.194 USD -> floor(10194 / 110 = 92,67) = 92,6 Kontrakte
    assert trades[1].qty == pytest.approx(92.6)
    assert trades[1].pnl == pytest.approx((99 - 110) * 92.6)
    assert curve.iloc[-1] == pytest.approx(10_000 + 194 + (99 - 110) * 92.6)


def test_daily_equity_curve_marks_open_position_to_market(tmp_path):
    _, curve = _daily(tmp_path, WEEK_JAN)
    # Do: Einstieg zum Close -> unverändert, Fr: realisiert +194, Mo: flach
    assert list(curve.round(6)) == [10_000, 10_000, 10_000, 10_000, 10_194, 10_194]


def test_daily_sma_filter_blocks_until_enough_history_and_below_sma(tmp_path):
    # SMA(3) am Do = (101 + 102 + 103) / 3 = 102 < 103 -> Einstieg erlaubt
    assert len(_daily(tmp_path, WEEK_JAN, sma_len=3)[0]) == 1
    # SMA(5) ist am Do (4. Kerze) noch na -> kein Einstieg
    assert _daily(tmp_path, WEEK_JAN, sma_len=5)[0] == []
    # Close unter dem SMA -> kein Einstieg
    falling = [("2024-01-01", 110), ("2024-01-02", 108), ("2024-01-03", 106),
               ("2024-01-04", 103), ("2024-01-05", 105), ("2024-01-08", 104)]
    assert _daily(tmp_path, falling, sma_len=3)[0] == []


def test_daily_commission_and_slippage_hand_calculation(tmp_path):
    """0,1 % Provision je Order, 2 Ticks (0,2 Punkte) Slippage je Order."""
    t = _daily(tmp_path, WEEK_JAN, commission_pct=0.1, slippage_ticks=2)[0][0]
    assert t.entry_price == pytest.approx(103.2)      # 103 + 0,2 (Kauf teurer)
    assert t.exit_price == pytest.approx(104.8)       # 105 - 0,2 (Verkauf billiger)
    assert t.qty == pytest.approx(97.0)               # Größe aus dem Signal-Close
    commission = 0.001 * 97 * (103.2 + 104.8)         # = 20,176
    assert t.commission == pytest.approx(commission)
    assert t.pnl == pytest.approx(1.6 * 97 - commission)   # 155,2 - 20,176 = 135,024
    assert t.ret_pct == pytest.approx(t.pnl / (103.2 * 97) * 100)


def test_daily_costs_zero_by_default_are_identical(tmp_path):
    a = _daily(tmp_path, WEEK_JAN)[0][0]
    b = _daily(tmp_path, WEEK_JAN, commission_pct=0.0, slippage_ticks=0)[0][0]
    assert (a.entry_price, a.exit_price, a.pnl, a.commission) == (b.entry_price, b.exit_price, b.pnl, 0.0)


# ───────────────────────── Weekly: 11. rein, 27. raus ─────────────────────────
# Freitage: 5.1., 12.1. (Einstieg 8.–14.), 19.1., 26.1. (>= 24. -> Exit Ziel), 2.2.,
#           9.2. (Einstieg), 16.2., 23.2. (< 24.), 1.3. (neuer Monat -> Exit Ersatz)
WEEKS = [("2024-01-01", 100), ("2024-01-08", 100), ("2024-01-15", 103), ("2024-01-22", 110),
         ("2024-01-29", 111), ("2024-02-05", 112), ("2024-02-12", 113), ("2024-02-19", 114),
         ("2024-02-26", 108)]


def _weekly(tmp_path, bars, **kw):
    df = weekly.load_weekly(str(write_weekly_csv(tmp_path / "w.csv", bars)))
    return weekly.run_backtest(df, start="2024-01-01", end="2024-12-31", **kw)


def test_weekly_entry_window_and_exit_rules(tmp_path):
    trades, curve = _weekly(tmp_path, WEEKS)
    assert [(str(t.entry_day.date()), str(t.exit_day.date()), t.exit_signal) for t in trades] == [
        ("2024-01-08", "2024-01-22", "Exit Ziel"),
        ("2024-02-05", "2024-02-26", "Exit Ersatz"),
    ]
    t1, t2 = trades
    assert t1.qty == pytest.approx(100.0) and t1.pnl == pytest.approx(1000.0) and t1.bars == 2
    # 11.000 USD / 112 = 98,21 -> 98,2 Kontrakte, (108 - 112) * 98,2 = -392,8
    assert t2.qty == pytest.approx(98.2) and t2.pnl == pytest.approx(-392.8) and t2.bars == 3
    assert curve.iloc[-1] == pytest.approx(10_000 + 1000 - 392.8)


@pytest.mark.parametrize("entry_day,expected_first", [(11, "2024-01-08"), (18, "2024-01-15")])
def test_weekly_entry_day_parameter(tmp_path, entry_day, expected_first):
    trades, _ = _weekly(tmp_path, WEEKS, entry_day=entry_day)
    assert str(trades[0].entry_day.date()) == expected_first


def _sltp_trade(tmp_path, bar, direction="Long", sl=6, tp=5):
    """Einstieg 8.1. zum Close 100, danach eine Prüfkerze (open, high, low, close)."""
    o, h, l, c = bar
    bars = [("2024-01-01", 100), ("2024-01-08", 100), ("2024-01-15", c, (o, h, l)), ("2024-01-22", 100)]
    trades, _ = _weekly(tmp_path, bars, direction=direction, sl_pct=sl, tp_pct=tp)
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
