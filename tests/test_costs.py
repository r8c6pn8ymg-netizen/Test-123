"""Transaktionskosten und Slippage.

Wichtig: Alle bestehenden Backtests rechnen ohne Kosten. In friday_rush_backtest.Settings
gibt es zwar commission_pct und slippage_ticks, sie werden in run_backtest aber nicht
verwendet. Die Kosten werden deshalb hier nachträglich mit backtest_checks.apply_costs
angerechnet (Annahmen: conftest.REALISTIC_COSTS).
"""

from __future__ import annotations

import pytest

import backtest_checks as bc
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly
from conftest import CAPITAL, IS_END, IS_START, REALISTIC_COSTS


# ───────────────────────── Kostenmodell ─────────────────────────
class _T:
    def __init__(self, entry, exit_, qty):
        self.entry_price, self.exit_price, self.qty = entry, exit_, qty
        self.pnl = (exit_ - entry) * qty


def test_cost_per_trade_hand_calculation():
    m = bc.CostModel(spread_points=0.5, slippage_ticks=1, tick=0.1, commission_pct=0.01)
    # Punkte: 0,5 + 2 * 1 * 0,1 = 0,7 je Kontrakt -> 0,7 * 2 = 1,4
    # Provision: 0,01 % * (4000 + 4100) * 2 = 1,62
    assert m.cost_per_trade(4000, 4100, 2) == pytest.approx(1.4 + 1.62)


def test_zero_costs_change_nothing():
    trades = [_T(100, 110, 1), _T(110, 105, 1)]
    out = bc.apply_costs(trades, bc.CostModel(0, 0, 0.1, 0), 1000)
    assert (out.pnl_net == out.pnl_gross).all()
    assert out.equity_net.iloc[-1] == pytest.approx(1005)


def test_costs_are_monotonic():
    trades = [_T(100, 110, 1), _T(110, 105, 1), _T(105, 108, 2)]
    nets = [bc.apply_costs(trades, bc.CostModel(s, 1, 0.1), 1000).pnl_net.sum() for s in (0, 0.5, 1, 2)]
    assert all(a > b for a, b in zip(nets, nets[1:]))


def test_slippage_hits_both_entry_and_exit():
    one_side = bc.CostModel(spread_points=0, slippage_ticks=1, tick=0.1).cost_per_trade(100, 100, 1)
    assert one_side == pytest.approx(0.2)


# ───────────────────────── Bestehender Code ─────────────────────────
def test_daily_settings_costs_are_applied(daily_df):
    """Früher ein Befund (Kosten-Settings wurden ignoriert), seit dem Fix ein Regressionstest."""
    base, _ = daily.run_backtest(daily_df, daily.Settings())
    costly, _ = daily.run_backtest(daily_df, daily.Settings(commission_pct=0.1, slippage_ticks=5))
    assert sum(t.pnl for t in costly) < sum(t.pnl for t in base)
    assert [t.entry_time for t in costly] == [t.entry_time for t in base]


def _one_trade_df():
    """Do 100 -> Fr 110 (Kerzen öffnen 22:00 UTC am Vorabend, wie Capital.com)."""
    import pandas as pd
    times = pd.to_datetime(["2024-01-02 22:00", "2024-01-03 22:00", "2024-01-04 22:00"], utc=True)
    df = pd.DataFrame({"time": times,
                       "open": [99.0, 100.0, 101.0], "high": [101.0, 102.0, 111.0],
                       "low": [98.0, 99.0, 100.0], "close": [100.0, 100.0, 110.0]})
    df["trading_day"] = (df["time"] + pd.Timedelta(hours=6)).dt.normalize()
    return df


def test_daily_costs_hand_calculation():
    """Long, Einstieg Do-Close 100, Ausstieg Fr-Close 110, Kapital 1000 -> 10 Kontrakte.
    Slippage 2 Ticks à 0,1 = 0,2 Punkte je Order: Einstieg 100,2, Ausstieg 109,8.
    Provision 0,1 % je Order: 0,1 % * (100,2 + 109,8) * 10 = 2,10.
    G&V = (109,8 - 100,2) * 10 - 2,10 = 93,90 (ohne Kosten 100)."""
    df = _one_trade_df()
    s0 = daily.Settings(start="2024-01-01", end="2024-12-31", initial_capital=1000)
    (t0,), _ = daily.run_backtest(df, s0)
    assert (t0.qty, t0.entry_price, t0.exit_price, t0.pnl) == (pytest.approx(10), 100, 110, pytest.approx(100))
    s = daily.Settings(start="2024-01-01", end="2024-12-31", initial_capital=1000,
                       commission_pct=0.1, slippage_ticks=2, tick=0.1)
    (t,), curve = daily.run_backtest(df, s)
    assert t.qty == pytest.approx(10)
    assert t.entry_price == pytest.approx(100.2)
    assert t.exit_price == pytest.approx(109.8)
    assert t.commission == pytest.approx(2.10)
    assert t.pnl == pytest.approx(93.90)
    assert curve.iloc[-1] == pytest.approx(1093.90)


def test_daily_costs_short_slippage_direction():
    """Short: Slippage verschlechtert Einstieg nach unten und Ausstieg nach oben."""
    df = _one_trade_df()
    s = daily.Settings(direction="Short", start="2024-01-01", end="2024-12-31",
                       initial_capital=1000, slippage_ticks=1, tick=0.1)
    (t,), _ = daily.run_backtest(df, s)
    assert t.entry_price == pytest.approx(99.9)
    assert t.exit_price == pytest.approx(110.1)
    assert t.pnl == pytest.approx((99.9 - 110.1) * t.qty)


# ───────────────────────── Strategien nach Kosten ─────────────────────────
@pytest.fixture(scope="module")
def weekly_variants(weekly_df):
    return {name: weekly.run_backtest(weekly_df, sl_pct=sl, tp_pct=tp, start=IS_START, end=IS_END)[0]
            for name, (sl, tp) in {"Basis": (0, 0), "SL6/TP5": (6, 5)}.items()}


@pytest.mark.quality
@pytest.mark.parametrize("variant", ["Basis", "SL6/TP5"])
def test_weekly_profitable_after_costs(weekly_variants, variant):
    trades = weekly_variants[variant]
    k = bc.apply_costs(trades, REALISTIC_COSTS, CAPITAL)
    print(f"{variant}: brutto {k.pnl_gross.sum():.0f} USD, Kosten {k.cost.sum():.0f} USD, "
          f"netto {k.pnl_net.sum():.0f} USD")
    assert k.pnl_net.sum() > 0
    assert k.cost.sum() < 0.25 * k.pnl_gross.sum(), "Kosten fressen > 25 % des Bruttogewinns"


@pytest.mark.quality
def test_weekly_survives_double_costs(weekly_variants):
    """Puffer: auch bei doppelten Kosten noch profitabel."""
    double = bc.CostModel(REALISTIC_COSTS.spread_points * 2, REALISTIC_COSTS.slippage_ticks * 2,
                          REALISTIC_COSTS.tick, REALISTIC_COSTS.commission_pct * 2)
    for name, trades in weekly_variants.items():
        assert bc.apply_costs(trades, double, CAPITAL).pnl_net.sum() > 0, name


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: Die Donnerstag/Freitag-Strategie verdient brutto nur ~230 USD "
                          "und wird durch ~260 USD Kosten negativ.", strict=False)
def test_daily_profitable_after_costs(daily_df):
    trades, _ = daily.run_backtest(daily_df, daily.Settings())
    k = bc.apply_costs(trades, REALISTIC_COSTS, CAPITAL)
    print(f"Daily: brutto {k.pnl_gross.sum():.0f} USD, Kosten {k.cost.sum():.0f} USD, "
          f"netto {k.pnl_net.sum():.0f} USD")
    assert k.pnl_net.sum() > 0
