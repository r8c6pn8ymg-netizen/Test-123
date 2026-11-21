"""Transaktionskosten und Slippage.

Die Backtests rechnen standardmäßig ohne Kosten (wie die Pine-Strategien).
friday_rush_backtest.Settings wendet commission_pct und slippage_ticks an, wenn sie gesetzt
werden (Handrechnung: tests/test_signals.py). Für die Weekly-Strategie werden die Kosten
hier nachträglich mit backtest_checks.apply_costs angerechnet (Annahmen: conftest.REALISTIC_COSTS).
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
    """Behoben: Settings.commission_pct / slippage_ticks wurden früher stillschweigend ignoriert."""
    base, _ = daily.run_backtest(daily_df, daily.Settings())
    costly, _ = daily.run_backtest(daily_df, daily.Settings(commission_pct=0.1, slippage_ticks=5))
    assert sum(t.pnl for t in costly) < sum(t.pnl for t in base)


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
