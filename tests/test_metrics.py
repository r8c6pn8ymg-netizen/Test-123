"""Renditeberechnung, Sharpe Ratio und Maximum Drawdown.

Teil 1: die Prüfwerkzeuge gegen von Hand nachgerechnete Werte.
Teil 2: die Backtests rechnen intern dasselbe wie die unabhängigen Werkzeuge.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import backtest_checks as bc
import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly
from conftest import CAPITAL, weekly_returns


# ───────────────────────── Teil 1: Handrechnungen ─────────────────────────
def test_total_return_and_compounding():
    assert bc.total_return([100, 110, 99]) == pytest.approx(-0.01)
    # +10 % dann -10 % ergibt nicht 0, sondern -1 %
    assert bc.compound([0.10, -0.10]) == pytest.approx(-0.01)
    assert bc.compound([0.5, 0.5]) == pytest.approx(1.25)


def test_cagr():
    # Verdopplung in 10 Jahren = 2^(1/10) - 1 = 7,177 %
    assert bc.cagr([100, 200], 10) == pytest.approx(0.0717735, abs=1e-6)


def test_sharpe_ratio_hand_calculation():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    mean, sd = 0.00625, math.sqrt(((0.00375**2) + (0.01125**2) + (0.01375**2) + (0.00625**2)) / 3)
    assert bc.sharpe_ratio(r, periods_per_year=52) == pytest.approx(mean / sd * math.sqrt(52))
    assert math.isnan(bc.sharpe_ratio([0.01, 0.01, 0.01]))      # keine Streuung -> undefiniert


def test_sharpe_ratio_risk_free_rate_reduces_sharpe():
    r = np.random.default_rng(1).normal(0.002, 0.02, 500)
    assert bc.sharpe_ratio(r, rf_per_period=0.001) < bc.sharpe_ratio(r)


def test_max_drawdown_hand_calculation():
    # Hoch 120, Tief 90 -> 30 absolut, 25 % relativ (nicht vom Start, sondern vom Hochpunkt)
    dd_abs, dd_pct = bc.max_drawdown([100, 120, 90, 110, 130, 125])
    assert dd_abs == pytest.approx(30)
    assert dd_pct == pytest.approx(0.25)
    assert bc.max_drawdown([1, 2, 3]) == (0.0, 0.0)


# ───────────────────────── Teil 2: Backtests nachgerechnet ─────────────────────────
def _closed_bars(df, trades):
    return {t.no: df.index[df.time == t.exit_time][0] for t in trades}


@pytest.mark.parametrize("sl,tp", [(0, 0), (6, 5)])
def test_weekly_trade_pnl_and_return(weekly_df, sl, tp):
    trades, _ = weekly.run_backtest(weekly_df, sl_pct=sl, tp_pct=tp, capital=CAPITAL)
    assert trades
    for t in trades:
        assert t.pnl == pytest.approx((t.exit_price - t.entry_price) * t.qty)
        assert t.ret_pct == pytest.approx((t.exit_price / t.entry_price - 1) * 100)


def test_short_direction_inverts_pnl(weekly_df):
    longs, _ = weekly.run_backtest(weekly_df, direction="Long")
    shorts, _ = weekly.run_backtest(weekly_df, direction="Short")
    for t in shorts:
        assert t.pnl == pytest.approx((t.entry_price - t.exit_price) * t.qty)
    # erster Trade: gleiche Preise, entgegengesetztes Vorzeichen
    assert shorts[0].ret_pct == pytest.approx(-longs[0].ret_pct)


def test_equity_curve_matches_realised_pnl(weekly_df):
    """Am Ausstiegsbalken jedes Trades muss die Equity = Startkapital + Summe der G&V sein."""
    trades, curve = weekly.run_backtest(weekly_df, capital=CAPITAL)
    bars = _closed_bars(weekly_df, trades)
    cum = CAPITAL
    for t in trades:
        cum += t.pnl
        # ein neuer Einstieg am selben Balken erfolgt zum Close -> Mark-to-Market exakt 0
        assert curve.iloc[bars[t.no]] == pytest.approx(cum)


def test_position_sizing_uses_current_equity(weekly_df):
    """100 % des aktuellen Kapitals, abgerundet auf 0,1 Kontrakte, nie mehr als verfügbar."""
    trades, _ = weekly.run_backtest(weekly_df, capital=CAPITAL)
    equity = CAPITAL
    for t in trades:
        assert t.qty == pytest.approx(math.floor(equity / t.entry_price / 0.1 + 1e-9) * 0.1)
        assert t.qty * t.entry_price <= equity + 1e-6
        equity += t.pnl


def test_trade_returns_compound_to_total_return(weekly_df):
    """Die Summe der G&V entspricht der Gesamtrendite. Abweichung zur Verkettung der
    Trade-Renditen entsteht nur durch das Abrunden der Positionsgröße (< 0,1 Kontrakt)."""
    trades, _ = weekly.run_backtest(weekly_df, capital=CAPITAL)
    net = sum(t.pnl for t in trades) / CAPITAL
    exposure = []
    equity = CAPITAL
    for t in trades:
        exposure.append(t.qty * t.entry_price / equity)
        equity += t.pnl
    exact = bc.compound([e * t.ret_pct / 100 for e, t in zip(exposure, trades)])
    assert net == pytest.approx(exact, rel=1e-9)
    assert min(exposure) > 0.95                      # Rundungsverlust klein


def test_summary_max_drawdown_matches_independent_calc(daily_df):
    trades, _ = daily.run_backtest(daily_df, daily.Settings())
    tdf = daily.trades_df(trades)
    s = daily.summary(tdf, None, CAPITAL)
    realised = np.concatenate([[CAPITAL], CAPITAL + tdf["G&V USD"].cumsum().values])
    dd_abs, _ = bc.max_drawdown(realised)
    assert s["Max. Drawdown (Close zu Close) USD"] == pytest.approx(dd_abs, abs=0.01)
    assert s["Nettogewinn USD"] == pytest.approx(tdf["G&V USD"].sum(), abs=0.01)


def test_quant_analysis_maxdd_matches_independent_calc():
    import quant_analysis as qa                     # liest nur Daten, schreibt beim Import nichts
    eq = np.array([100, 120, 90, 110, 130, 125.0])
    assert qa.maxdd(eq) == pytest.approx(bc.max_drawdown(eq)[1])


def test_significance_sharpe_formula_matches_independent_calc(weekly_long_df):
    """significance_tests.py: Sharpe = Mittel / Std * sqrt(52) auf Wochenrenditen."""
    _, curve = weekly.run_backtest(weekly_long_df, sl_pct=6, tp_pct=5)
    r = weekly_returns(curve)
    assert r.mean() / r.std() * np.sqrt(52) == pytest.approx(bc.sharpe_ratio(r, 52))
    # Bezugswert aus reports/significance_tests.txt (In-Sample, SL 6 / TP 5): 0,543
    assert bc.sharpe_ratio(r, 52) == pytest.approx(0.543, abs=0.01)


def test_sharpe_is_annualised_with_correct_frequency(weekly_df):
    """Wochendaten -> Faktor sqrt(52). Ein falscher Faktor (z. B. 252) würde die Sharpe
    Ratio um sqrt(252/52) = 2,2 aufblähen."""
    _, curve = weekly.run_backtest(weekly_df)
    gaps = pd.Series(curve.index).diff().dropna().dt.days
    assert gaps.median() == 7
