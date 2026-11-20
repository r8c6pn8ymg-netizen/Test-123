"""Unit-Tests für die Rechen-Bausteine der Backtests (mit kleinen, von Hand prüfbaren Daten)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_stop_backtest as ds
import friday_rush_backtest as fr
import monthly_mid_weekly_backtest as mw
import robustness_sl6_tp5 as rob
import weekly_sltp_montecarlo as wmc


# ───────────────────────── Rundung ─────────────────────────
@pytest.mark.parametrize(
    ("x", "step", "expected"),
    [(10.0, 0.1, 10.0), (10.09, 0.1, 10.0), (10.1, 0.1, 10.1), (3.29999999999, 0.1, 3.3), (0.05, 0.1, 0.0)],
)
def test_floor_step_rounds_down_to_quantity_step(x, step, expected):
    assert fr.floor_step(x, step) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "expected"),
    [(100.05, 100.1), (100.15, 100.2), (100.25, 100.3), (100.04, 100.0), (-0.04, -0.0)],
)
def test_round_tick_is_half_up_like_pine(x, expected):
    # Pine math.round_to_mintick rundet kaufmännisch (0,5 -> auf), Python round() nicht
    assert mw.round_tick(x, 0.1) == pytest.approx(expected)


# ───────────────────────── Drawdown ─────────────────────────
@pytest.mark.parametrize("fn", [rob.maxdd, wmc.max_dd], ids=["robustness.maxdd", "montecarlo.max_dd"])
def test_max_drawdown(fn):
    eq = np.array([100.0, 120.0, 90.0, 130.0, 117.0])
    assert fn(eq) == pytest.approx(0.25)  # 120 -> 90
    assert fn(np.array([1.0, 2.0, 3.0])) == 0.0
    # vektorisiert: eine Zeile je Simulation
    both = fn(np.array([[100.0, 50.0, 100.0], [100.0, 100.0, 80.0]]))
    assert both == pytest.approx([0.5, 0.2])


# ───────────────────────── Tages-Stopp-Simulation ─────────────────────────
def test_simulate_without_stops_holds_until_planned_exit():
    closes = np.array([100.0, 101.0, 95.0, 110.0])
    t = ds.simulate(closes, [(0, 3, "Exit Ziel")], capital=1000.0)
    row = t.iloc[0]
    assert row.qty == pytest.approx(10.0)
    assert row.out == 110.0 and row.signal == "Exit Ziel"
    assert row.pnl == pytest.approx(100.0)


def test_simulate_stop_loss_and_take_profit_on_close():
    closes = np.array([100.0, 101.0, 94.0, 110.0])
    sl = ds.simulate(closes, [(0, 3, "Exit Ziel")], sl=5, capital=1000.0).iloc[0]
    assert (sl.signal, sl.xi, sl.out) == ("Stop-Loss", 2, 94.0)
    tp = ds.simulate(closes, [(0, 3, "Exit Ziel")], tp=1, capital=1000.0).iloc[0]
    assert (tp.signal, tp.xi, tp.out) == ("Take-Profit", 1, 101.0)


def test_simulate_compounds_equity_and_floors_quantity():
    closes = np.array([100.0, 200.0, 300.0, 300.0])
    t = ds.simulate(closes, [(0, 1, "a"), (2, 3, "b")], capital=1005.0)
    # 1005/100 = 10,05 -> 10,0 · nach +1000 Gewinn: 2005/300 = 6,68 -> abgerundet 6,6
    assert t.qty.tolist() == pytest.approx([10.0, 6.6])
    assert t.pnl.iloc[0] == pytest.approx(1000.0)


def test_metrics_basic_values():
    t = pd.DataFrame({"pnl": [100.0, -50.0, 200.0], "signal": ["Exit Ziel", "Stop-Loss", "Take-Profit"]})
    m = ds.metrics(t, years=1.0, capital=1000.0)
    assert m["Trades"] == 3
    assert m["Treffer %"] == pytest.approx(66.7)
    assert m["Netto USD"] == pytest.approx(250.0)
    assert m["PF"] == pytest.approx(6.0)
    assert m["Max DD %"] == pytest.approx(round(50 / 1100 * 100, 1))
    assert (m["SL"], m["TP"]) == (1, 1)


# ───────────────────────── Auswertung & Vergleich ─────────────────────────
def _trades(pnls, signals=None):
    n = len(pnls)
    return pd.DataFrame({
        "Einstieg": pd.date_range("2020-01-02", periods=n, freq="7D").date,
        "Preis rein": [100.0] * n, "Ausstieg": pd.date_range("2020-01-03", periods=n, freq="7D").date,
        "Preis raus": [100.0 + p for p in pnls], "Signal": signals or ["Exit"] * n, "Menge": [1.0] * n,
        "G&V USD": pnls, "Run-up USD": [0.0] * n, "Drawdown USD": [0.0] * n, "Balken": [1] * n,
    })


def test_summary_counts_and_drawdown():
    s = fr.summary(_trades([10.0, -30.0, 5.0, 0.0]), pd.Series(dtype=float), 1000.0)
    assert (s["Trades"], s["Gewinner"], s["Verlierer"], s["Ausgeglichen"]) == (4, 2, 1, 1)
    assert s["Nettogewinn USD"] == pytest.approx(-15.0)
    assert s["Profitfaktor"] == pytest.approx(0.5)
    assert s["Max. Drawdown (Close zu Close) USD"] == pytest.approx(30.0)


def test_compare_detects_equal_and_differing_trades():
    py, tv = _trades([10.0, -5.0]), _trades([10.0, -5.0])
    assert (fr.compare(py, tv)["Abweichung"] == "").all()

    tv.loc[1, "G&V USD"] = -6.0
    extra = _trades([1.0, 2.0, 3.0]).iloc[[2]]
    m = fr.compare(py, pd.concat([tv, extra], ignore_index=True))
    assert "G&V USD" in m.loc[m["Einstieg"] == py.loc[1, "Einstieg"], "Abweichung"].iloc[0]
    assert (m["Abweichung"] == "nur in TradingView").sum() == 1


def test_compare_tolerates_cent_rounding():
    py, tv = _trades([10.0]), _trades([10.0])
    tv.loc[0, "G&V USD"] = 10.01
    assert (fr.compare(py, tv)["Abweichung"] == "").all()
