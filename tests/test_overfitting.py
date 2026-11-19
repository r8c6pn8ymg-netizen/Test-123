"""Overfitting: Hält das, was im Backtest optimiert wurde, auch auf unbekannten Daten?

  * Werkzeugtest: PBO erkennt reines Rauschen (≈ 50 %) und eine echte Kante (≈ 0 %)
  * PBO (Probability of Backtest Overfitting) auf dem SL/TP-Raster der Weekly-Strategie
  * Out-of-Sample 2000–2009: Sharpe > 0 und Vorteil von SL 6 / TP 5 bleibt erhalten
  * Parameterstabilität: Nachbarn von SL 6 / TP 5 sind nicht deutlich schlechter

Tests mit 'xfail' zeigen bekannte Schwächen der Strategie. Sie sind Befunde, keine
Programmfehler, und werden beim Testlauf als 'XFAIL' mit Begründung aufgelistet.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import backtest_checks as bc
import monthly_mid_weekly_backtest as weekly
from conftest import IS_END, IS_START, OOS_END, OOS_START, weekly_returns

SL_GRID = [0, 3, 4, 5, 6, 8, 10]
TP_GRID = [0, 3, 4, 5, 6, 8, 10]


# ───────────────────────── Werkzeugtests ─────────────────────────
def test_pbo_detects_pure_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.02, (800, 40))
    assert 0.3 < bc.probability_of_backtest_overfitting(noise, 8) < 0.7


def test_pbo_low_for_genuine_edge():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.02, (800, 40))
    m[:, 7] += 0.004                       # eine Variante mit echter, stabiler Überrendite
    assert bc.probability_of_backtest_overfitting(m, 8) < 0.1


# ───────────────────────── Strategie ─────────────────────────
@pytest.fixture(scope="module")
def sltp_grid_returns(weekly_df):
    cols = []
    for sl, tp in itertools.product(SL_GRID, TP_GRID):
        _, curve = weekly.run_backtest(weekly_df, sl_pct=sl, tp_pct=tp, start=IS_START, end=IS_END)
        cols.append(weekly_returns(curve).values)
    return np.column_stack(cols)


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: PBO ≈ 0,8 – die im Raster beste SL/TP-Kombination liegt "
                          "auf den jeweils anderen Datenhälften meist unter dem Median.",
                   strict=False)
def test_sltp_grid_probability_of_overfitting(sltp_grid_returns):
    pbo = bc.probability_of_backtest_overfitting(sltp_grid_returns, n_blocks=8)
    print(f"PBO SL/TP-Raster: {pbo:.2f}")
    assert pbo < 0.5


@pytest.fixture(scope="module")
def oos_returns(weekly_long_df):
    out = {}
    for name, (sl, tp) in {"Basis": (0, 0), "SL6/TP5": (6, 5)}.items():
        _, c_is = weekly.run_backtest(weekly_long_df, sl_pct=sl, tp_pct=tp, start=IS_START, end=IS_END)
        _, c_oos = weekly.run_backtest(weekly_long_df, sl_pct=sl, tp_pct=tp, start=OOS_START, end=OOS_END)
        c_oos = c_oos[c_oos.index <= pd.Timestamp(IS_START, tz="UTC")]
        out[name] = (weekly_returns(c_is, IS_START), weekly_returns(c_oos, OOS_START))
    return out


def test_in_sample_results_are_positive(oos_returns):
    """Sanity: In-Sample sind beide Varianten profitabel (Ausgangspunkt der Optimierung)."""
    for name, (r_is, _) in oos_returns.items():
        assert bc.sharpe_ratio(r_is) > 0.3, name


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: Out-of-Sample 2000–2009 ist die Sharpe Ratio ≤ 0 "
                          "(siehe reports/significance_tests.txt).", strict=False)
@pytest.mark.parametrize("variant", ["Basis", "SL6/TP5"])
def test_out_of_sample_sharpe_positive(oos_returns, variant):
    _, r_oos = oos_returns[variant]
    sr = bc.sharpe_ratio(r_oos)
    print(f"{variant}: Sharpe OOS {sr:.3f}")
    assert sr > 0


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: Der In-Sample-Vorteil von SL 6 / TP 5 gegenüber der Basis "
                          "kehrt sich Out-of-Sample um.", strict=False)
def test_sltp_improvement_holds_out_of_sample(oos_returns):
    sr = {k: (bc.sharpe_ratio(a), bc.sharpe_ratio(b)) for k, (a, b) in oos_returns.items()}
    assert sr["SL6/TP5"][0] > sr["Basis"][0]          # In-Sample besser ...
    assert sr["SL6/TP5"][1] > sr["Basis"][1]          # ... und Out-of-Sample auch?


@pytest.mark.quality
def test_parameter_neighbourhood_is_stable():
    """Aus data/robustness_sl6_tp5.csv (nur gelesen): Alle Nachbarn im Umkreis ±1 %
    müssen profitabel sein, ihr Median mindestens 70 % des Mittelpunkts erreichen."""
    rb = pd.read_csv("data/robustness_sl6_tp5.csv")
    center = rb[(rb.SL.round(1) == 6.0) & (rb.TP.round(1) == 5.0)].Netto.iloc[0]
    nb = rb[((rb.SL - 6).abs() <= 1 + 1e-9) & ((rb.TP - 5).abs() <= 1 + 1e-9)].Netto
    print(f"Mittelpunkt {center:.0f}, Median Nachbarn {nb.median():.0f}, Minimum {nb.min():.0f}")
    assert (nb > 0).all()
    assert nb.median() >= 0.7 * center
