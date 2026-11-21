"""Data Snooping: Wie viel vom Ergebnis ist Glück durch das Ausprobieren vieler Varianten?

Laut significance_tests.py wurden etwa 2.385 Varianten getestet (SL/TP-Raster,
0,1er-Raster, SMA-Filter, Kalender-Landkarte). Bei so vielen Versuchen findet man
fast immer eine Variante, die zufällig gut aussieht.

  * Werkzeugtest: Die Deflated Sharpe Ratio entlarvt den Bestwert aus reinem Rauschen
  * Deflated Sharpe Ratio der Weekly-Strategie (SL 6 / TP 5) bei 2.385 Versuchen
  * Bonferroni-korrigierter t-Test der Trade-Renditen
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import backtest_checks as bc
import monthly_mid_weekly_backtest as weekly
from conftest import IS_END, IS_START, N_TRIALS, weekly_returns


# ───────────────────────── Werkzeugtests ─────────────────────────
def test_psr_basic_properties():
    rng = np.random.default_rng(3)
    good = rng.normal(0.003, 0.02, 900)
    assert bc.probabilistic_sharpe_ratio(good) > 0.95
    assert bc.probabilistic_sharpe_ratio(-good) < 0.05


def test_deflated_sharpe_falls_with_number_of_trials():
    r = np.random.default_rng(4).normal(0.002, 0.02, 900)
    values = [bc.deflated_sharpe_ratio(r, n) for n in (1, 10, 100, 1000)]
    assert values[0] == pytest.approx(bc.probabilistic_sharpe_ratio(r))
    assert all(a > b for a, b in zip(values, values[1:]))


def test_deflated_sharpe_exposes_best_of_random_strategies():
    """Je 200 Zufallsstrategien ohne Kante, 40 Wiederholungen: Der Bestwert sieht naiv
    fast immer signifikant aus, die Deflated Sharpe Ratio aber nur selten."""
    rng = np.random.default_rng(5)
    psr, dsr = [], []
    for _ in range(40):
        m = rng.normal(0, 0.02, (850, 200))
        best = m[:, np.argmax(m.mean(0) / m.std(0, ddof=1))]
        psr.append(bc.probabilistic_sharpe_ratio(best))
        dsr.append(bc.deflated_sharpe_ratio(best, n_trials=200))
    assert np.mean(np.array(psr) > 0.95) > 0.9               # naiv: "signifikant"
    assert np.mean(np.array(dsr) > 0.95) < 0.15              # korrigiert: selten
    assert 0.3 < np.median(dsr) < 0.7                        # um 50 % = reiner Zufall


def test_n_trials_matches_significance_script():
    """Liest N_TESTED direkt aus dem Quelltext. Ein Import von significance_tests würde
    statsmodels voraussetzen, das für diese Prüfung nicht nötig ist."""
    tree = ast.parse(Path("significance_tests.py").read_text(encoding="utf-8"))
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "N_TESTED" for t in n.targets))
    assert eval(compile(ast.Expression(node), "N_TESTED", "eval"), {"__builtins__": {}}) == N_TRIALS


# ───────────────────────── Strategie ─────────────────────────
@pytest.fixture(scope="module")
def strategy(weekly_df):
    trades, curve = weekly.run_backtest(weekly_df, sl_pct=6, tp_pct=5, start=IS_START, end=IS_END)
    return trades, weekly_returns(curve)


def test_naive_significance_before_correction(strategy):
    """Ohne Korrektur wirkt das Ergebnis signifikant (PSR > 95 %)."""
    _, r = strategy
    assert bc.probabilistic_sharpe_ratio(r) > 0.95


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: Bei ~2.385 getesteten Varianten liegt die Deflated Sharpe "
                          "Ratio deutlich unter 95 % – das Ergebnis ist mit Glück erklärbar.",
                   strict=False)
def test_deflated_sharpe_ratio(strategy):
    _, r = strategy
    dsr = bc.deflated_sharpe_ratio(r, N_TRIALS)
    print(f"Sharpe p.a. {bc.sharpe_ratio(r):.3f}, PSR {bc.probabilistic_sharpe_ratio(r):.3f}, "
          f"DSR {dsr:.3f} bei {N_TRIALS} Versuchen")
    assert dsr > 0.95


@pytest.mark.quality
@pytest.mark.xfail(reason="Befund: p ≈ 0,01 liegt weit über der Bonferroni-Schwelle "
                          "0,05 / 2.385 ≈ 0,00002.", strict=False)
def test_trade_returns_survive_bonferroni(strategy):
    trades, _ = strategy
    r = np.array([t.ret_pct / 100 for t in trades])
    t_stat, p_two = stats.ttest_1samp(r, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    print(f"t = {t_stat:.2f}, p (einseitig) = {p_one:.4f}")
    assert p_one < bc.bonferroni_threshold(0.05, N_TRIALS)
