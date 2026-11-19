"""
Prüfwerkzeuge für die Backtests (Qualitätskontrolle).

Dieses Modul verändert keine Strategie. Es rechnet Kennzahlen unabhängig nach
und stellt Diagnosen bereit, die von den Tests in tests/ verwendet werden:

  * Renditen:        total_return, cagr, compound
  * Risiko:          sharpe_ratio, max_drawdown
  * Kosten:          CostModel, apply_costs
  * Overfitting:     probability_of_backtest_overfitting (CSCV, Bailey et al. 2015)
  * Data Snooping:   probabilistic_sharpe_ratio, deflated_sharpe_ratio
                     (Bailey & López de Prado 2014), bonferroni_threshold
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ───────────────────────── Renditen ─────────────────────────
def total_return(equity) -> float:
    """Gesamtrendite einer Equity-Kurve (letzter / erster Wert - 1)."""
    eq = np.asarray(equity, dtype=float)
    return eq[-1] / eq[0] - 1


def compound(returns) -> float:
    """Verkettete Rendite einfacher Einzelrenditen: Π(1 + r) - 1."""
    return float(np.prod(1 + np.asarray(returns, dtype=float)) - 1)


def cagr(equity, years: float) -> float:
    """Durchschnittliche jährliche Wachstumsrate."""
    return (1 + total_return(equity)) ** (1 / years) - 1


# ───────────────────────── Risiko ─────────────────────────
def sharpe_ratio(returns, periods_per_year: int = 52, rf_per_period: float = 0.0) -> float:
    """Annualisierte Sharpe Ratio (Stichproben-Std mit ddof=1)."""
    r = np.asarray(returns, dtype=float) - rf_per_period
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return r.mean() / sd * math.sqrt(periods_per_year)


def max_drawdown(equity) -> tuple[float, float]:
    """Maximaler Drawdown als (Betrag, Anteil vom jeweiligen Hochpunkt)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    dd_pct = np.divide(dd_abs, peak, out=np.zeros_like(eq), where=peak > 0)
    return float(dd_abs.max()), float(dd_pct.max())


# ───────────────────────── Kosten ─────────────────────────
@dataclass(frozen=True)
class CostModel:
    """Kostenannahmen je Trade (Hin- und Rückweg zusammen).

    spread_points:   Geld-Brief-Spanne in Indexpunkten, einmal pro Round-Trip bezahlt
    slippage_ticks:  Ausführungsabweichung je Order (Ein- und Ausstieg) in Ticks
    tick:            Tickgröße (CAPITALCOM:SPX500 = 0,1)
    commission_pct:  Provision in % des Ordervolumens je Order
    """
    spread_points: float = 0.5
    slippage_ticks: float = 1.0
    tick: float = 0.1
    commission_pct: float = 0.0

    def cost_per_trade(self, entry_price: float, exit_price: float, qty: float) -> float:
        points = self.spread_points + 2 * self.slippage_ticks * self.tick
        commission = self.commission_pct / 100 * (entry_price + exit_price) * qty
        return points * qty + commission


def apply_costs(trades, model: CostModel, capital: float) -> pd.DataFrame:
    """Rechnet Kosten nachträglich auf eine Trade-Liste an.

    Näherung: Die Positionsgröße bleibt wie im kostenfreien Lauf. Der Rückkopplungs-
    effekt (weniger Kapital -> kleinere Position) ist bei kleinen Kosten vernachlässigbar.
    Finanzierungskosten über Nacht (CFD) sind nicht enthalten.
    """
    rows = []
    equity = capital
    for t in trades:
        cost = model.cost_per_trade(t.entry_price, t.exit_price, t.qty)
        net = t.pnl - cost
        rows.append({"pnl_gross": t.pnl, "cost": cost, "pnl_net": net,
                     "ret_gross": t.pnl / equity, "ret_net": net / equity})
        equity += net
    out = pd.DataFrame(rows)
    out["equity_net"] = capital + out["pnl_net"].cumsum()
    return out


# ───────────────────────── Overfitting ─────────────────────────
def probability_of_backtest_overfitting(returns: np.ndarray, n_blocks: int = 8) -> float:
    """Probability of Backtest Overfitting per Combinatorially Symmetric Cross-Validation.

    returns: Matrix T x N (Perioden x Parametervarianten).
    Für jede Aufteilung der Blöcke in je eine Hälfte In-Sample/Out-of-Sample wird die
    beste In-Sample-Variante gewählt und ihr Rang Out-of-Sample bestimmt.
    PBO = Anteil der Aufteilungen, in denen diese Variante OOS unter dem Median liegt.
    """
    m = np.asarray(returns, dtype=float)
    t, n = m.shape
    blocks = np.array_split(np.arange(t), n_blocks)

    def sr(x):
        sd = x.std(axis=0, ddof=1)
        return np.divide(x.mean(axis=0), sd, out=np.full(x.shape[1], -np.inf), where=sd > 0)

    logits = []
    for is_ids in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_ids])
        best = int(np.argmax(sr(m[is_rows])))
        oos = sr(m[oos_rows])
        rank = stats.rankdata(oos)[best]            # 1 = schlechteste, n = beste
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(logits) <= 0))


# ───────────────────────── Data Snooping ─────────────────────────
def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """Wahrscheinlichkeit, dass die wahre (nicht annualisierte) Sharpe Ratio > sr_benchmark ist.
    Berücksichtigt Schiefe und Kurtosis der Renditen."""
    r = np.asarray(returns, dtype=float)
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Erwartete beste Sharpe Ratio unter n_trials Varianten ohne echte Überrendite."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR gegen die Sharpe Ratio, die man bei n_trials
    getesteten Varianten allein durch Zufall als Bestwert erwarten würde.
    sr_variance: Streuung der Sharpe Ratios über die Varianten; Standard 1/(T-1)
    (Varianz des Schätzers unter der Nullhypothese)."""
    r = np.asarray(returns, dtype=float)
    if sr_variance is None:
        sr_variance = 1 / (len(r) - 1)
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(n_trials, sr_variance))


def bonferroni_threshold(alpha: float, n_trials: int) -> float:
    return alpha / n_trials
