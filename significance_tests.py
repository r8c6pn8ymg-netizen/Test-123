"""
Statistische Signifikanz der Weekly-Monatsmitte-Strategie (S&P 500).

Tests je Zeitraum (In-Sample 2010–2026, Out-of-Sample 2000–2009):
  1. t-Test: ist die durchschnittliche Trade-Rendite > 0?
  2. Bootstrap-Konfidenzintervall der durchschnittlichen Trade-Rendite
  3. Sharpe-Ratio (annualisiert) und ihr t-Wert
  4. Regression gegen den Markt (CAPM): Alpha, Beta, R², t- und p-Werte
  5. Permutationstest: Kalender vs. zufällige Einstiegswochen
  6. Mehrfachtest-Korrektur (Bonferroni) für die Anzahl getesteter Varianten

  python significance_tests.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import monthly_mid_weekly_backtest as weekly

RNG = np.random.default_rng(11)
df = weekly.load_weekly("data/CAPITALCOM_SPX500_1W_long.csv")
wk = df.set_index("week_start")

PERIODS = {"In-Sample 2010–2026": ("2010-01-01", "2026-10-30"),
           "Out-of-Sample 2000–2009": ("2000-01-03", "2009-10-30")}
VARIANTS = {"ohne SL/TP": (0.0, 0.0), "SL 6 / TP 5": (6.0, 5.0)}
N_TESTED = 600 + 1681 + 5 + 99   # SL/TP-Raster, 0,1er-Raster, SMA-Filter, Kalender-Landkarte


def analyse(start, end, sl, tp):
    tr, curve = weekly.run_backtest(df, start=start, end=end, sl_pct=sl, tp_pct=tp)
    r = np.array([t.ret_pct / 100 for t in tr])
    n = len(r)
    t_stat, p_two = stats.ttest_1samp(r, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    boot = RNG.choice(r, (10_000, n)).mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])

    # wöchentliche Renditen der Strategie (inkl. Wochen ohne Position) und des Marktes
    s0, s1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=30)
    c = curve[(curve.index >= s0) & (curve.index <= s1)]
    strat = c.pct_change().dropna()
    mkt = wk.close.reindex(c.index).pct_change().dropna()
    strat, mkt = strat.loc[mkt.index], mkt
    years = len(strat) / 52
    sharpe = strat.mean() / strat.std() * np.sqrt(52)
    sharpe_t = sharpe * np.sqrt(years)
    sharpe_mkt = mkt.mean() / mkt.std() * np.sqrt(52)

    ols = sm.OLS(strat.values, sm.add_constant(mkt.values)).fit(cov_type="HC1")
    alpha_w, beta = ols.params
    t_alpha, t_beta = ols.tvalues
    p_alpha, p_beta = ols.pvalues

    # Permutation: gleiche Haltedauern, zufällige Startwochen im selben Zeitraum
    closes = df.close.values
    valid = np.where((df.week_start >= s0) & (df.week_start <= pd.Timestamp(end, tz="UTC")))[0]
    holds = np.array([t.bars for t in tr])
    starts = RNG.choice(valid[valid < len(df) - 4], (10_000, n))
    perm = np.prod(1 + closes[starts + holds] / closes[starts] - 1, axis=1) - 1
    real = np.prod(1 + r) - 1
    p_perm = (perm >= real).mean()

    return {
        "Trades": n, "Ø Trade %": r.mean() * 100, "Std Trade %": r.std(ddof=1) * 100,
        "t-Wert (Ø>0)": t_stat, "p-Wert (einseitig)": p_one,
        "95%-KI Ø Trade %": f"{ci[0]*100:+.2f} bis {ci[1]*100:+.2f}",
        "Sharpe p.a.": sharpe, "Sharpe Markt p.a.": sharpe_mkt, "t-Wert Sharpe": sharpe_t,
        "Alpha p.a. %": alpha_w * 52 * 100, "t-Wert Alpha": t_alpha, "p-Wert Alpha": p_alpha,
        "Beta": beta, "t-Wert Beta": t_beta, "R²": ols.rsquared,
        "Gesamtrendite %": real * 100, "Zufall Median %": np.median(perm) * 100, "p-Wert Permutation": p_perm,
    }


def main():
    pd.set_option("display.width", 200)
    rows = {}
    for pname, (s, e) in PERIODS.items():
        for vname, (sl, tp) in VARIANTS.items():
            rows[f"{pname} | {vname}"] = analyse(s, e, sl, tp)
    res = pd.DataFrame(rows)
    fmt = res.copy().astype(object)
    for idx in res.index:
        fmt.loc[idx] = [v if isinstance(v, str) else (f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}") for v in res.loc[idx]]
    print(fmt.to_string())
    bonf = 0.05 / N_TESTED
    print(f"\nMehrfachtest: ca. {N_TESTED} getestete Varianten -> Bonferroni-Schwelle p < {bonf:.6f} "
          f"(entspricht etwa t > {stats.norm.ppf(1 - bonf):.2f})")
    r_is = np.array([t.ret_pct for t in weekly.run_backtest(df, start="2010-01-01", end="2026-10-30")[0]]) / 100
    need = (1.96 * r_is.std(ddof=1) / r_is.mean()) ** 2
    print(f"Benötigte Trades für t = 1,96 bei Ø {r_is.mean()*100:.2f} % und Std {r_is.std(ddof=1)*100:.2f} %: ca. {need:.0f} "
          f"(= {need/12:.0f} Jahre bei 12 Trades/Jahr)")
    fmt.to_csv("reports/significance_tests.csv")


if __name__ == "__main__":
    main()
