"""
Robustheitstest der finalen Strategie: Weekly Monatsmitte + SL 6 % / TP 5 %.
Alles andere unverändert. Raster in 0,1-%-Schritten:
  SL 4,0–8,0 %, TP 3,0–7,0 %  (41 x 41 = 1.681 Varianten)

Ausgabe:
  * Nettogewinn-Matrix um SL 6 / TP 5 (±1 %)
  * Anteil der Nachbarn, die die Basis (ohne SL/TP) schlagen, für ±0,3 / ±0,5 / ±1,0 %
  * Zeitblock-Stabilität und gepaartes Monte-Carlo für die Nachbarschaft ±0,5 %

  python robustness_sl6_tp5.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import monthly_mid_weekly_backtest as weekly

CAPITAL = 10_000.0
SL_GRID = np.round(np.arange(4.0, 8.001, 0.1), 1)
TP_GRID = np.round(np.arange(3.0, 7.001, 0.1), 1)
BLOCKS = [(2010, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
N_SIM, SEED = 10_000, 7


def run(df, sl, tp):
    tr, _ = weekly.run_backtest(df, start="2010-01-01", end="2026-10-30", capital=CAPITAL, sl_pct=sl, tp_pct=tp)
    return tr


def maxdd(eq):
    pk = np.maximum.accumulate(eq, axis=-1)
    return ((pk - eq) / pk).max(axis=-1)


def main() -> None:
    pd.set_option("display.width", 250)
    df = weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")

    base = run(df, 0, 0)
    base_net = sum(t.pnl for t in base)
    years = np.array([t.entry_day.year for t in base])
    base_r = np.array([t.ret_pct / 100 for t in base])
    base_blocks = [np.prod(1 + base_r[(years >= a) & (years <= b)]) - 1 for a, b in BLOCKS]
    base_eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum([t.pnl for t in base])])
    base_dd = maxdd(base_eq)

    rows, R = [], {}
    for sl in SL_GRID:
        for tp in TP_GRID:
            tr = run(df, sl, tp)
            pnl = np.array([t.pnl for t in tr]); r = np.array([t.ret_pct / 100 for t in tr])
            eq = np.concatenate([[CAPITAL], CAPITAL + pnl.cumsum()])
            blocks = [np.prod(1 + r[(years >= a) & (years <= b)]) - 1 for a, b in BLOCKS]
            sig = [t.exit_signal for t in tr]
            rows.append({"SL": sl, "TP": tp, "Netto": pnl.sum(), "DD %": maxdd(eq) * 100,
                         "Blöcke>Basis": sum(b > bb for b, bb in zip(blocks, base_blocks)),
                         "SL-Treffer": sig.count("Stop-Loss"), "TP-Treffer": sig.count("Take-Profit")})
            R[(sl, tp)] = r
    g = pd.DataFrame(rows)
    g.to_csv("data/robustness_sl6_tp5.csv", index=False)

    tgt = g[(g.SL == 6.0) & (g.TP == 5.0)].iloc[0]
    print(f"Basis ohne SL/TP: Netto {base_net:.2f} USD, Max DD {base_dd*100:.1f} %")
    print(f"Ziel SL 6,0 / TP 5,0: Netto {tgt.Netto:.2f} USD, Max DD {tgt['DD %']:.1f} %, "
          f"SL {tgt['SL-Treffer']}x, TP {tgt['TP-Treffer']}x, Blöcke>Basis {tgt['Blöcke>Basis']}/4\n")

    piv = g.pivot(index="SL", columns="TP", values="Netto")
    sub = piv.loc[5.0:7.0, 4.0:6.0]
    print(f"═══ Nettogewinn in Tsd. USD (Zeilen SL %, Spalten TP %), Basis = {base_net / 1000:.1f} ═══")
    print((sub / 1000).round(1).to_string())

    print("\n═══ Nachbarschaft um SL 6 / TP 5 ═══")
    for w in [0.3, 0.5, 1.0]:
        n = g[(abs(g.SL - 6.0) <= w + 1e-9) & (abs(g.TP - 5.0) <= w + 1e-9)]
        print(f"±{w:.1f} %: {len(n):4d} Varianten | schlagen Basis: {(n.Netto > base_net).mean()*100:5.1f} % | "
              f"Netto Median {n.Netto.median():8.0f}  Min {n.Netto.min():8.0f}  Max {n.Netto.max():8.0f} | "
              f"DD Median {n['DD %'].median():4.1f} %  | ≥3 Blöcke besser: {(n['Blöcke>Basis'] >= 3).mean()*100:5.1f} %")

    print("\n═══ Eine Größe fest, die andere variiert ═══")
    print("TP 5,0 fest, SL variiert:")
    print(g[g.TP == 5.0][["SL", "Netto", "DD %", "SL-Treffer", "Blöcke>Basis"]].round(1).to_string(index=False))
    print("\nSL 6,0 fest, TP variiert:")
    print(g[g.SL == 6.0][["TP", "Netto", "DD %", "TP-Treffer", "Blöcke>Basis"]].round(1).to_string(index=False))

    # gepaartes Monte-Carlo für die Nachbarschaft ±0,5
    keys = [(s, t) for s, t in R if abs(s - 6.0) <= 0.5 + 1e-9 and abs(t - 5.0) <= 0.5 + 1e-9]
    M = np.array([base_r] + [R[k] for k in keys])
    rng = np.random.default_rng(SEED)
    fin, dd = [], []
    for _ in range(N_SIM // 1000):
        idx = rng.integers(0, M.shape[1], (1000, M.shape[1]))
        eq = np.cumprod(1 + M[:, idx], axis=-1)
        eq = np.concatenate([np.ones(eq.shape[:-1] + (1,)), eq], axis=-1)
        fin.append(eq[..., -1]); dd.append(maxdd(eq))
    fin, dd = np.concatenate(fin, axis=1), np.concatenate(dd, axis=1)
    beat = (fin[1:] > fin[0]).mean(axis=1) * 100
    lower_dd = (dd[1:] < dd[0]).mean(axis=1) * 100
    i6 = keys.index((6.0, 5.0))
    print(f"\n═══ Monte-Carlo ({N_SIM} Simulationen), Nachbarschaft ±0,5 % ({len(keys)} Varianten) ═══")
    print(f"SL 6/TP 5: schlägt Basis in {beat[i6]:.1f} %, geringerer DD in {lower_dd[i6]:.1f} % der Simulationen; "
          f"Median-Endkapital {np.median(fin[i6+1])*CAPITAL:.0f} vs. Basis {np.median(fin[0])*CAPITAL:.0f}; "
          f"95%-DD {np.percentile(dd[i6+1],95)*100:.1f} % vs. {np.percentile(dd[0],95)*100:.1f} %")
    print(f"Nachbarschaft: P(> Basis) Median {np.median(beat):.1f} % (Min {beat.min():.1f}, Max {beat.max():.1f}); "
          f"P(DD < Basis) Median {np.median(lower_dd):.1f} %")


if __name__ == "__main__":
    main()
