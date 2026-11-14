"""
Weekly Monatsmitte-Strategie: Stop-Loss / Take-Profit – vollständige Prüfung nur auf Wochendaten.

1. Raster-Backtest: SL aus/1–10 %, TP aus/1–15 % (0,5er Schritte), 2010–2026,
   Intrabar-Logik wie TradingView (monthly_mid_weekly_backtest.run_backtest).
2. Monte-Carlo (gepaartes Bootstrap): 10.000 Mal werden 202 Trades mit Zurücklegen gezogen.
   Für jede Ziehung werden ALLE Varianten mit denselben Trades gerechnet, so dass direkt
   messbar ist, wie oft eine Variante die Basis schlägt.
3. Stabilität: Ergebnis je Zeitblock (2010–14, 2015–18, 2019–22, 2023–26) und
   Glättung über Nachbarwerte.

  python weekly_sltp_montecarlo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import monthly_mid_weekly_backtest as weekly

CAPITAL = 10_000.0
N_SIM = 10_000
SEED = 42
SL_GRID = [0.0] + [round(x, 1) for x in np.arange(1.0, 10.01, 0.5)]
TP_GRID = [0.0] + [round(x, 1) for x in np.arange(1.0, 15.01, 0.5)]
BLOCKS = [("2010", "2014"), ("2015", "2018"), ("2019", "2022"), ("2023", "2026")]


def returns_for(df, sl, tp):
    tr, _ = weekly.run_backtest(df, start="2010-01-01", end="2026-10-30", capital=CAPITAL, sl_pct=sl, tp_pct=tp)
    return tr


def max_dd(eq: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(eq, axis=-1)
    return ((peak - eq) / peak).max(axis=-1)


def main() -> None:
    pd.set_option("display.width", 230)
    df = weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")

    configs, R, rows = [], [], []
    entry_years = None
    for sl in SL_GRID:
        for tp in TP_GRID:
            tr = returns_for(df, sl, tp)
            r = np.array([t.ret_pct / 100 for t in tr])
            pnl = np.array([t.pnl for t in tr])
            if entry_years is None:
                entry_years = np.array([t.entry_day.year for t in tr])
            eq = np.concatenate([[CAPITAL], CAPITAL + pnl.cumsum()])
            sig = [t.exit_signal for t in tr]
            row = {"SL": sl, "TP": tp, "Netto USD": round(pnl.sum(), 2),
                   "Max DD %": round(max_dd(eq) * 100, 1),
                   "SL-Treffer": sig.count("Stop-Loss"), "TP-Treffer": sig.count("Take-Profit")}
            for a, b in BLOCKS:
                m = (entry_years >= int(a)) & (entry_years <= int(b))
                row[f"{a}-{b[2:]} %"] = round((np.prod(1 + r[m]) - 1) * 100, 1)
            rows.append(row)
            configs.append((sl, tp))
            R.append(r)
    R = np.array(R)                      # (Varianten, Trades)
    res = pd.DataFrame(rows)
    base_i = configs.index((0.0, 0.0))

    # ── Monte-Carlo, gepaart ──
    rng = np.random.default_rng(SEED)
    n_tr = R.shape[1]
    final = np.empty((len(configs), N_SIM))
    dd = np.empty((len(configs), N_SIM))
    for start in range(0, N_SIM, 1000):
        idx = rng.integers(0, n_tr, size=(1000, n_tr))
        sample = R[:, idx]                                   # (V, 1000, n_tr)
        eq = np.cumprod(1 + sample, axis=-1)
        eq = np.concatenate([np.ones(eq.shape[:-1] + (1,)), eq], axis=-1)
        final[:, start:start + 1000] = eq[..., -1]
        dd[:, start:start + 1000] = max_dd(eq)
    res["MC Median Endkapital"] = np.round(np.median(final, axis=1) * CAPITAL, 0)
    res["MC 5%-Endkapital"] = np.round(np.percentile(final, 5, axis=1) * CAPITAL, 0)
    res["MC P(Verlust) %"] = np.round((final < 1).mean(axis=1) * 100, 1)
    res["MC Median DD %"] = np.round(np.median(dd, axis=1) * 100, 1)
    res["MC 95%-DD %"] = np.round(np.percentile(dd, 95, axis=1) * 100, 1)
    res["MC P(> Basis) %"] = np.round((final > final[base_i]).mean(axis=1) * 100, 1)
    res["MC P(DD < Basis) %"] = np.round((dd < dd[base_i]).mean(axis=1) * 100, 1)
    res["MC Rendite/95DD"] = np.round((res["MC Median Endkapital"] / CAPITAL - 1) * 100 / res["MC 95%-DD %"], 2)
    blk = [f"{a}-{b[2:]} %" for a, b in BLOCKS]
    res["Blöcke > Basis"] = sum((res[c] > res.loc[base_i, c]).astype(int) for c in blk)
    res.to_csv("data/weekly_sltp_montecarlo.csv", index=False)

    cols = ["SL", "TP", "Netto USD", "Max DD %", "SL-Treffer", "TP-Treffer", *blk, "Blöcke > Basis",
            "MC Median Endkapital", "MC 5%-Endkapital", "MC Median DD %", "MC 95%-DD %",
            "MC P(> Basis) %", "MC P(DD < Basis) %", "MC Rendite/95DD"]
    print(f"Monte-Carlo: {N_SIM} Simulationen x {n_tr} Trades (gepaartes Bootstrap, Seed {SEED})\n")
    print("═══ Basis ═══"); print(res.loc[[base_i], cols].to_string(index=False))
    print("\n═══ Nur Stop-Loss (kein TP, Exit am letzten Tag) ═══")
    print(res[res.TP == 0][cols].to_string(index=False))
    print("\n═══ Nur Take-Profit ═══")
    print(res[(res.SL == 0) & (res.TP <= 9)][cols].to_string(index=False))
    print("\n═══ Top 15 nach MC Rendite/95%-DD (mit SL) ═══")
    print(res[res.SL > 0].sort_values("MC Rendite/95DD", ascending=False)[cols].head(15).to_string(index=False))
    print("\n═══ Top 15 nach MC P(> Basis) ═══")
    print(res.sort_values("MC P(> Basis) %", ascending=False)[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
