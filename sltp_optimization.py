"""
Stop-Loss / Take-Profit-Optimierung für die Weekly-Monatsmitte-Strategie (ohne SMA-Filter).

  * Raster: SL aus + 1..10 % (0,5er Schritte), TP aus + 1..15 % (0,5er Schritte)
  * Gesamtzeitraum 2010-01-01 bis 2026-10-30 (wie TradingView)
  * Robustheit: Optimierung 2010–2018 (In-Sample), Prüfung 2019–2026 (Out-of-Sample),
    außerdem Glättung über die Nachbarfelder (3x3), damit kein Zufallstreffer gewinnt

  python sltp_optimization.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import monthly_mid_weekly_backtest as weekly
from sma_filter_test import stats

CAPITAL = 10_000.0
SL_GRID = [0.0] + list(np.arange(1.0, 10.01, 0.5))
TP_GRID = [0.0] + list(np.arange(1.0, 15.01, 0.5))
PERIODS = {
    "Gesamt 2010–2026": ("2010-01-01", "2026-10-30"),
    "In-Sample 2010–2018": ("2010-01-01", "2018-12-31"),
    "Out-of-Sample 2019–2026": ("2019-01-01", "2026-10-30"),
}


def grid(df, start, end) -> pd.DataFrame:
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            tr, _ = weekly.run_backtest(df, start=start, end=end, capital=CAPITAL, sl_pct=sl, tp_pct=tp)
            st = stats(tr, CAPITAL, years)
            sig = pd.Series([t.exit_signal for t in tr])
            st.update(SL=sl, TP=tp, **{"SL-Treffer": int((sig == "Stop-Loss").sum()),
                                       "TP-Treffer": int((sig == "Take-Profit").sum())})
            rows.append(st)
    return pd.DataFrame(rows)


def smoothed(g: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mittelwert jedes Feldes mit seinen Nachbarn (nur echte SL/TP-Werte, ohne 'aus')."""
    piv = g[(g.SL > 0) & (g.TP > 0)].pivot(index="SL", columns="TP", values=col)
    sm = piv.rolling(3, center=True, min_periods=1).mean()
    sm = sm.T.rolling(3, center=True, min_periods=1).mean().T
    return sm


def label(v: float) -> str:
    return "aus" if v == 0 else f"{v:g} %"


def main() -> None:
    pd.set_option("display.width", 220)
    df = weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")
    cols = ["SL", "TP", "Trades", "Trefferquote %", "Netto USD", "CAGR %", "Profitfaktor",
            "Max DD %", "Netto/MaxDD", "SL-Treffer", "TP-Treffer"]

    results = {name: grid(df, *rng) for name, rng in PERIODS.items()}
    full = results["Gesamt 2010–2026"]
    full.to_csv("data/sltp_grid_full.csv", index=False)

    base = full[(full.SL == 0) & (full.TP == 0)]
    print("═══ Basis (ohne SL/TP), Gesamtzeitraum ═══")
    print(base[cols].to_string(index=False))

    print("\n═══ Top 10 nach Nettogewinn, Gesamtzeitraum ═══")
    print(full.sort_values("Netto USD", ascending=False)[cols].head(10).to_string(index=False))

    print("\n═══ Top 10 nach Netto / Max-Drawdown, Gesamtzeitraum ═══")
    print(full.sort_values("Netto/MaxDD", ascending=False)[cols].head(10).to_string(index=False))

    print("\n═══ Nur Stop-Loss (TP aus) ═══")
    print(full[full.TP == 0][cols].to_string(index=False))
    print("\n═══ Nur Take-Profit (SL aus) ═══")
    print(full[full.SL == 0][cols].to_string(index=False))

    # Robustheit: geglättete Netto/MaxDD-Fläche
    sm = smoothed(full, "Netto/MaxDD")
    sl_best, tp_best = np.unravel_index(np.nanargmax(sm.values), sm.shape)
    print(f"\n═══ Robusteste Zone (3x3 geglättet, Netto/MaxDD): SL {sm.index[sl_best]:g} %, "
          f"TP {sm.columns[tp_best]:g} %, Ø-Wert {sm.values[sl_best, tp_best]:.2f} ═══")

    # Walk-Forward: im In-Sample optimieren, im Out-of-Sample prüfen
    ins, oos = results["In-Sample 2010–2018"], results["Out-of-Sample 2019–2026"]
    print("\n═══ Walk-Forward: Bestes im In-Sample -> Ergebnis im Out-of-Sample ═══")
    for crit in ["Netto USD", "Netto/MaxDD"]:
        top = ins.sort_values(crit, ascending=False).head(5)
        for _, r in top.iterrows():
            o = oos[(oos.SL == r.SL) & (oos.TP == r.TP)].iloc[0]
            ob = oos[(oos.SL == 0) & (oos.TP == 0)].iloc[0]
            print(f"[{crit:11s}] SL {label(r.SL):6s} TP {label(r.TP):6s} | IS: Netto {r['Netto USD']:9.2f} "
                  f"N/DD {r['Netto/MaxDD']:5.2f} | OOS: Netto {o['Netto USD']:9.2f} N/DD {o['Netto/MaxDD']:5.2f} "
                  f"(Basis OOS: {ob['Netto USD']:.2f} / {ob['Netto/MaxDD']:.2f})")

    for name, g in results.items():
        b = g[(g.SL == 0) & (g.TP == 0)].iloc[0]
        better = g[(g["Netto USD"] > b["Netto USD"]) & (g["Max DD %"] < b["Max DD %"])]
        print(f"\n{name}: {len(better)} von {len(g)} Kombinationen schlagen die Basis bei Gewinn UND Drawdown")


if __name__ == "__main__":
    main()
