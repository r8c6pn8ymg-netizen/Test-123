"""
Weekly Monatsmitte-Strategie (11. rein, 27. raus) mit Stop-Loss / Take-Profit
auf TAGES-Schlusskursen.

  * Einstieg/Ausstieg wie im Weekly-Code: Wochenschluss (letzter Handelstag der Woche),
    Woche über Freitag = Montag + 4 Tage; Einstieg, wenn Freitag 8.–14. (Zieltag 11 ±3),
    Ausstieg, wenn Freitag >= 24. im selben Monat oder neuer Monat.
  * SL/TP werden an JEDEM Tagesschluss nach dem Einstieg geprüft. Schließt ein Tag
    unter dem SL (bzw. über dem TP), wird zu diesem Tagesschluss verkauft.
    Die neue CSV enthält nur Schlusskurse -> Intraday-Stopps (High/Low) sind nicht möglich.
  * Ohne SL/TP ist das Ergebnis identisch mit dem Weekly-Backtest (Kontrolle).

  python daily_stop_backtest.py data/CAPITALCOM_SPX500_1D_close.csv
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

CAPITAL = 10_000.0
QTY_STEP = 0.1


def load_daily_close(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    t = pd.to_datetime(d["time"], unit="s", utc=True)
    d["day"] = (t + pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    d["monday"] = d["day"] - pd.to_timedelta(d["day"].dt.dayofweek, unit="D")
    d["friday"] = d["monday"] + pd.Timedelta(days=4)
    return d.reset_index(drop=True)


def trade_windows(d: pd.DataFrame, start="2010-01-01", end="2026-10-30", entry_day=11, exit_day=27):
    """Liefert (Einstiegs-Index, Index des planmäßigen Ausstiegs) je Trade."""
    weeks = d.groupby("monday").agg(last=("close", lambda s: s.index[-1]), friday=("friday", "first"))
    weeks["mkey"] = weeks.friday.dt.year * 12 + weeks.friday.dt.month
    # Datumsgrenze wie Pine: Öffnungszeit der Wochenkerze (Sonntagabend) zwischen Start und Ende
    open_time = weeks.index - pd.Timedelta(hours=1)
    weeks["in_range"] = (open_time >= pd.Timestamp(start) - pd.Timedelta(days=1)) & (weeks.index <= pd.Timestamp(end))
    out, pos = [], None
    for mon, w in weeks.iterrows():
        if pos is not None:
            if (w.friday.day >= exit_day - 3 and w.mkey == pos[1]) or w.mkey != pos[1]:
                out.append((pos[0], int(w["last"]), "Exit Ziel" if w.mkey == pos[1] else "Exit Ersatz"))
                pos = None
        elif entry_day - 3 <= w.friday.day <= entry_day + 3 and w.in_range:
            pos = (int(w["last"]), w.mkey)
    return out


def simulate(closes: np.ndarray, windows, sl=0.0, tp=0.0, capital=CAPITAL):
    eq, rows = capital, []
    for ei, xi, sig in windows:
        p_in = closes[ei]
        qty = math.floor(eq / p_in / QTY_STEP + 1e-9) * QTY_STEP
        p_out, why, day_i = closes[xi], sig, xi
        slp = p_in * (1 - sl / 100) if sl > 0 else -np.inf
        tpp = p_in * (1 + tp / 100) if tp > 0 else np.inf
        for k in range(ei + 1, xi + 1):
            c = closes[k]
            if c <= slp:
                p_out, why, day_i = c, "Stop-Loss", k
                break
            if c >= tpp:
                p_out, why, day_i = c, "Take-Profit", k
                break
        pnl = (p_out - p_in) * qty
        eq += pnl
        rows.append((ei, day_i, p_in, p_out, qty, pnl, (p_out / p_in - 1) * 100, why))
    return pd.DataFrame(rows, columns=["ei", "xi", "in", "out", "qty", "pnl", "ret", "signal"])


def metrics(t: pd.DataFrame, years: float, capital=CAPITAL) -> dict:
    eq = np.concatenate([[capital], capital + t.pnl.cumsum().values])
    dd_pct = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    dd_usd = (np.maximum.accumulate(eq) - eq).max()
    w, l = t.pnl[t.pnl > 0], t.pnl[t.pnl < 0]
    return {"Trades": len(t), "Treffer %": round(len(w) / len(t) * 100, 1),
            "Netto USD": round(t.pnl.sum(), 2), "CAGR %": round(((eq[-1] / capital) ** (1 / years) - 1) * 100, 2),
            "PF": round(w.sum() / -l.sum(), 2), "Max DD %": round(dd_pct, 1),
            "Netto/MaxDD": round(t.pnl.sum() / dd_usd, 2),
            "SL": int((t.signal == "Stop-Loss").sum()), "TP": int((t.signal == "Take-Profit").sum())}


def main() -> None:
    pd.set_option("display.width", 220)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/CAPITALCOM_SPX500_1D_close.csv"
    d = load_daily_close(path)
    closes = d.close.values
    print(f"Tagesdaten: {d.day.min().date()} bis {d.day.max().date()} ({len(d)} Tage, nur Schlusskurse)")

    periods = {"Gesamt 2010–2026": ("2010-01-01", "2026-10-30"),
               "IS 2010–2018": ("2010-01-01", "2018-12-31"),
               "OOS 2019–2026": ("2019-01-01", "2026-10-30")}
    wins = {k: trade_windows(d, *v) for k, v in periods.items()}
    yrs = {k: (pd.Timestamp(v[1]) - pd.Timestamp(v[0])).days / 365.25 for k, v in periods.items()}

    base = simulate(closes, wins["Gesamt 2010–2026"])
    print(f"\nKontrolle ohne SL/TP: {len(base)} Trades, Netto {base.pnl.sum():.2f} USD "
          f"(Weekly/TradingView: 202 Trades, 11212.27 USD)")

    sl_grid = [0.0] + list(np.arange(1.0, 10.01, 0.5))
    tp_grid = [0.0] + list(np.arange(1.0, 15.01, 0.5))
    res = {}
    for name, win in wins.items():
        rows = []
        for sl in sl_grid:
            for tp in tp_grid:
                m = metrics(simulate(closes, win, sl, tp), yrs[name])
                rows.append({"SL %": sl, "TP %": tp, **m})
        res[name] = pd.DataFrame(rows)
    full = res["Gesamt 2010–2026"]
    full.to_csv("data/daily_stop_grid_full.csv", index=False)

    b = full[(full["SL %"] == 0) & (full["TP %"] == 0)]
    print("\n═══ Basis ═══"); print(b.to_string(index=False))
    print("\n═══ Top 10 Netto ═══"); print(full.sort_values("Netto USD", ascending=False).head(10).to_string(index=False))
    print("\n═══ Top 10 Netto/MaxDD ═══"); print(full.sort_values("Netto/MaxDD", ascending=False).head(10).to_string(index=False))
    print("\n═══ Nur SL ═══"); print(full[full["TP %"] == 0].to_string(index=False))
    print("\n═══ Nur TP ═══"); print(full[full["SL %"] == 0].head(16).to_string(index=False))

    ins, oos = res["IS 2010–2018"], res["OOS 2019–2026"]
    ob = oos[(oos["SL %"] == 0) & (oos["TP %"] == 0)].iloc[0]
    ib = ins[(ins["SL %"] == 0) & (ins["TP %"] == 0)].iloc[0]
    print(f"\n═══ Walk-Forward (Basis IS {ib['Netto USD']:.0f} / OOS {ob['Netto USD']:.0f} USD, "
          f"OOS N/DD {ob['Netto/MaxDD']}) ═══")
    for crit in ["Netto USD", "Netto/MaxDD"]:
        for _, r in ins.sort_values(crit, ascending=False).head(5).iterrows():
            o = oos[(oos["SL %"] == r["SL %"]) & (oos["TP %"] == r["TP %"])].iloc[0]
            print(f"[{crit:11s}] SL {r['SL %']:4.1f} TP {r['TP %']:4.1f} | IS {r['Netto USD']:8.0f} N/DD {r['Netto/MaxDD']:5.2f}"
                  f" | OOS {o['Netto USD']:8.0f} N/DD {o['Netto/MaxDD']:5.2f}")

    both = ins.merge(oos, on=["SL %", "TP %"], suffixes=("_is", "_oos"))
    good = both[(both["Netto USD_is"] > ib["Netto USD"]) & (both["Netto USD_oos"] > ob["Netto USD"])]
    good = good.merge(full, on=["SL %", "TP %"])
    print(f"\n═══ In BEIDEN Hälften besser als Basis: {len(good)} Kombinationen ═══")
    print(good.sort_values("Netto/MaxDD", ascending=False)[["SL %", "TP %", "Netto USD_is", "Netto USD_oos",
          "Netto USD", "Max DD %", "Netto/MaxDD", "SL", "TP"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
