"""
FX-Opens-Backtest – drei vorab festgelegte Varianten auf Weekly/Monthly Open

Regeln (vor dem Blick auf die Ergebnisse festgelegt, NICHT nachträglich optimiert):

  V1 "Weekly-Bias"   Jeden Tag zum Close: Close > Weekly Open -> Long,
                     Close < Weekly Open -> Short. Man ist immer im Markt;
                     die Position wird nur gedreht, wenn der Kurs die Seite wechselt.
  V2 "Weekly+Monthly" Wie V1, aber nur wenn Close auf derselben Seite von
                     Weekly UND Monthly Open liegt. Sonst flat.
  V3 "Mittwoch->Freitag" Mittwoch-Close: Richtung = Seite des Weekly Open.
                     Einstieg zum Mittwoch-Close, Ausstieg zum letzten Close der Woche
                     (normal Freitag). Haltedauer ~2 Tage.

Annahmen:
  * Startkapital 1.000, jede Position mit 100 % des aktuellen Kapitals, kein Hebel.
  * Kosten 0,3 % pro Trade (Round-Trip): 0,15 % beim Einstieg, 0,15 % beim Ausstieg.
    Ein Drehen Long->Short zählt als Ausstieg + Einstieg.
  * Signal und Ausführung zum Tages-Close (Signal nutzt nur Daten bis zu diesem Close).
  * Short-Ertrag = -Kursrendite (Swaps/Zinsen nicht berücksichtigt).
  * Entwicklung (In-Sample) bis 2017-12-31, Test (Out-of-Sample) ab 2018-01-01.
  * Daten: Yahoo Finance Tageskerzen.

Aufruf:
  python fx_opens_backtest.py [--cost 0.003] [--split 2018-01-01] [--out reports/fx_opens_results.txt]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/NZD": "AUDNZD=X",
    "AUD/CAD": "AUDCAD=X",
    "NZD/CAD": "NZDCAD=X",
}

INITIAL_CAPITAL = 1_000.0


def load_pair(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.dayofweek < 5]           # Wochenend-Kerzen verwerfen
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    return clean_bad_ticks(df)


def clean_bad_ticks(df: pd.DataFrame, max_move: float = 0.10) -> pd.DataFrame:
    """Entfernt offensichtliche Datenfehler: Tagesbewegungen > 10 %, die sich am
    nächsten Tag wieder umkehren (Yahoo-FX hat vereinzelt solche Ausreißer)."""
    r = df["Close"].pct_change()
    bad = (r.abs() > max_move) & (r.shift(-1).abs() > max_move) & (np.sign(r) != np.sign(r.shift(-1)))
    return df[~bad]


def add_opens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    week = df.index.to_period("W-FRI")
    month = df.index.to_period("M")
    df["week"] = week
    df["weekly_open"] = df.groupby(week)["Open"].transform("first")
    df["monthly_open"] = df.groupby(month)["Open"].transform("first")
    return df


# ───────────────────────── Signale ─────────────────────────
# Jede Funktion liefert die Position, die man vom Close dieses Tages bis zum
# Close des nächsten Handelstags hält (+1 Long, -1 Short, 0 flat).

def signal_v1(df: pd.DataFrame) -> pd.Series:
    return np.sign(df["Close"] - df["weekly_open"]).astype(float)


def signal_v2(df: pd.DataFrame) -> pd.Series:
    w = np.sign(df["Close"] - df["weekly_open"])
    m = np.sign(df["Close"] - df["monthly_open"])
    return w.where(w == m, 0.0).astype(float)


def signal_v3(df: pd.DataFrame) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    for _, g in df.groupby("week"):
        wed = g[g.index.dayofweek == 2]
        if wed.empty:
            continue
        d = wed.index[0]
        direction = float(np.sign(wed["Close"].iloc[0] - wed["weekly_open"].iloc[0]))
        last = g.index[-1]
        if last <= d or direction == 0:
            continue
        # gehalten vom Mittwoch-Close bis zum letzten Close der Woche
        hold_days = g.index[(g.index >= d) & (g.index < last)]
        pos.loc[hold_days] = direction
    return pos


STRATEGIES = {
    "V1 Weekly-Bias": signal_v1,
    "V2 Weekly+Monthly": signal_v2,
    "V3 Mittwoch->Freitag": signal_v3,
}


# ───────────────────────── Simulation ─────────────────────────
@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame


def simulate(df: pd.DataFrame, pos: pd.Series, cost_rt: float) -> Result:
    """pos[t] = Position von Close t bis Close t+1. Kosten: cost_rt/2 je Seite."""
    side_cost = cost_rt / 2
    close = df["Close"].to_numpy()
    p = pos.to_numpy()
    n = len(close)
    equity = np.empty(n)
    eq = INITIAL_CAPITAL
    prev = 0.0
    trades = []
    cur = None
    for t in range(n):
        # 1) Ertrag der Position, die seit gestern gehalten wird
        if t > 0 and prev != 0:
            eq *= 1 + prev * (close[t] / close[t - 1] - 1)
        # 2) Positionswechsel zum heutigen Close
        new = p[t] if t < n - 1 else 0.0       # letzter Tag: alles schließen
        if new != prev:
            if prev != 0:                       # Ausstieg
                eq *= 1 - side_cost
                cur.update(exit_date=df.index[t], exit_price=close[t], exit_equity=eq)
                trades.append(cur)
                cur = None
            if new != 0:                        # Einstieg
                eq *= 1 - side_cost
                cur = dict(direction="Long" if new > 0 else "Short",
                           entry_date=df.index[t], entry_price=close[t], entry_equity=eq / (1 - side_cost))
        equity[t] = eq
        prev = new
    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr["return_pct"] = (tr["exit_equity"] / tr["entry_equity"] - 1) * 100
        tr["days"] = (tr["exit_date"] - tr["entry_date"]).dt.days
    return Result(pd.Series(equity, index=df.index), tr)


def stats(res: Result, start=None, end=None) -> dict:
    eq = res.equity
    if start is not None:
        eq = eq[eq.index >= start]
    if end is not None:
        eq = eq[eq.index < end]
    tr = res.trades
    if not tr.empty:
        if start is not None:
            tr = tr[tr["entry_date"] >= start]
        if end is not None:
            tr = tr[tr["entry_date"] < end]
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0]
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return {
        "Start": eq.index[0].date(),
        "Ende": eq.index[-1].date(),
        "Endkapital*": INITIAL_CAPITAL * total,
        "Rendite %": (total - 1) * 100,
        "p.a. %": (total ** (1 / years) - 1) * 100 if years > 0 else np.nan,
        "Max DD %": dd * 100,
        "Sharpe": sharpe,
        "Trades": len(tr),
        "Trefferq. %": (tr["return_pct"] > 0).mean() * 100 if len(tr) else np.nan,
        "Ø Trade %": tr["return_pct"].mean() if len(tr) else np.nan,
        "Ø Tage": tr["days"].mean() if len(tr) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.003, help="Kosten je Trade (Round-Trip), 0.003 = 0,3 %%")
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--out", default="reports/fx_opens_results.txt")
    args = ap.parse_args()
    split = pd.Timestamp(args.split)

    rows = []
    for name, ticker in PAIRS.items():
        df = add_opens(load_pair(ticker))
        for sname, fn in STRATEGIES.items():
            pos = fn(df)
            for cost_label, cost in (("0,3 %", args.cost), ("ohne", 0.0)):
                res = simulate(df, pos, cost)
                for period, s, e in (("In-Sample", None, split), ("Out-of-Sample", split, None)):
                    row = {"Paar": name, "Strategie": sname, "Kosten": cost_label, "Zeitraum": period}
                    row.update(stats(res, s, e))
                    rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    fmt = out.copy()
    for c in ["Endkapital*", "Rendite %", "p.a. %", "Max DD %", "Trefferq. %"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:,.1f}")
    for c in ["Sharpe", "Ø Trade %", "Ø Tage"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")
    text = (
        f"FX-Opens-Backtest  |  Kosten {args.cost*100:.2f} % je Trade (Round-Trip)  |  "
        f"Split {split.date()}\n"
        "* Endkapital: Wert, auf den 1.000 im jeweiligen Zeitraum angewachsen wären.\n\n"
        + fmt.to_string(index=False)
    )
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    out.to_csv(args.out.replace(".txt", ".csv"), index=False)


if __name__ == "__main__":
    main()
