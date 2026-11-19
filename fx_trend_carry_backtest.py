"""
FX Trend + Carry – Monatlicher Backtest, Regeln vorab festgelegt (nicht optimiert)

Entscheidung nur einmal im Monat, zum Close des letzten Handelstags des Monats.
Die Position wird dann den ganzen nächsten Monat gehalten. Bleibt das Signal
gleich, wird nicht gehandelt (keine Kosten).

  T  "Trend 12M"     Paar in den letzten 12 Monaten gestiegen -> Long, gefallen -> Short.
  C  "Carry"         Basiswährung mit höherem Leitzins als Gegenwährung -> Long, sonst Short.
  TC "Trend+Carry"   Nur handeln, wenn Trend und Carry in dieselbe Richtung zeigen, sonst flat.

Annahmen:
  * Startkapital 1.000, jede Position mit 100 % des Kapitals, kein Hebel.
  * Kosten 0,3 % je Trade (Round-Trip), 0,15 % je Seite; Drehen = Aus- + Einstieg.
  * Zinsertrag/-kosten (Swap) werden täglich verbucht: Long erhält
    (Zins Basis - Zins Gegenwährung) * Tage/365, Short zahlt ihn. Verwendet wird der
    Leitzins des Vormonats (kein Blick in die Zukunft).
  * Leitzinsen: BIS, Datensatz WS_CBPOL (monatlich). Lücken (Japan 2013-2016) werden
    mit dem letzten bekannten Wert aufgefüllt.
  * In-Sample bis 2017-12-31, Out-of-Sample ab 2018-01-01.
  * "Portfolio": alle 5 Paare gleichzeitig mit je 20 % des Kapitals (zusammen 100 %).

Aufruf:
  python fx_trend_carry_backtest.py [--cost 0.003] [--split 2018-01-01]
"""

from __future__ import annotations

import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

from fx_opens_backtest import INITIAL_CAPITAL, load_pair

# Paar -> (Yahoo-Ticker, Basis-Land, Gegen-Land) – Ländercodes wie bei der BIS
PAIRS = {
    "EUR/USD": ("EURUSD=X", "XM", "US"),
    "USD/JPY": ("JPY=X", "US", "JP"),
    "AUD/NZD": ("AUDNZD=X", "AU", "NZ"),
    "AUD/CAD": ("AUDCAD=X", "AU", "CA"),
    "NZD/CAD": ("NZDCAD=X", "NZ", "CA"),
}

BIS_URL = "https://stats.bis.org/api/v1/data/WS_CBPOL/M.US+XM+JP+AU+NZ+CA/all?format=csv"


def load_policy_rates() -> pd.DataFrame:
    with urllib.request.urlopen(BIS_URL, timeout=60) as r:
        raw = pd.read_csv(io.BytesIO(r.read()), usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    rates = raw.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    rates.index = pd.PeriodIndex(rates.index, freq="M")
    full = pd.period_range(rates.index.min(), rates.index.max(), freq="M")
    return rates.reindex(full).ffill()


# ───────────────────────── Signale (monatlich) ─────────────────────────
def monthly_signals(df: pd.DataFrame, diff: pd.Series) -> pd.DataFrame:
    """diff: Zinsdifferenz Basis-Gegen je Monat (Prozentpunkte, Stand Monatsende)."""
    month = df.index.to_period("M")
    mclose = df["Close"].groupby(month).last()
    trend = np.sign(mclose / mclose.shift(12) - 1)
    carry = np.sign(diff.reindex(mclose.index))
    both = trend.where(trend == carry, 0.0)
    sig = pd.DataFrame({"T Trend 12M": trend, "C Carry": carry, "TC Trend+Carry": both})
    return sig.fillna(0.0)


def daily_position(df: pd.DataFrame, msig: pd.Series) -> pd.Series:
    """Signal von Monat M gilt ab dem letzten Close von M bis zum letzten Close von M+1."""
    month = df.index.to_period("M")
    is_last = pd.Series(month, index=df.index) != pd.Series(month, index=df.index).shift(-1)
    pos = pd.Series(np.nan, index=df.index)
    pos[is_last] = msig.reindex(month[is_last.to_numpy()]).to_numpy()
    return pos.ffill().fillna(0.0)


# ───────────────────────── Simulation ─────────────────────────
def simulate(df: pd.DataFrame, pos: pd.Series, daily_diff: pd.Series, cost_rt: float):
    side_cost = cost_rt / 2
    close = df["Close"].to_numpy()
    p = pos.to_numpy()
    dd = daily_diff.to_numpy()
    dates = df.index
    n = len(close)
    equity = np.empty(n)
    eq, prev = INITIAL_CAPITAL, 0.0
    trades, cur = [], None
    for t in range(n):
        if t > 0 and prev != 0:
            days = (dates[t] - dates[t - 1]).days
            carry = prev * dd[t] / 100 * days / 365
            eq *= 1 + prev * (close[t] / close[t - 1] - 1) + carry
        new = p[t] if t < n - 1 else 0.0
        if new != prev:
            if prev != 0:
                eq *= 1 - side_cost
                cur.update(exit_date=dates[t], exit_equity=eq)
                trades.append(cur)
            cur = None
            if new != 0:
                cur = dict(direction="Long" if new > 0 else "Short", entry_date=dates[t], entry_equity=eq)
                eq *= 1 - side_cost
        equity[t] = eq
        prev = new
    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr["return_pct"] = (tr["exit_equity"] / tr["entry_equity"] - 1) * 100
        tr["days"] = (tr["exit_date"] - tr["entry_date"]).dt.days
    return pd.Series(equity, index=dates), tr


def stats(eq: pd.Series, tr: pd.DataFrame | None, start=None, end=None) -> dict:
    if start is not None:
        eq = eq[eq.index >= start]
    if end is not None:
        eq = eq[eq.index < end]
    if tr is not None and not tr.empty:
        tr = tr[(tr["entry_date"] >= (start or eq.index[0])) & (tr["entry_date"] < (end or eq.index[-1] + pd.Timedelta(days=1)))]
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0]
    has_tr = tr is not None and len(tr) > 0
    return {
        "Start": eq.index[0].date(),
        "Ende": eq.index[-1].date(),
        "Endkapital*": INITIAL_CAPITAL * total,
        "Rendite %": (total - 1) * 100,
        "p.a. %": (total ** (1 / years) - 1) * 100,
        "Max DD %": (eq / eq.cummax() - 1).min() * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "Trades": len(tr) if tr is not None else np.nan,
        "Trefferq. %": (tr["return_pct"] > 0).mean() * 100 if has_tr else np.nan,
        "Ø Trade %": tr["return_pct"].mean() if has_tr else np.nan,
        "Ø Tage": tr["days"].mean() if has_tr else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--out", default="reports/fx_trend_carry_results.txt")
    args = ap.parse_args()
    split = pd.Timestamp(args.split)

    rates = load_policy_rates()
    rows, curves = [], {}
    for name, (ticker, base, quote) in PAIRS.items():
        df = load_pair(ticker)
        diff_m = rates[base] - rates[quote]
        # für den Zinsertrag: Leitzins des Vormonats
        daily_diff = pd.Series(diff_m.shift(1).reindex(df.index.to_period("M")).to_numpy(), index=df.index).fillna(0.0)
        sig = monthly_signals(df, diff_m)
        # erst handeln, wenn 12 Monate Historie für den Trend vorliegen
        first = df.index[0] + pd.DateOffset(months=13)
        df = df[df.index >= first]
        daily_diff = daily_diff[df.index]
        for sname in sig.columns:
            pos = daily_position(df, sig[sname])
            for cost_label, cost in (("0,3 %", args.cost), ("ohne", 0.0)):
                eq, tr = simulate(df, pos, daily_diff, cost)
                curves[(sname, cost_label, name)] = eq
                for period, s, e in (("In-Sample", None, split), ("Out-of-Sample", split, None)):
                    row = {"Paar": name, "Strategie": sname, "Kosten": cost_label, "Zeitraum": period}
                    row.update(stats(eq, tr, s, e))
                    rows.append(row)

    # Portfolio: je 20 % in allen Paaren, täglicher Mittelwert der Renditen
    for sname in ["T Trend 12M", "C Carry", "TC Trend+Carry"]:
        for cost_label in ("0,3 %", "ohne"):
            rets = pd.concat({p: curves[(sname, cost_label, p)].pct_change() for p in PAIRS}, axis=1, sort=True).dropna()
            eq = INITIAL_CAPITAL * (1 + rets.mean(axis=1)).cumprod()
            for period, s, e in (("In-Sample", None, split), ("Out-of-Sample", split, None)):
                row = {"Paar": "Portfolio 5x20%", "Strategie": sname, "Kosten": cost_label, "Zeitraum": period}
                row.update(stats(eq, None, s, e))
                rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    fmt = out.copy()
    for c in ["Endkapital*", "Rendite %", "p.a. %", "Max DD %", "Trefferq. %"]:
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}")
    for c in ["Sharpe", "Ø Trade %", "Ø Tage"]:
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    fmt["Trades"] = fmt["Trades"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    text = (
        f"FX Trend + Carry (monatlich)  |  Kosten {args.cost*100:.2f} % je Trade  |  Zinsertrag/Swap enthalten  |  "
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
