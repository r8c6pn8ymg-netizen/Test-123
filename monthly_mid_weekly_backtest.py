"""
SPX500 – Monatsmitte Weekly (11. rein, 27. raus)
1:1-Nachbau der TradingView-Strategie (PineScript v6) für den Wochenchart.

Regeln (wie im Pine-Code):
  * Wochenkerze -> Freitag der Handelswoche = time + 4 Tage 6 Stunden (UTC)
  * Einstieg zum Wochen-Close, wenn dieser Freitag im Bereich Zieltag ±3 liegt
    (Standard 11 -> Freitag zwischen 8. und 14.)
  * Ausstieg zum Wochen-Close, wenn der Freitag >= Ausstiegstag-3 im selben Monat
    liegt ("Exit Ziel", Standard 27 -> ab dem 24.) oder ein neuer Monat begonnen
    hat ("Exit Ersatz")
  * process_orders_on_close, 100 % des Kapitals (abgerundet auf 0,1),
    Pyramiding 1, keine Kosten, Datumsbereich gilt nur für den Einstieg

Aufruf:
  python monthly_mid_weekly_backtest.py data/CAPITALCOM_SPX500_1W.csv \
         --tv data/tradingview_trades_weekly.xlsx
"""

from __future__ import annotations

import argparse

import pandas as pd

from friday_rush_backtest import Trade, compare, floor_step, load_tv, summary, trades_df


def load_weekly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)        # Sonntagabend UTC
    df["week_start"] = (df["time"] + pd.Timedelta(hours=6)).dt.normalize()  # Montag (wie TV-Liste)
    df["friday"] = (df["time"] + pd.Timedelta(days=4, hours=6)).dt.normalize()
    return df.reset_index(drop=True)


def run_backtest(df: pd.DataFrame, direction="Long", entry_day=11, exit_day=27,
                 start="2010-01-01", end="2026-10-30", capital=10_000.0,
                 qty_step=0.1, exchange_tz="America/New_York"):
    sign = 1 if direction == "Long" else -1
    start_ts = pd.Timestamp(start, tz=exchange_tz)
    end_ts = pd.Timestamp(end, tz=exchange_tz)
    equity = capital
    trades: list[Trade] = []
    open_tr: Trade | None = None
    entry_bar = -1
    entry_month_key = None
    curve = []

    for i, bar in df.iterrows():
        f = bar.friday
        f_day, month_key = f.day, f.year * 12 + f.month
        is_entry_week = entry_day - 3 <= f_day <= entry_day + 3
        in_range = start_ts <= bar.time <= end_ts
        pos_before = open_tr is not None

        if open_tr is not None and i > entry_bar:
            open_tr._hi = max(open_tr._hi, bar.high)
            open_tr._lo = min(open_tr._lo, bar.low)

        exit_now, signal = False, ""
        if pos_before and i > entry_bar:
            is_exit_week = f_day >= exit_day - 3 and month_key == entry_month_key
            new_month = month_key != entry_month_key
            if is_exit_week or new_month:
                exit_now, signal = True, "Exit Ziel" if is_exit_week else "Exit Ersatz"

        enter_now = is_entry_week and in_range and not pos_before

        if exit_now:
            t, c = open_tr, bar.close
            t.exit_time, t.exit_day, t.exit_price, t.exit_signal = bar.time, bar.week_start, c, signal
            t.bars = i - entry_bar
            t.pnl = sign * (c - t.entry_price) * t.qty
            t.ret_pct = sign * (c / t.entry_price - 1) * 100
            if sign == 1:
                t.run_up = max(0.0, (t._hi - t.entry_price) * t.qty)
                t.drawdown = min(0.0, (t._lo - t.entry_price) * t.qty)
            else:
                t.run_up = max(0.0, (t.entry_price - t._lo) * t.qty)
                t.drawdown = min(0.0, (t.entry_price - t._hi) * t.qty)
            equity += t.pnl
            trades.append(t)
            open_tr = None

        if enter_now:
            qty = floor_step(equity / bar.close, qty_step)
            if qty > 0:
                open_tr = Trade(len(trades) + 1, direction, bar.time, bar.week_start, bar.close, qty)
                entry_bar, entry_month_key = i, month_key

        mtm = equity + (sign * (bar.close - open_tr.entry_price) * open_tr.qty if open_tr else 0.0)
        curve.append((bar.week_start, mtm))

    return trades, pd.Series([v for _, v in curve], index=[d for d, _ in curve])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--direction", choices=["Long", "Short"], default="Long")
    ap.add_argument("--entry-day", type=int, default=11)
    ap.add_argument("--exit-day", type=int, default=27)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-10-30")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--tv", help="TradingView-Export (.xlsx) zum Vergleich")
    ap.add_argument("--out", default="python_trades_weekly.csv")
    a = ap.parse_args()

    df = load_weekly(a.csv)
    print(f"Daten: {df.week_start.min().date()} bis {df.week_start.max().date()} ({len(df)} Wochenkerzen)")
    trades, curve = run_backtest(df, a.direction, a.entry_day, a.exit_day, a.start, a.end, a.capital)
    py = trades_df(trades)
    py.to_csv(a.out, index=False)

    print("\n── Python-Backtest ──")
    for k, v in summary(py, curve, a.capital).items():
        if k != "Exit Ersatztag":
            print(f"{k:38s} {v}")
    print(f"{'Exit Ersatz':38s} {int((py['Signal'] == 'Exit Ersatz').sum())}")

    if a.tv:
        tv = load_tv(a.tv)
        print("\n── TradingView ──")
        for k, v in summary(tv, pd.Series(dtype=float), a.capital).items():
            if not k.startswith("Max. Drawdown") and k != "Exit Ersatztag":
                print(f"{k:38s} {v}")
        print(f"{'Exit Ersatz':38s} {int((tv['Signal'] == 'Exit Ersatz').sum())}")
        cmp = compare(py, tv)
        bad = cmp[cmp["Abweichung"] != ""]
        print(f"\n── Trade-für-Trade-Vergleich: {int((cmp['_merge'] == 'both').sum())} gemeinsame Trades, "
              f"{len(bad)} mit Abweichung ──")
        if len(bad):
            print(bad[["Einstieg", "Ausstieg_py", "Ausstieg_tv", "G&V USD_py", "G&V USD_tv", "Abweichung"]].to_string(index=False))
        print("\nErsatz-Exits (Python):")
        print(py[py["Signal"] == "Exit Ersatz"][["Einstieg", "Ausstieg", "G&V USD"]].to_string(index=False))

    print(f"\nTrade-Liste gespeichert: {a.out}")


if __name__ == "__main__":
    main()
