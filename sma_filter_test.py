"""
SMA-Trendfilter-Test für beide Strategien.
Einstieg nur, wenn der Close der Einstiegskerze ÜBER der SMA liegt
(SMA auf dem Zeitrahmen der Strategie). Getestet: ohne Filter, SMA 20/30/50/200.

  python sma_filter_test.py
"""

from __future__ import annotations

import pandas as pd

import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly

SMAS = [0, 20, 30, 50, 200]
CAPITAL = 10_000.0


def stats(trades, capital, years) -> dict:
    tdf = daily.trades_df(trades)
    pnl = tdf["G&V USD"]
    eq = pd.concat([pd.Series([capital]), capital + pnl.cumsum()], ignore_index=True)
    dd_usd = (eq.cummax() - eq).max()
    dd_pct = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    end = eq.iloc[-1]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "Trades": len(tdf),
        "Trefferquote %": round(len(wins) / len(tdf) * 100, 1) if len(tdf) else 0,
        "Netto USD": round(pnl.sum(), 2),
        "Netto %": round(pnl.sum() / capital * 100, 1),
        "CAGR %": round(((end / capital) ** (1 / years) - 1) * 100, 2),
        "Profitfaktor": round(wins.sum() / -losses.sum(), 2) if len(losses) else float("inf"),
        "Ø Trade %": round(tdf["Rendite %"].mean(), 2) if len(tdf) else 0,
        "Max DD USD": round(dd_usd, 2),
        "Max DD %": round(dd_pct, 1),
        "Netto/MaxDD": round(pnl.sum() / dd_usd, 2) if dd_usd else float("inf"),
    }


def table(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "Filter"
    return df


def main() -> None:
    pd.set_option("display.width", 200)

    # ── Weekly: Monatsmitte 11./27. ──
    w = weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")
    first_sma200 = w.week_start.iloc[199].date()
    for label, start in [("2010-01-01 bis 2026-10-30 (wie TradingView)", "2010-01-01"),
                         (f"{first_sma200} bis 2026-10-30 (alle SMAs verfügbar)", str(first_sma200))]:
        years = (pd.Timestamp("2026-10-30") - pd.Timestamp(start)).days / 365.25
        rows = {}
        for n in SMAS:
            tr, _ = weekly.run_backtest(w, start=start, end="2026-10-30", capital=CAPITAL, sma_len=n)
            rows["ohne Filter" if n == 0 else f"SMA {n}"] = stats(tr, CAPITAL, years)
        print(f"\n═══ WEEKLY Monatsmitte, {label} ═══")
        print(table(rows).to_string())

    # ── Daily: Donnerstag/Freitag (CSV erst ab 2023) ──
    d = daily.load_csv("data/CAPITALCOM_SPX500_1D.csv")
    first_sma200 = d.trading_day.iloc[199].date()
    for label, start in [("2023-01-06 bis 2026-10-30", "2023-01-01"),
                         (f"{first_sma200} bis 2026-10-30 (alle SMAs verfügbar)", str(first_sma200))]:
        years = (pd.Timestamp("2026-10-30") - pd.Timestamp(start)).days / 365.25
        rows = {}
        for n in SMAS:
            s = daily.Settings(start=start, end="2026-10-30", initial_capital=CAPITAL, sma_len=n)
            tr, _ = daily.run_backtest(d, s)
            rows["ohne Filter" if n == 0 else f"SMA {n}"] = stats(tr, CAPITAL, years)
        print(f"\n═══ DAILY Donnerstag→Freitag, {label} ═══")
        print(table(rows).to_string())


if __name__ == "__main__":
    main()
