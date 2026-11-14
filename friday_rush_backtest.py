"""
Friday SP500 Rush – 1:1-Nachbau der TradingView-Strategie (PineScript v6)
"SPX500 – Donnerstag Close rein, Freitag Close raus"

Regeln (wie im Pine-Code):
  * Einstieg zum Close eines echten Donnerstags (process_orders_on_close = true)
  * Ausstieg zum Close des nächsten gültigen Handelstags (normal Freitag,
    sonst "Ersatztag", z. B. Montag)
  * Flache Kerzen (High == Low) zählen optional als Feiertag (skip_flat)
  * Handelstag über time_tradingday (UTC), weil Capital.com-Kerzen
    am Vorabend (22:00/23:00 UTC) öffnen
  * Positionsgröße: 100 % des Kapitals, abgerundet auf die Mengenschrittweite
    (bei CAPITALCOM:SPX500 0,1 Kontrakte), Pyramiding 1
  * Keine Provision, keine Slippage, Hebel 1x

Aufruf:
  python friday_rush_backtest.py DATEN.csv [--start 2010-01-01] [--end 2026-10-30]
         [--direction Long|Short] [--no-skip-flat] [--capital 10000]
         [--tv TradingView-Export.xlsx]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ───────────────────────── Einstellungen ─────────────────────────
@dataclass
class Settings:
    direction: str = "Long"            # input "Richtung"
    skip_flat: bool = True             # input "Flache Kerzen als Feiertag behandeln"
    start: str = "2010-01-01"          # input "Startdatum"
    end: str = "2026-10-30"            # input "Enddatum"
    initial_capital: float = 10_000.0  # strategy(initial_capital)
    pct_of_equity: float = 100.0       # default_qty_value (percent_of_equity)
    qty_step: float = 0.1              # Mengenschrittweite des Symbols
    commission_pct: float = 0.0
    slippage_ticks: int = 0
    exchange_tz: str = "America/New_York"  # Zeitzone, in der timestamp() die Datums-Inputs liest


@dataclass
class Trade:
    no: int
    direction: str
    entry_time: pd.Timestamp
    entry_day: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: pd.Timestamp | None = None
    exit_day: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_signal: str = ""
    bars: int = 0
    pnl: float = 0.0
    ret_pct: float = 0.0
    run_up: float = 0.0
    drawdown: float = 0.0
    _hi: float = field(default=-math.inf, repr=False)
    _lo: float = field(default=math.inf, repr=False)


# ───────────────────────── Daten ─────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)  # Kerzen-Öffnungszeit
    # time_tradingday: Die Kerze öffnet am Vorabend (22/23 Uhr UTC) und gehört
    # zum Folgetag. +6 h landet sicher auf dem richtigen Kalendertag.
    df["trading_day"] = (df["time"] + pd.Timedelta(hours=6)).dt.normalize()
    return df.reset_index(drop=True)


def floor_step(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


# ───────────────────────── Strategie ─────────────────────────
def run_backtest(df: pd.DataFrame, s: Settings) -> tuple[list[Trade], pd.Series]:
    is_long = s.direction == "Long"
    sign = 1 if is_long else -1
    start = pd.Timestamp(s.start, tz=s.exchange_tz)
    end = pd.Timestamp(s.end, tz=s.exchange_tz)

    trades: list[Trade] = []
    open_tr: Trade | None = None
    entry_bar = -1
    equity = s.initial_capital         # realisiertes Kapital
    equity_curve = []

    for i, bar in df.iterrows():
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        trade_day = bar.trading_day
        is_thursday = trade_day.dayofweek == 3           # dayofweek.thursday
        is_friday = trade_day.dayofweek == 4
        is_holiday_bar = s.skip_flat and h == l
        is_valid_day = not is_holiday_bar
        in_range = start <= bar.time <= end

        # Pine wertet beide Blöcke mit der Position VOR den Fills dieses Balkens aus.
        pos_before = open_tr is not None

        # Excursion (MFE/MAE) läuft über alle Balken nach dem Einstiegsbalken
        if open_tr is not None and i > entry_bar:
            open_tr._hi = max(open_tr._hi, h)
            open_tr._lo = min(open_tr._lo, l)

        # ── Ausstieg: nächster gültiger Handelstag zum Close ──
        exit_now = pos_before and i > entry_bar and is_valid_day

        # ── Einstieg: echter Donnerstag zum Close ──
        enter_now = is_thursday and is_valid_day and in_range and not pos_before

        if exit_now:
            t = open_tr
            t.exit_time, t.exit_day, t.exit_price = bar.time, trade_day, c
            t.exit_signal = "Exit Freitag" if is_friday else "Exit Ersatztag"
            t.bars = i - entry_bar
            t.pnl = sign * (c - t.entry_price) * t.qty
            t.ret_pct = sign * (c / t.entry_price - 1) * 100
            if is_long:
                t.run_up = max(0.0, (t._hi - t.entry_price) * t.qty)
                t.drawdown = min(0.0, (t._lo - t.entry_price) * t.qty)
            else:
                t.run_up = max(0.0, (t.entry_price - t._lo) * t.qty)
                t.drawdown = min(0.0, (t.entry_price - t._hi) * t.qty)
            equity += t.pnl
            trades.append(t)
            open_tr = None

        if enter_now:
            qty = floor_step(equity * s.pct_of_equity / 100 / c, s.qty_step)
            if qty > 0:
                open_tr = Trade(len(trades) + 1, s.direction, bar.time, trade_day, c, qty)
                entry_bar = i

        # Equity zum Schlusskurs (für Drawdown "Schlusskurs zu Schlusskurs")
        mtm = equity + (sign * (c - open_tr.entry_price) * open_tr.qty if open_tr else 0.0)
        equity_curve.append((trade_day, mtm))

    curve = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve])
    return trades, curve


# ───────────────────────── Auswertung ─────────────────────────
def trades_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Nr": t.no, "Einstieg": t.entry_day.date(), "Preis rein": t.entry_price,
        "Ausstieg": t.exit_day.date(), "Preis raus": t.exit_price, "Signal": t.exit_signal,
        "Menge": round(t.qty, 4), "G&V USD": round(t.pnl, 2), "Rendite %": round(t.ret_pct, 2),
        "Run-up USD": round(t.run_up, 2), "Drawdown USD": round(t.drawdown, 2), "Balken": t.bars,
    } for t in trades])


def summary(tdf: pd.DataFrame, curve: pd.Series, capital: float) -> dict:
    pnl = tdf["G&V USD"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    # TradingView "Max. Drawdown (Schlusskurs zu Schlusskurs)" = auf Basis der
    # realisierten Equity nach jedem geschlossenen Trade (Startkapital als erster Hochpunkt)
    eq = pd.concat([pd.Series([0.0]), pnl.cumsum()], ignore_index=True)
    dd = (eq.cummax() - eq).max()
    return {
        "Trades": len(tdf),
        "Gewinner": len(wins), "Verlierer": len(losses), "Ausgeglichen": int((pnl == 0).sum()),
        "Trefferquote %": round(len(wins) / len(tdf) * 100, 2) if len(tdf) else 0,
        "Nettogewinn USD": round(pnl.sum(), 2),
        "Nettogewinn %": round(pnl.sum() / capital * 100, 2),
        "Bruttogewinn USD": round(wins.sum(), 2), "Bruttoverlust USD": round(-losses.sum(), 2),
        "Profitfaktor": round(wins.sum() / -losses.sum(), 3) if len(losses) else float("inf"),
        "Durchschn. Gewinn USD": round(wins.mean(), 2), "Durchschn. Verlust USD": round(-losses.mean(), 2),
        "Größter Gewinn USD": round(wins.max(), 2), "Größter Verlust USD": round(-losses.min(), 2),
        "Max. Drawdown (Close zu Close) USD": round(dd, 2),
        "Exit Ersatztag": int((tdf["Signal"] == "Exit Ersatztag").sum()),
    }


def load_tv(path: str) -> pd.DataFrame:
    t = pd.read_excel(path, sheet_name="Handelsgeschäfte")
    ein = t[t["Typ"].str.contains("Einstieg")].set_index("Trade-Nummer")
    aus = t[t["Typ"].str.contains("Ausstieg")].set_index("Trade-Nummer")
    return pd.DataFrame({
        "Einstieg": pd.to_datetime(ein["Datum und Uhrzeit"]).dt.date, "Preis rein": ein["Preis USD"],
        "Ausstieg": pd.to_datetime(aus["Datum und Uhrzeit"]).dt.date, "Preis raus": aus["Preis USD"],
        "Signal": aus["Signal"], "Menge": ein["Größe (Menge)"], "G&V USD": aus["Netto G&V USD"],
        "Run-up USD": aus["Positive Exkursion USD"], "Drawdown USD": aus["Negative Exkursion USD"],
        "Balken": aus["Dauer (Balken)"], "Kum G&V": aus["Kumulierter G&V USD"],
    }).reset_index()


def compare(py: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    m = py.merge(tv, on="Einstieg", how="outer", suffixes=("_py", "_tv"), indicator=True)
    cols = ["Preis rein", "Preis raus", "Menge", "G&V USD", "Run-up USD", "Drawdown USD", "Balken"]
    m["Abweichung"] = ""
    both = m["_merge"] == "both"
    for c in cols:
        diff = both & ~np.isclose(m[f"{c}_py"].astype(float), m[f"{c}_tv"].astype(float), atol=0.011)
        m.loc[diff, "Abweichung"] += c + "; "
    for c in ["Ausstieg", "Signal"]:
        diff = both & (m[f"{c}_py"] != m[f"{c}_tv"])
        m.loc[diff, "Abweichung"] += c + "; "
    m.loc[m["_merge"] == "left_only", "Abweichung"] = "nur in Python"
    m.loc[m["_merge"] == "right_only", "Abweichung"] = "nur in TradingView"
    return m


# ───────────────────────── Main ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--start", default=Settings.start)
    ap.add_argument("--end", default=Settings.end)
    ap.add_argument("--direction", choices=["Long", "Short"], default="Long")
    ap.add_argument("--no-skip-flat", action="store_true")
    ap.add_argument("--capital", type=float, default=Settings.initial_capital)
    ap.add_argument("--tv", help="TradingView-Export (.xlsx) zum Vergleich")
    ap.add_argument("--out", default="python_trades.csv")
    a = ap.parse_args()

    df = load_csv(a.csv)
    print(f"Daten: {df.trading_day.min().date()} bis {df.trading_day.max().date()} ({len(df)} Kerzen)")

    tv = load_tv(a.tv) if a.tv else None
    capital = a.capital
    if tv is not None:
        # TradingView startet 2010. Die CSV beginnt später. Damit die Positionsgrößen
        # 1:1 vergleichbar sind, startet Python mit dem TV-Kapital vor dem ersten
        # gemeinsamen Trade.
        first_day = df.trading_day.min().date()
        before = tv[tv["Einstieg"] < first_day]
        if len(before):
            capital = a.capital + before["Kum G&V"].iloc[-1]
            print(f"TV-Kapital vor {first_day}: {capital:.2f} USD (nach TV-Trade {before['Trade-Nummer'].iloc[-1]})")

    s = Settings(direction=a.direction, skip_flat=not a.no_skip_flat,
                 start=a.start, end=a.end, initial_capital=capital)
    trades, curve = run_backtest(df, s)
    py = trades_df(trades)
    py.to_csv(a.out, index=False)

    print("\n── Python-Backtest ──")
    for k, v in summary(py, curve, capital).items():
        print(f"{k:38s} {v}")

    if tv is not None:
        tv_win = tv[(tv["Einstieg"] >= df.trading_day.min().date())]
        tv_curve = pd.Series(dtype=float)
        print("\n── TradingView, gleicher Zeitraum ──")
        for k, v in summary(tv_win, tv_curve, capital).items():
            print(f"{k:38s} {v}")
        cmp = compare(py, tv_win)
        bad = cmp[cmp["Abweichung"] != ""]
        print(f"\n── Trade-für-Trade-Vergleich: {int((cmp['_merge'] == 'both').sum())} gemeinsame Trades, "
              f"{len(bad)} mit Abweichung ──")
        if len(bad):
            print(bad[["Einstieg", "Ausstieg_py", "Ausstieg_tv", "G&V USD_py", "G&V USD_tv", "Abweichung"]].to_string(index=False))

    print(f"\nTrade-Liste gespeichert: {a.out}")


if __name__ == "__main__":
    main()
