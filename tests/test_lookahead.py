"""Look-ahead Bias: Eine Entscheidung zum Zeitpunkt t darf nur Daten bis t verwenden.

Prüfmethoden:
  1. Abschneiden: Ein Backtest auf den Daten bis Balken k muss für alle Einstiege und
     abgeschlossenen Trades vor k exakt dasselbe liefern wie der Lauf auf allen Daten.
  2. Zukunft verfälschen: Werden alle Balken ab k durch Zufallswerte ersetzt, dürfen
     sich Einstiege, Trades und Equity vor k nicht ändern.
     Die Schnitte liegen direkt hinter Ein- und Ausstiegsbalken (siehe _cut_points);
     eine Gegenprobe mit absichtlichem Zukunftsblick stellt sicher, dass das greift.
  3. Ausführungspreise: Einstieg genau zum Close des Signalbalkens, Ausstieg zum Close
     oder (bei SL/TP) innerhalb der Spanne des Ausstiegsbalkens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import friday_rush_backtest as daily
import monthly_mid_weekly_backtest as weekly

WEEKLY_VARIANTS = {
    "ohne SL/TP": dict(),
    "SL 6 / TP 5": dict(sl_pct=6, tp_pct=5),
    "SMA-Filter 20": dict(sma_len=20),
    "Short": dict(direction="Short", sl_pct=4, tp_pct=3),
}
DAILY_VARIANTS = {
    "Standard": daily.Settings(),
    "SMA-Filter 50": daily.Settings(sma_len=50),
}


def _key(t):
    return (t.entry_time, t.entry_price, t.qty, t.exit_time, t.exit_price, t.exit_signal, round(t.pnl, 9))


def _closed_before(trades, ts):
    return [_key(t) for t in trades if t.exit_time < ts]


def _scramble_future(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Ersetzt OHLC ab Zeile k durch plausible, aber zufällige Kurse."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df) - k
    base = df["close"].iloc[k - 1] * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    o = base * np.exp(rng.normal(0, 0.01, n))
    c = base * np.exp(rng.normal(0, 0.01, n))
    out.loc[out.index[k:], "open"] = o
    out.loc[out.index[k:], "close"] = c
    out.loc[out.index[k:], "high"] = np.maximum(o, c) * (1 + rng.uniform(0, 0.03, n))
    out.loc[out.index[k:], "low"] = np.minimum(o, c) * (1 - rng.uniform(0, 0.03, n))
    return out


def _entry_key(t):
    return (t.entry_time, t.entry_price, t.qty)


def _cut_points(df, trades, n_trades=12):
    """Schnitte direkt nach Einstiegs- und Ausstiegsbalken ausgewählter Trades.
    Ein Zukunftsblick von nur einem Balken verändert dann genau die Entscheidung
    am letzten Balken vor dem Schnitt und wird sicher erkannt."""
    idx = {t: i for i, t in enumerate(df.time)}
    picks = [trades[i] for i in np.linspace(0, len(trades) - 1, n_trades).astype(int)]
    cuts = {idx[t.entry_time] + 1 for t in picks} | {idx[t.exit_time] + 1 for t in picks}
    return sorted(k for k in cuts if 10 < k < len(df))


def find_lookahead(run, df) -> list[str]:
    """Liefert eine Liste von Verstößen (leer = kein Look-ahead gefunden).
    run: Funktion df -> (trades, equity_curve)."""
    full, curve_full = run(df)
    problems = []
    for k in _cut_points(df, full):
        cut = df.time.iloc[k]
        exp_closed = _closed_before(full, cut)
        exp_entries = [_entry_key(t) for t in full if t.entry_time < cut]
        for label, data in (("abgeschnitten", df.iloc[:k].reset_index(drop=True)),
                            ("Zukunft verfälscht", _scramble_future(df, k, seed=k))):
            got, curve = run(data)
            if _closed_before(got, cut) != exp_closed:
                problems.append(f"{label} ab {cut.date()}: abgeschlossene Trades weichen ab")
            # (Beim Abschneiden gibt run_backtest eine am Ende offene Position nicht zurück,
            #  daher werden Einstiege nur gegen die verfälschte Zukunft verglichen.)
            if label == "Zukunft verfälscht" and \
                    [_entry_key(t) for t in got if t.entry_time < cut] != exp_entries:
                problems.append(f"{label} ab {cut.date()}: Einstiege weichen ab")
            if not np.allclose(curve.values[:k], curve_full.values[:k]):
                problems.append(f"{label} ab {cut.date()}: Equity-Kurve vor dem Schnitt weicht ab")
    return problems


@pytest.mark.parametrize("variant", WEEKLY_VARIANTS)
def test_weekly_no_lookahead(weekly_df, variant):
    kw = WEEKLY_VARIANTS[variant]
    assert find_lookahead(lambda d: weekly.run_backtest(d, **kw), weekly_df) == []


@pytest.mark.parametrize("variant", DAILY_VARIANTS)
def test_daily_no_lookahead(daily_df, variant):
    s = DAILY_VARIANTS[variant]
    assert find_lookahead(lambda d: daily.run_backtest(d, s), daily_df) == []


def test_detector_catches_one_bar_lookahead(weekly_df):
    """Gegenprobe: Eine absichtlich betrügende Variante (Einstieg nur, wenn der
    NÄCHSTE Close höher ist) muss als Look-ahead erkannt werden."""
    def cheating(d):
        trades, curve = weekly.run_backtest(d)
        nxt = d.close.shift(-1).values
        pos = {t: i for i, t in enumerate(d.time)}
        return [t for t in trades if nxt[pos[t.entry_time]] > t.entry_price], curve
    assert find_lookahead(cheating, weekly_df), "Zukunftsblick wurde nicht erkannt"


@pytest.mark.parametrize("variant", WEEKLY_VARIANTS)
def test_weekly_execution_prices_are_observable(weekly_df, variant):
    trades, _ = weekly.run_backtest(weekly_df, **WEEKLY_VARIANTS[variant])
    by_time = weekly_df.set_index("time")
    for t in trades:
        entry_bar = by_time.loc[t.entry_time]
        exit_bar = by_time.loc[t.exit_time]
        assert t.entry_price == entry_bar.close, "Einstieg muss zum Close des Signalbalkens erfolgen"
        assert t.exit_time > t.entry_time, "Ausstieg erst nach dem Einstiegsbalken"
        if t.exit_signal in ("Stop-Loss", "Take-Profit"):
            assert exit_bar.low - 1e-9 <= t.exit_price <= exit_bar.high + 1e-9
        else:
            assert t.exit_price == exit_bar.close


def test_daily_execution_prices_are_observable(daily_df):
    trades, _ = daily.run_backtest(daily_df, daily.Settings())
    by_time = daily_df.set_index("time")
    for t in trades:
        assert t.entry_price == by_time.loc[t.entry_time].close
        assert t.exit_price == by_time.loc[t.exit_time].close
        assert t.exit_time > t.entry_time
        assert t.entry_day.dayofweek == 3            # Signal: Donnerstag


def test_sma_filter_uses_only_past_closes(weekly_df):
    """Ein Einstieg mit SMA-Filter darf nur stattfinden, wenn der Close über dem SMA
    der letzten n Closes (inkl. aktuellem Balken, ohne Zukunft) liegt."""
    n = 20
    trades, _ = weekly.run_backtest(weekly_df, sma_len=n)
    idx = {t: i for i, t in enumerate(weekly_df.time)}
    closes = weekly_df.close.values
    for t in trades:
        i = idx[t.entry_time]
        assert i >= n - 1
        assert closes[i] > closes[i - n + 1:i + 1].mean()
