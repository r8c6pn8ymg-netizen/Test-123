"""Kleine, künstliche Kursreihen mit von Hand nachrechenbaren Ergebnissen.

Die Daten werden wie echte TradingView-/Capital.com-Exporte als CSV geschrieben
(Spalte time in Unix-Sekunden) und über die Ladefunktionen der Skripte eingelesen.
So wird die Datenaufbereitung (Zeitzonen, Handelstag, Wochen-Freitag) mitgetestet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _unix(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def write_daily_csv(path: Path, bars: list[tuple]) -> Path:
    """bars: (Handelstag 'YYYY-MM-DD', close[, 'flat'|(open, high, low)]).

    Die Tageskerze öffnet wie bei Capital.com am Vorabend um 22:00 UTC.
    Standard-Kerze: open = close, high = close + 1, low = close - 1.
    'flat' erzeugt eine Feiertagskerze mit High == Low == Close.
    """
    rows = []
    for day, close, *extra in bars:
        t = pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=2)        # Vortag 22:00 UTC
        if extra and extra[0] == "flat":
            o = h = l = close
        elif extra:
            o, h, l = extra[0]
        else:
            o, h, l = close, close + 1, close - 1
        rows.append({"time": _unix(t), "open": o, "high": h, "low": l, "close": close})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_weekly_csv(path: Path, bars: list[tuple]) -> Path:
    """bars: (Montag 'YYYY-MM-DD', close[, (open, high, low)]).

    Die Wochenkerze öffnet wie bei Capital.com am Sonntag um 23:00 UTC.
    """
    rows = []
    for monday, close, *extra in bars:
        t = pd.Timestamp(monday, tz="UTC") - pd.Timedelta(hours=1)     # Sonntag 23:00 UTC
        o, h, l = extra[0] if extra else (close, close + 1, close - 1)
        rows.append({"time": _unix(t), "open": o, "high": h, "low": l, "close": close})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
