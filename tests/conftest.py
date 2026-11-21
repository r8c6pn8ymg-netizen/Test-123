"""Gemeinsame Einstellungen und Daten für die Backtest-Tests.

Sicherheitsregel: Die Tests lesen nur. Ein Wächter prüft am Ende der Testsitzung,
dass keine Datei in data/, reports/ oder kein Strategie-Skript verändert wurde.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# keine .pyc-Dateien schreiben (__pycache__/ ist zusätzlich in .gitignore ausgeschlossen)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                      # die Skripte arbeiten mit relativen Pfaden (data/...)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_checks as bc                  # noqa: E402
import friday_rush_backtest as daily          # noqa: E402
import monthly_mid_weekly_backtest as weekly  # noqa: E402

CAPITAL = 10_000.0
IS_START, IS_END = "2010-01-01", "2026-10-30"     # In-Sample (wie in den Skripten)
OOS_START, OOS_END = "2000-01-03", "2009-10-30"   # Out-of-Sample
N_TRIALS = 600 + 1681 + 5 + 99                    # = significance_tests.N_TESTED

# Kostenannahmen (SPX500-CFD). Hier anpassen, falls dein Broker andere Werte hat.
REALISTIC_COSTS = bc.CostModel(spread_points=0.5, slippage_ticks=1, tick=0.1, commission_pct=0.0)


# ───────────────────────── Schutz vor Überschreiben ─────────────────────────
def _snapshot() -> dict[str, str]:
    files = [p for d in ("data", "reports") for p in (ROOT / d).rglob("*") if p.is_file()]
    files += list(ROOT.glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.fixture(scope="session", autouse=True)
def project_files_unchanged():
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"Tests haben Projektdateien verändert: {changed}"


# ───────────────────────── Daten ─────────────────────────
@pytest.fixture(scope="session")
def weekly_df() -> pd.DataFrame:
    return weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")


@pytest.fixture(scope="session")
def weekly_long_df() -> pd.DataFrame:
    return weekly.load_weekly("data/CAPITALCOM_SPX500_1W_long.csv")


@pytest.fixture(scope="session")
def daily_df() -> pd.DataFrame:
    return daily.load_csv("data/CAPITALCOM_SPX500_1D.csv")


def weekly_returns(curve: pd.Series, start: str = IS_START) -> pd.Series:
    """Wöchentliche Renditen der Equity-Kurve ab start (inkl. Wochen ohne Position)."""
    c = curve[curve.index >= pd.Timestamp(start, tz="UTC")]
    return c.pct_change().dropna()
