"""Gemeinsame Fixtures für alle Tests.

Hinweis zur Umgebung: pytest ist hier als eigenständiges Werkzeug (uv tool) installiert und
sieht die Projekt-Pakete (pandas, numpy, openpyxl) des System-Pythons nicht. Gleiche
Python-Version vorausgesetzt, wird deren site-packages-Ordner ergänzt – so muss nichts
doppelt installiert werden. In CI/venv (pandas direkt verfügbar) passiert nichts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    # Suchpfade des System-Pythons übernehmen (nur bei gleicher Version, sonst passen die Pakete nicht)
    exe = shutil.which(f"python{sys.version_info.major}.{sys.version_info.minor}") or shutil.which("python3")
    if exe:
        probe = "import json, sys; print(json.dumps([sys.version_info[:2], sys.path]))"
        res = subprocess.run([exe, "-c", probe], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            ver, paths = json.loads(res.stdout)
            if tuple(ver) == sys.version_info[:2]:
                sys.path.extend(p for p in paths if p and p not in sys.path)

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PINE = ROOT / "pine"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def pine_dir() -> Path:
    return PINE


@pytest.fixture(scope="session")
def daily_df() -> pd.DataFrame:
    import friday_rush_backtest as fr

    return fr.load_csv(str(DATA / "CAPITALCOM_SPX500_1D.csv"))


@pytest.fixture(scope="session")
def weekly_df() -> pd.DataFrame:
    import monthly_mid_weekly_backtest as w

    return w.load_weekly(str(DATA / "CAPITALCOM_SPX500_1W.csv"))


@pytest.fixture(scope="session")
def daily_close_df() -> pd.DataFrame:
    import daily_stop_backtest as ds

    return ds.load_daily_close(str(DATA / "CAPITALCOM_SPX500_1D_close.csv"))
