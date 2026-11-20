"""Gemeinsame Fixtures für alle Tests.

Hinweis zur Umgebung: pytest ist hier als eigenständiges Werkzeug (uv tool) installiert und
sieht die Projekt-Pakete (pandas, numpy, openpyxl) des System-Pythons nicht. Gleiche
Python-Version vorausgesetzt, wird deren site-packages-Ordner ergänzt – so muss nichts
doppelt installiert werden. In CI/venv (pandas direkt verfügbar) passiert nichts.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for base in (sysconfig.get_config_var("prefix"), "/usr/local", "/usr"):
        cand = Path(base) / "lib" / ver / "site-packages"
        if (cand / "pandas").is_dir():
            sys.path.append(str(cand))
            break

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
