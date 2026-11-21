"""Testumgebung: pytest muss im selben Python laufen wie die Backtests.

Aufruf immer mit  python -m pytest  – dann ist garantiert, dass das Python verwendet
wird, in dem pandas & Co. installiert sind (ein separates 'pytest'-Programm kann zu
einer anderen Umgebung gehören).
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import re
import sys
from pathlib import Path

import pytest

# Für Backtests und Tests zwingend nötig
CORE = {"pandas", "numpy", "scipy", "matplotlib", "openpyxl", "pytest"}
# Nur für einzelne Analyse-Skripte (significance_tests.py: statsmodels)
IMPORT_NAME = {"scikit-learn": "sklearn"}


def _requirements() -> dict[str, tuple[str, str]]:
    out = {}
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([^\s;]+)", line)
        if m:
            out[m.group(1).lower()] = (m.group(2), m.group(3))
    return out


REQ = _requirements()


def test_python_version():
    assert sys.version_info[:2] == (3, 13), sys.version


def test_pytest_runs_in_project_interpreter():
    """pytest ist als Modul dieses Interpreters installiert und pandas ist importierbar."""
    import pandas  # noqa: F401
    assert Path(pytest.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) or \
        Path(pytest.__file__).resolve().is_relative_to(Path(sys.base_prefix).resolve()), \
        f"pytest stammt aus {pytest.__file__}, Python ist {sys.executable}"


@pytest.mark.parametrize("pkg", sorted(CORE))
def test_core_packages_installed_with_pinned_version(pkg):
    assert pkg in REQ, f"{pkg} fehlt in requirements.txt"
    importlib.import_module(IMPORT_NAME.get(pkg, pkg))
    op, want = REQ[pkg]
    have = md.version(pkg)
    if op == "==":
        assert have == want, f"{pkg}: installiert {have}, requirements.txt verlangt {want}"


@pytest.mark.parametrize("pkg", sorted(set(REQ) - CORE))
def test_optional_packages(pkg):
    try:
        importlib.import_module(IMPORT_NAME.get(pkg, pkg))
    except ImportError:
        pytest.skip(f"{pkg} ist nicht installiert (nur für einzelne Analyse-Skripte nötig)")
