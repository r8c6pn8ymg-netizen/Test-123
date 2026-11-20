#!/usr/bin/env bash
# Führt alle Prüfungen aus: Ruff (Linting) und pytest (Unit-, Daten-, Regressions- und Pine-Tests).
# Aufruf aus dem Projektordner:  scripts/check.sh            (alles)
#                                scripts/check.sh -m pine    (nur eine Testgruppe, Marker siehe pyproject.toml)
# Optional als Git-Hook vor jedem Commit:  ln -s ../../scripts/check.sh .git/hooks/pre-commit
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── Ruff ──"
ruff check .

echo "── pytest ──"
pytest "$@"
