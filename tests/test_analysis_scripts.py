"""Schnelltests für die Analyse-Skripte.

Jedes Skript läuft in einem leeren Testordner (Kopie der Eingabedaten), damit die
gespeicherten Ergebnisse in data/ und reports/ nicht überschrieben werden. Die Ergebnisse
werden mit den gespeicherten Dateien verglichen:
  * kleine/schnelle Skripte laufen komplett und müssen ihr Ergebnis exakt reproduzieren,
  * große Raster laufen mit wenigen SL/TP-Werten und weniger Monte-Carlo-Simulationen;
    diese Zeilen müssen mit dem gespeicherten Gesamtraster übereinstimmen.
"""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.analysis

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUTS = ["CAPITALCOM_SPX500_1D.csv", "CAPITALCOM_SPX500_1D_close.csv", "CAPITALCOM_SPX500_1W.csv"]


@pytest.fixture
def sandbox(tmp_path, monkeypatch) -> Path:
    """Leerer Arbeitsordner mit den Kursdaten; Skripte schreiben nur hierhin."""
    (tmp_path / "data").mkdir()
    for name in INPUTS:
        shutil.copy(DATA / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_script(name: str, *args: str) -> None:
    old = sys.argv
    sys.argv = [name, *args]
    try:
        runpy.run_path(str(ROOT / name), run_name="__main__")
    finally:
        sys.argv = old


def _rows(df: pd.DataFrame, sl_col: str, tp_col: str) -> pd.DataFrame:
    return df.set_index([df[sl_col].round(1), df[tp_col].round(1)]).sort_index()


def _assert_rows_match(new: pd.DataFrame, stored: pd.DataFrame, sl_col: str, tp_col: str, cols: list[str]) -> None:
    a, b = _rows(new, sl_col, tp_col), _rows(stored, sl_col, tp_col)
    b = b.loc[a.index]
    for c in cols:
        pd.testing.assert_series_equal(a[c].astype(float), b[c].astype(float), check_names=False, atol=0.011)


# ───────────────────────── komplette Läufe, exakter Vergleich ─────────────────────────
def test_daily_stop_backtest_reproduces_grid(sandbox):
    _run_script("daily_stop_backtest.py")
    new = pd.read_csv(sandbox / "data" / "daily_stop_grid_full.csv")
    pd.testing.assert_frame_equal(new, pd.read_csv(DATA / "daily_stop_grid_full.csv"))


def test_sma_filter_reproduces_stored_results(sandbox, capsys):
    _run_script("sma_filter_test.py")
    out = capsys.readouterr().out
    assert out == (DATA / "sma_filter_results.txt").read_text(encoding="utf-8")


@pytest.mark.slow
def test_quant_analysis_reproduces_summary_and_charts(sandbox):
    _run_script("quant_analysis.py")
    rep = sandbox / "reports"
    assert (rep / "quant_summary.txt").read_text(encoding="utf-8") == (ROOT / "reports" / "quant_summary.txt").read_text(
        encoding="utf-8"
    )
    for png in sorted((ROOT / "reports").glob("*.png")):
        made = rep / png.name
        assert made.is_file() and made.stat().st_size > 20_000, f"Grafik fehlt/leer: {png.name}"


# ───────────────────────── große Raster, verkleinert ─────────────────────────
def test_weekly_sltp_montecarlo_small_grid(sandbox, monkeypatch):
    import weekly_sltp_montecarlo as m

    monkeypatch.setattr(m, "SL_GRID", [0.0, 6.0])
    monkeypatch.setattr(m, "TP_GRID", [0.0, 5.0])
    monkeypatch.setattr(m, "N_SIM", 1_000)
    m.main()
    new = pd.read_csv(sandbox / "data" / "weekly_sltp_montecarlo.csv")
    assert len(new) == 4
    exact = ["Netto USD", "Max DD %", "SL-Treffer", "TP-Treffer", "2010-14 %", "2015-18 %", "2019-22 %", "2023-26 %"]
    _assert_rows_match(new, pd.read_csv(DATA / "weekly_sltp_montecarlo.csv"), "SL", "TP", exact)
    # Monte-Carlo-Spalten: weniger Simulationen -> nur Plausibilität
    base = new[(new.SL == 0) & (new.TP == 0)].iloc[0]
    assert base["MC P(> Basis) %"] == 0 and base["MC P(DD < Basis) %"] == 0
    for c in ["MC P(Verlust) %", "MC Median DD %", "MC 95%-DD %", "MC P(> Basis) %"]:
        assert new[c].between(0, 100).all(), c
    assert (new["MC 5%-Endkapital"] <= new["MC Median Endkapital"]).all()


def test_robustness_small_grid(sandbox, monkeypatch, capsys):
    import numpy as np

    import robustness_sl6_tp5 as r

    monkeypatch.setattr(r, "SL_GRID", np.array([5.5, 6.0, 6.5]))
    monkeypatch.setattr(r, "TP_GRID", np.array([4.5, 5.0, 5.5]))
    monkeypatch.setattr(r, "N_SIM", 1_000)
    r.main()
    new = pd.read_csv(sandbox / "data" / "robustness_sl6_tp5.csv")
    assert len(new) == 9
    _assert_rows_match(new, pd.read_csv(DATA / "robustness_sl6_tp5.csv"), "SL", "TP",
                       ["Netto", "DD %", "Blöcke>Basis", "SL-Treffer", "TP-Treffer"])
    assert "Monte-Carlo (1000 Simulationen)" in capsys.readouterr().out


def test_sltp_optimization_small_grid(sandbox, monkeypatch):
    import sltp_optimization as o

    monkeypatch.setattr(o, "SL_GRID", [0.0, 5.5, 6.0, 6.5])
    monkeypatch.setattr(o, "TP_GRID", [0.0, 4.5, 5.0, 5.5])
    o.main()
    new = pd.read_csv(sandbox / "data" / "sltp_grid_full.csv")
    assert len(new) == 16
    _assert_rows_match(new, pd.read_csv(DATA / "sltp_grid_full.csv"), "SL", "TP",
                       ["Trades", "Netto USD", "Max DD %", "Netto/MaxDD", "SL-Treffer", "TP-Treffer"])


def test_repository_outputs_untouched():
    """Die Tests dürfen die gespeicherten Ergebnisse nie verändern (läuft nach den anderen Tests dieser Datei)."""
    import subprocess

    changed = subprocess.run(["git", "status", "--porcelain", "data", "reports"], cwd=ROOT,
                             capture_output=True, text=True, check=False).stdout
    assert changed.strip() == "", f"Tests haben gespeicherte Dateien verändert:\n{changed}"
