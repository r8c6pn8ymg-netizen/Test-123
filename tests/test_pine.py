"""Statische Prüfung der TradingView-Indikatoren (Pine Script v6).

Pine lässt sich lokal nicht kompilieren. Diese Tests fangen deshalb die Fehler ab, die sich
ohne TradingView erkennen lassen: falsche Version, kaputte Klammern, und vor allem
versehentlich geänderte Standardwerte, die am Original kalibriert wurden.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.pine

PINE_DIR = Path(__file__).resolve().parent.parent / "pine"
PINE_FILES = sorted(PINE_DIR.glob("*.pine"))


def _src(name: str) -> str:
    return (PINE_DIR / name).read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    """Entfernt Kommentare und Zeichenketten, damit Klammern darin nicht mitgezählt werden."""
    out = []
    for line in src.splitlines():
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
        out.append(line.split("//", 1)[0])
    return "\n".join(out)


def _default(src: str, var: str) -> float:
    m = re.search(rf"\b{var}\s*=\s*input\.(?:float|int)\(\s*([-0-9.]+)", src)
    assert m, f"Eingabe {var} nicht gefunden"
    return float(m.group(1))


def test_all_indicators_present():
    names = {p.name for p in PINE_FILES}
    assert {"Regime_Kalman.pine", "COT_Commercials_Extreme.pine", "COT_Positionen_Index.pine",
            "Saison_Vorschau.pine", "Saison_Auswertung.pine"} <= names


@pytest.mark.parametrize("path", PINE_FILES, ids=lambda p: p.name)
def test_pine_structure(path):
    src = path.read_text(encoding="utf-8")
    assert src.splitlines()[0].strip() == "//@version=6"
    assert len(re.findall(r"^indicator\(", src, flags=re.M)) == 1, "genau eine indicator()-Deklaration"
    assert "\t" not in src, "Tabs statt Leerzeichen (Pine-Einrückung)"
    code = _code_only(src)
    for o, c in ["()", "[]", "{}"]:
        assert code.count(o) == code.count(c), f"Klammern {o}{c} unausgeglichen"


# ───────────────────────── Regime: am Original kalibrierte Schwellen ─────────────────────────
def _kalf(x: float, steep: float = 17.6211) -> float:
    return 100 / (1 + math.exp(-steep * x))


def test_regime_kalf_constants():
    src = _src("Regime_Kalman.pine")
    assert _default(src, "len") == 23
    assert _default(src, "stp") == pytest.approx(17.6211)


def test_regime_thresholds_match_original_calibration():
    src = _src("Regime_Kalman.pine")
    hy, lr = _default(src, "hy"), _default(src, "lr")
    hg, lg = _default(src, "hg"), _default(src, "lg")
    hc, lc = _default(src, "hc"), _default(src, "lc")
    # 93,36 / 6,64 = Schlusskurs 15 % über/unter der Mitte · 97,14 / 2,86 = 20 %
    assert hy == pytest.approx(_kalf(0.15), abs=0.005)
    assert hg == pytest.approx(_kalf(0.20), abs=0.005)
    assert (hy + lr, hg + lg) == (pytest.approx(100), pytest.approx(100)), "Schwellen müssen symmetrisch sein"
    assert (hc, lc) == (hg, lg)


def test_regime_yellow_trigger_is_level_cross():
    src = _code_only(_src("Regime_Kalman.pine"))
    assert "bool up = conf and k1 < lr and kalf >= lr" in src


# ───────────────────────── COT ─────────────────────────
@pytest.mark.parametrize("name", ["COT_Commercials_Extreme.pine", "COT_Positionen_Index.pine"])
def test_cot_uses_legacy_report_without_week_lag(name):
    src = _src(name)
    assert 'cot.COTTickerid("Legacy"' in src
    assert 'request.security(tL, "1D"' in src and 'request.security(tS, "1D"' in src, '"1D" vermeidet 1 Woche Verzug'
    assert '"098662"' in src, "USD-Code (Dollar-Index-Future)"


def test_cot_extreme_matches_opt_cot_rule():
    src = _src("COT_Commercials_Extreme.pine")
    assert _default(src, "lookback") == 26
    assert _default(src, "commThr") == 75
    assert _default(src, "smallThr") == 25
    assert _default(src, "rocLen") == 5
    assert _default(src, "rocMax") == 0, "Opt-COT: Netto-Position darf sich gar nicht gegen das Extrem bewegen"
    code = _code_only(src)
    assert "float old = nets.get(n - 1 - rocLen)" in code, "Veränderungsrate auf Netto-Position, nicht auf Index"


def test_cot_index_formula():
    for name in ["COT_Commercials_Extreme.pine", "COT_Positionen_Index.pine"]:
        assert "(net - lo) / (hi - lo) * 100" in _src(name)


# ───────────────────────── Saisonalität ─────────────────────────
def test_seasonal_evaluation_has_seasonax_options():
    src = _src("Saison_Auswertung.pine")
    assert 'options = ["Eröffnung", "Schlusskurs"]' in src
    assert "bool useThis = exitNext and tdPrev < exitTs" in src
    assert 'options = ["Letzte X Jahre", "Von–Bis", "Manuelle Jahre"]' in src


def test_seasonal_preview_defaults():
    src = _src("Saison_Vorschau.pine")
    assert _default(src, "fwdDays") == 14
    assert re.search(r'bool\s+detr\s*=\s*input\.bool\(true', src), "Trendbereinigung standardmäßig an"
