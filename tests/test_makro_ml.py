"""Selbsttests fuer das Makro-ML-Cockpit (makro_ml/).

Nutzen nur den lokalen Cache (kein Netz noetig, sobald die Pipeline einmal gelaufen ist).
Ausfuehren: python3 -m pytest tests/test_makro_ml.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
CACHE = Path(__file__).resolve().parents[1] / "makro_ml" / "cache"
pytestmark = pytest.mark.skipif(not any(CACHE.glob("fred_DEXUSEU.csv")), reason="Cache fehlt: erst pipeline.py ausfuehren")


@pytest.fixture(scope="module")
def px():
    import pipeline as P
    return P.load_fx(False)


@pytest.fixture(scope="module")
def core(px):
    import factors as F
    return F.core(px)


def test_kein_phantom_monat_am_ende(core):
    ex = core["excess"]
    last = ex.iloc[-1]
    assert last.isna().all(), "letzte Zeile darf keine Rendite haben (auch USD nicht)"


def test_euro_erst_ab_1999_handelbar(core):
    import factors as F
    for k in ("carry", "value", "momentum", "kombi"):
        z = core["z"][k]
        pre = z.loc[z.index < "1999-01-01", "EUR"]
        assert pre.isna().all(), f"EUR darf vor 1999 im Faktor {k} keinen Score haben"
        w = F.ls_weights(z[z.index >= "1976-12-31"])
        assert (w.loc[w.index < "1999-01-01", "EUR"].fillna(0) == 0).all()


def test_einstiegskosten_werden_berechnet():
    import factors as F
    w = pd.DataFrame({"A": [1 / 3, 1 / 3, np.nan, 1 / 3], "B": [-1 / 3, -1 / 3, np.nan, -1 / 3]})
    t = F.turnover(w)
    assert t.iloc[0] == pytest.approx(2 / 3)          # Einstieg
    assert t.iloc[1] == pytest.approx(0)               # unveraendert
    assert t.iloc[2] == pytest.approx(2 / 3)           # Ausstieg (Pause)
    assert t.iloc[3] == pytest.approx(2 / 3)           # Wiedereinstieg


def test_faktor_placebo_erkennt_leck(core):
    """Score = Rendite des FOLGEmonats (Leck) muss riesig sein, Zufall muss nahe 0 sein."""
    import factors as F
    ex = core["excess"]
    def sharpe(score):
        w = F.ls_weights(score[score.index >= "1990-01-31"])
        g = (w * ex.reindex(w.index)).sum(axis=1, min_count=1).dropna()
        return g.mean() / g.std() * np.sqrt(12)
    leak = ex.copy()
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.standard_normal(ex.shape), index=ex.index, columns=ex.columns).where(ex.notna())
    assert sharpe(leak) > 3
    assert abs(sharpe(noise)) < 0.5


def test_makro_walk_forward_ohne_leck(px):
    """Gleiche Reihe einmal mit Leck (sofort verfuegbar) und einmal korrekt verzoegert."""
    import pipeline as P
    p = px["EUR"].dropna()
    dates = pd.DatetimeIndex(p.resample("W-FRI").last().dropna().index)
    dates = dates[dates >= "1999-01-08"]
    pm = p.resample("MS").last()
    fut = (pm.shift(-1) / pm - 1) * 100
    today = pd.Timestamp.today().normalize()
    leak = P.run_model(P.features(P.Basis("X", "x", fut.dropna(), -31, "T"), dates), p, 30, today)
    ok = P.run_model(P.features(P.Basis("X", "x", fut.dropna(), 32, "T"), dates), p, 30, today)
    assert leak["da"] > 0.6, "Placebo-Leck muss erkannt werden"
    assert ok["da"] < 0.56, "korrekt verzoegerte Zukunftsreihe darf keinen Vorteil haben"


def test_features_erst_nach_veroeffentlichung():
    import pipeline as P
    s = pd.Series([1.0, 2.0, 3.0] * 10, index=pd.date_range("2020-01-01", periods=30, freq="MS"))
    b = P.Basis("X", "x", s, 45, "T")
    dates = pd.date_range("2021-01-01", "2022-06-30", freq="W-FRI")
    f = P.features(b, dates)
    # Wert des Monats M darf erst ab Monatsende + 45 Tage auftauchen
    s_ok = s[s.index >= "2021-01-01"]   # ab hier gibt es 12 Monate Vorlauf (fuer d12)
    checked = 0
    for d in dates:
        known = s_ok[(s_ok.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)) <= d]
        if len(known):
            assert f.loc[d, "lvl"] == known.iloc[-1], f"{d.date()}: falscher/zu frueher Wert"
            checked += 1
        else:
            assert pd.isna(f.loc[d, "lvl"]), f"{d.date()}: Wert vor Veroeffentlichung genutzt"
    assert checked > 20


def test_quartals_cpi_erst_ab_quartalsende():
    import pipeline as P
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    q = pd.Series(np.repeat(np.arange(8.0), 3), index=idx)   # Quartalswert x3
    out = P.quarter_end_if_quarterly(q)
    assert out.loc["2020-01-01":"2020-02-01"].isna().all() or out.index[0] >= pd.Timestamp("2020-03-01")
    assert out.loc["2020-03-01"] == 0 and out.loc["2020-04-01"] == 0 and out.loc["2020-06-01"] == 1


def test_curve_lag_ist_maximum_der_eingaenge():
    import pipeline as P
    s = pd.Series(1.0, index=pd.date_range("2000-01-01", periods=60, freq="MS"))
    cc = {"Y10": (s, 62), "POL": (s, 32)}
    us = {"Y10": (s, 62), "POL": (s, 32)}
    curve = [b for b in P.diff_bases(cc, us) if b.id == "CURVE-Diff"][0]
    assert curve.lag == 62
