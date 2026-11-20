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


def test_neue_basen_mit_gemessenen_fristen():
    """Kern-PCE, realer Einzelhandel, Erzeugerpreise: vorhanden, mit ALFRED-gemessenen Fristen und in der Echtzeit-Variante."""
    import pipeline as P
    b = {x.id: x for x in P.us_bases(False)}
    assert b["PCEPILFE"].lag >= 62 and b["RRSFS"].lag >= 40 and b["PPIACO"].lag >= 40
    for k in ("PCEPILFE", "RRSFS", "PPIACO"):
        assert k in P.ALFRED_SERIES


# ---------- Termine der Woche ----------
def _termine():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
    import termine
    return termine


def _fake_get(sid):
    idx = pd.date_range("2024-01-01", periods=30, freq="MS")
    return pd.Series(np.arange(100.0, 130.0), index=idx)


def test_termine_offizielle_uhrzeit_statt_fred_zeit():
    """FRED-Kalenderzeiten sind teils Einstellzeiten; verwendet werden die offiziellen New Yorker Zeiten."""
    from datetime import date
    t = _termine()
    rows = [dict(day=date(2026, 11, 20), rid=91, name="Surveys of Consumers"),          # FRED: 10:00 Central
            dict(day=date(2026, 11, 19), rid=180, name="Unemployment Insurance Weekly Claims Report")]
    out = t.build(today=date(2026, 11, 20), fred_rows=rows, pages={"Census": [], "BEA": [], "Fed-G.17": []}, get=_fake_get)
    um = next(e for e in out["events"] if "Michigan" in e["title"])
    cl = next(e for e in out["events"] if "Erstanträge" in e["title"])
    assert um["time"] == "16:00" and cl["time"] == "14:30"   # 10:00 bzw. 8:30 New York


def test_termine_offizielle_quelle_ersetzt_fred_eintraege():
    """Census/BEA/G.17 bestimmen die Termine; doppelte FRED-Eintraege (z. B. zweite Industrieproduktion) fallen weg."""
    from datetime import date
    t = _termine()
    rows = [dict(day=date(2026, 11, 24), rid=13, name="G.17 Industrial Production and Capacity Utilization"),
            dict(day=date(2026, 11, 25), rid=27, name="New Residential Construction")]
    pages = {"Census": [(date(2026, 11, 18), "x New Residential Construction (Building Permits) November 18")],
             "BEA": [(date(2026, 11, 25), "GDP (Second Estimate) and Corporate Profits, 3rd Quarter 2026")],
             "Fed-G.17": [date(2026, 11, 17), date(2026, 12, 16)]}
    out = t.build(today=date(2026, 11, 20), fred_rows=rows, pages=pages, get=_fake_get)
    ip = [e["date"] for e in out["events"] if e["title"] == "Industrieproduktion"]
    st = [e["date"] for e in out["events"] if e["title"].startswith("Baubeginne")]
    assert ip == ["2026-11-17"] and st == ["2026-11-18"]
    gdp = next(e for e in out["events"] if "BIP" in e["title"])
    assert gdp["time"] == "14:30" and gdp["src"] == "bea.gov"


def test_termine_parser_census_bea_g17():
    from datetime import date
    t = _termine()
    cen = t.parse_census("<td>Advance Monthly Sales for Retail and Food Services</td><td>November 17, 2026</td>"
                         "<td>8:30 AM</td><td>October 2026</td><td>A202611170830 A202610</td>")
    assert cen[0][0] == date(2026, 11, 17) and "Advance Monthly Sales" in cen[0][1]
    bea = t.parse_bea("November 25 8:30 AM N ews Personal Income and Outlays, October 2026 "
                      "December 8 8:30 AM N ews U.S. International Trade in Goods and Services, October 2026 View",
                      date(2026, 11, 20))
    assert bea[0] == (date(2026, 11, 25), "Personal Income and Outlays, October 2026")
    assert bea[1][0] == date(2026, 12, 8)
    assert t.parse_g17("17-November-2026 December 2026 16-December-2026") == [date(2026, 11, 17), date(2026, 12, 16)]


def test_termine_ism_regel():
    from datetime import date
    t = _termine()
    assert t.ism_dates(date(2026, 12, 1), date(2026, 12, 31), 1) == [date(2026, 12, 1)]   # Di. 1.12.
    assert t.ism_dates(date(2026, 12, 1), date(2026, 12, 31), 3) == [date(2026, 12, 3)]
    assert t.ism_dates(date(2027, 1, 1), date(2027, 1, 31), 1) == [date(2027, 1, 4)]     # 1.1. Feiertag, dann Wochenende


def test_termine_werte_umrechnung():
    t = _termine()
    v = t.latest_values([("M/M", "X", "mom"), ("J/J", "X", "yoy"), ("Diff", "X", "diff_k")], get=_fake_get)
    assert v[0]["value"] == "+0,8 %" and v[0]["period"] == "Jun 2026"     # 129/128-1
    assert v[1]["value"] == "10,3 %"                                        # 129/117-1
    assert v[2]["value"] == "+1 Tsd."


def test_termine_notenbanken_vollstaendig_und_sortiert():
    from datetime import date
    t = _termine()
    assert {c[0] for c in t.CENTRAL_BANKS} == {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
    for ccy, bank, *_, dates in t.CENTRAL_BANKS:
        assert dates == sorted(dates), bank
    out = t.build(today=date(2026, 12, 7), fred_rows=[], pages={"Census": [], "BEA": [], "Fed-G.17": []}, get=_fake_get)
    cb = [e for e in out["events"] if e["kind"] == "cb"]
    assert any(e["ccy"] == "AUD" and e["date"] == "2026-12-08" and e["time"] == "04:30" for e in cb)  # 14:30 Sydney
    assert any(e["ccy"] == "USD" and e["time"] == "20:00" for e in cb)                                 # 14:00 New York
    assert len(out["next_cb"]) == 8


def test_termine_geschuetzte_leerzeichen():
    """&nbsp; zwischen Monat und Tag (kommt bei BoE/BoC vor) darf Termine nicht verschlucken."""
    from datetime import date
    t = _termine()
    bea = t.parse_bea("November&nbsp;25 8:30&nbsp;AM N ews Personal Income and Outlays, October 2026 View", date(2026, 11, 20))
    assert bea == [(date(2026, 11, 25), "Personal Income and Outlays, October 2026")]


def test_termine_ergebnis_oder_letzter_stand():
    """Je Termin eindeutig: Ergebnis (Termin vorbei, neuer Zeitraum da) oder letzter Stand (Termin kommt noch)."""
    from datetime import date
    t = _termine()
    # Welcher Zeitraum erscheint mit dem Termin?
    assert str(t.expected_period("retail", date(2026, 11, 17), "M").date()) == "2026-10-01"
    assert str(t.expected_period("jolts", date(2026, 12, 8), "M").date()) == "2026-10-01"
    assert str(t.expected_period("claims", date(2026, 11, 19), "W").date()) == "2026-11-14"   # Do. -> Sa. davor
    assert str(t.expected_period("claims", date(2026, 11, 25), "W").date()) == "2026-11-21"   # Mi. (Thanksgiving)
    assert str(t.expected_period("gdp", date(2026, 11, 25), "Q").date()) == "2026-07-01"      # Q3
    idx = pd.date_range("2025-01-01", "2026-09-01", freq="MS")                                  # Daten bis Sep
    get = lambda sid: pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
    rows = [dict(day=date(2026, 11, 20), rid=91, name="Surveys of Consumers")]
    pages = {"Census": [(date(2026, 11, 17), "Advance Monthly Sales for Retail and Food Services"),
                        (date(2026, 11, 25), "Advance Report on Durable Goods")], "BEA": [], "Fed-G.17": []}
    out = t.build(today=date(2026, 11, 20), fred_rows=rows, pages=pages, get=get)
    ev = {e["title"]: e for e in out["events"]}
    assert ev["Einzelhandelsumsätze"]["vstate"] == "missing"            # vorbei, aber Okt fehlt noch
    assert ev["Auftragseingänge langlebige Güter"]["vstate"] == "before"
    assert ev["Auftragseingänge langlebige Güter"]["vnext"] == "Okt 2026"
    assert "Uni Michigan" in ev["Verbraucherstimmung Uni Michigan"]["vnote"]  # Grund statt leerer Zeile


# ---------- Pruefung 20.11. (Gesamt-Review) ----------
def test_json_ohne_nan():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
    import json
    import data_sources as ds
    s = ds.to_json({"a": float("nan"), "b": [np.float64(1.5), np.inf], "c": np.int64(3)})
    assert json.loads(s) == {"a": None, "b": [1.5, None], "c": 3}


def test_faktor_laufender_monat_zaehlt_nicht():
    """Kurse nur bis Monatsmitte: der laufende Monat darf keine Monatsrendite liefern."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
    import factors
    idx = pd.date_range("2026-01-01", "2026-11-13", freq="B")
    px = {c: pd.Series(np.linspace(1, 2, len(idx)), index=idx) for c in ("EUR", "GBP")}
    spot = pd.DataFrame({c: px[c].resample("ME").last() for c in px})
    spot["USD"] = 1.0
    sp = factors.complete_months(spot, px)
    assert sp.loc["2026-11-30", ["EUR", "GBP"]].isna().all()       # November unvollstaendig
    assert sp.loc["2026-10-31", ["EUR", "GBP"]].notna().all()
    full = {c: pd.Series(1.0, index=pd.date_range("2026-01-01", "2026-11-30", freq="D")) for c in px}
    assert factors.complete_months(spot, full).loc["2026-11-30", ["EUR", "GBP"]].notna().all()


def test_alfred_datenstand_am_monatsende_nicht_doppelt():
    import inspect
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
    import pipeline
    src = inspect.getsource(pipeline.realtime_features)
    assert "vint[-1] != today" in src


def test_termin_heute_morgens_noch_nicht_vorbei():
    """Lauf um 07:47: ein US-Termin um 14:30 desselben Tages ist noch nicht vorbei (nicht "fehlt bei FRED")."""
    from datetime import date
    t = _termine()
    widx = pd.date_range("2026-10-03", "2026-11-14", freq="W-SAT")              # Daten bis Woche 14.11.
    get = lambda sid: pd.Series(np.arange(len(widx), dtype=float) * 1000 + 200000, index=widx)
    rows = [dict(day=date(2026, 11, 25), rid=180, name="Unemployment Insurance Weekly Claims Report")]
    out = t.build(today=date(2026, 11, 25), fred_rows=rows, pages={"Census": [], "BEA": [], "Fed-G.17": []}, get=get)
    cl = next(e for e in out["events"] if e["title"].startswith("Erstanträge"))
    assert cl["vstate"] == "before" and cl["vnext"] == "Woche bis 21.11."


def test_werte_fuer_bestimmten_zeitraum():
    """upto: ein Termin zeigt die Zahl seines eigenen Zeitraums, nicht eine spaeter erschienene."""
    t = _termine()
    widx = pd.date_range("2026-10-03", "2026-11-28", freq="W-SAT")
    get = lambda sid: pd.Series(np.arange(len(widx), dtype=float) * 1000 + 200000, index=widx)
    v = t.latest_values([("Anträge", "ICSA", "k_persons")], get, upto=pd.Timestamp("2026-11-14"))
    assert v[0]["period"] == "Woche bis 14.11." and v[0]["prev_period"] == "Woche bis 07.11."
