"""Tests fuer makro_ml/returns_stats.py (Reiter "Statistik"), Definitionen: makro_ml/SPEC_STATISTIK.md.

Offline: nur Fixtures, kein Netz.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makro_ml"))
import returns_stats as R  # noqa: E402

A = np.array([0.012, -0.034, 0.008, 0.021, -0.015, 0.004, -0.052, 0.017, 0.009, -0.006, 0.026, -0.011])


@pytest.fixture
def ra():
    return pd.Series(A, index=pd.date_range("2020-01-03", periods=12, freq="W-FRI"))


def test_fixture_a_referenzwerte(ra):
    d = R.describe(ra, 52)
    D = d["distribution"]
    assert D["n"] == 12
    assert D["mean"] == pytest.approx(-0.00175, abs=1e-9)
    assert D["median"] == pytest.approx(0.006, abs=1e-9)
    assert D["sigma"] == pytest.approx(0.023191397307857693, abs=1e-9)
    assert D["skew"] == pytest.approx(-1.040126098649676, abs=1e-9)
    assert D["kurt"] == pytest.approx(0.6185745210167743, abs=1e-9)
    assert d["quantiles"]["0.05"]["obs"] == pytest.approx(-0.0421, abs=1e-9)
    assert d["quantiles"]["0.01"]["obs"] == pytest.approx(-0.05002, abs=1e-9)
    assert d["var_es"]["0.95"]["var_hist"] == pytest.approx(0.0421, abs=1e-9)
    assert d["var_es"]["0.95"]["es_hist"] == pytest.approx(0.052, abs=1e-9)
    t = d["tests"]
    assert t["ttest"]["t"] == pytest.approx(-0.26139769613782987, abs=1e-6)
    assert t["ttest"]["p"] == pytest.approx(0.7986154425805115, abs=1e-6)
    assert t["ttest"]["df"] == 11
    assert t["jarque_bera"]["stat"] == pytest.approx(1.6417935896510336, abs=1e-6)
    assert t["jarque_bera"]["p"] == pytest.approx(0.4400368547309016, abs=1e-6)
    assert t["shapiro"]["w"] == pytest.approx(0.9170597587384509, abs=1e-4)
    assert t["shapiro"]["p"] == pytest.approx(0.26248858790321716, abs=1e-4)
    assert D["best"] == pytest.approx(0.026) and D["best_date"] == str(ra.index[10].date())
    assert D["worst"] == pytest.approx(-0.052) and D["worst_date"] == str(ra.index[6].date())
    assert (D["up"], D["down"], D["zero"]) == (7, 5, 0)


def test_annualisierung_und_normal_quantile(ra):
    d = R.describe(ra, 52)
    s = d["distribution"]["sigma"]
    assert d["distribution"]["ann_vol"] == pytest.approx(s * math.sqrt(52))
    mu = d["distribution"]["mean"]
    assert d["quantiles"]["0.99"]["normal"] == pytest.approx(mu + s * st.norm.ppf(0.99))
    assert R.ann_factor("1d", True) == 365 and R.ann_factor("1d", False) == 252
    assert R.ann_factor("1wk", True) == 52 and R.ann_factor("1mo", False) == 12


def test_live_rang_screenshot_plausibel():
    mu, sd, n, rs = 0.002236, 0.023542, 1763, -0.0699
    z = (rs - mu) / sd
    assert n * st.norm.cdf(z) == pytest.approx(1.924, abs=1e-3)
    assert 100 * sd * math.sqrt(52) == pytest.approx(16.98, abs=1e-2)
    # dieselbe Rechnung ueber live_rank
    hist = np.random.default_rng(0).normal(mu, sd, n)
    lr = R.live_rank(hist, rs, mu, sd, pd.Timestamp("1993-02-05"), pd.Timestamp("2026-11-13"))
    assert lr["normal_expect"] == pytest.approx(1.924, abs=1e-3)
    assert lr["years"] == 34


def test_rang_logik_mit_gleichstand():
    h = np.array([-0.05, -0.03, -0.03, -0.01, 0.0, 0.02, 0.04])
    lr = R.live_rank(h, -0.03, h.mean(), h.std(ddof=1), pd.Timestamp("2000-01-01"), pd.Timestamp("2010-01-01"))
    assert lr["worse_count"] == 1          # nur -0.05 streng schlechter
    assert lr["at_least_as_bad"] == 4      # -0.05, -0.03, -0.03 und die aktuelle selbst
    assert lr["better_count"] == 4
    assert lr["at_least_as_good"] == 7
    assert lr["percentile"] == pytest.approx(100 / 7)
    assert lr["years"] == 10


def test_parametrischer_var_es_geschlossene_formel():
    mu, sd = 0.001, 0.02
    for a in (0.05, 0.01):
        v, e = R.var_es_param(mu, sd, a)
        z = st.norm.ppf(a)
        assert v == pytest.approx(-(mu + sd * z), abs=1e-12)
        assert e == pytest.approx(-(mu - sd * st.norm.pdf(z) / a), abs=1e-12)
        # ES aus Integration der Normalverteilung
        xs = np.linspace(mu - 12 * sd, mu + sd * z, 200001)
        es_num = -np.trapezoid(xs * st.norm.pdf(xs, mu, sd), xs) / a
        assert e == pytest.approx(es_num, rel=1e-5)


def test_hist_es_inklusive_gleichheit():
    r = np.array([-0.1, -0.1, 0.0, 0.1] * 25)
    v, e = R.var_es_hist(r, 0.05)
    assert v == pytest.approx(0.1) and e == pytest.approx(0.1)


def test_resampling_mit_feiertag_und_live():
    # 2 Wochen: Mo 6.4.2026 - Fr 17.4.2026, Karfreitag 3.4. liegt davor; Feiertag Fr 10.4. (fiktiv) fehlt
    days = pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
                           "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17", "2026-04-20"])
    px = pd.Series(np.arange(10, 20, dtype=float), index=days)
    w = R.resample_last(px.iloc[:9], "1wk")
    assert list(w.index) == [pd.Timestamp("2026-04-09"), pd.Timestamp("2026-04-17")]
    assert list(w.values) == [13.0, 18.0]
    # Montag 20.4. bildet eine angebrochene Woche: am 21.4. ist sie live, am 25.4. abgeschlossen
    w2 = R.resample_last(px, "1wk")
    assert R.split_live(w2, "1wk", pd.Timestamp("2026-04-21"))[1] is True
    assert R.split_live(w2, "1wk", pd.Timestamp("2026-04-25"))[1] is False
    assert R.split_live(w2, "1wk", pd.Timestamp("2026-04-24"))[1] is True   # Freitag selbst: Woche laeuft noch
    m = R.resample_last(px, "1mo")
    assert len(m) == 1 and R.split_live(m, "1mo", pd.Timestamp("2026-04-21"))[1] is True
    assert R.split_live(px, "1d", pd.Timestamp("2026-04-20"))[1] is True
    assert R.split_live(px, "1d", pd.Timestamp("2026-04-21"))[1] is False


def test_clean_prices():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-06", "2020-01-07"]).tz_localize("UTC")
    s = pd.Series([3.0, 1.0, 2.0, -1.0, np.nan], index=idx)
    c = R.clean_prices(s)
    assert c.index.tz is None
    assert list(c.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert list(c.values) == [2.0, 3.0]   # Duplikat: letzter Wert, Kurs <= 0 und NaN weg


def test_log_gleich_ln_simple():
    p = pd.Series([100, 103, 99, 101.5, 120], index=pd.date_range("2020-01-01", periods=5))
    s, lg = R.to_returns(p, "simple"), R.to_returns(p, "log")
    assert len(s) == 4   # erste Periode entfaellt
    assert np.allclose(np.log1p(s.values), lg.values, atol=1e-15)


def test_histogramm_null_ist_kante_und_summe():
    r = np.random.default_rng(3).standard_t(4, 1500) * 0.02
    for b in R.BINS:
        h = R.histogram(r, b, r.mean(), r.std(ddof=1))
        e = np.array(h["edges"])
        assert np.min(np.abs(e)) < 1e-12, "0 muss eine Kante sein"
        assert sum(h["counts"]) == r.size
        assert e[0] <= r.min() and e[-1] >= r.max()
        assert all(c in ("neg", "pos") for c in h["colors"])
        assert np.allclose(np.diff(e), h["width"])
    fd = np.histogram_bin_edges(r, bins="fd")
    assert R.histogram(r, "auto", 0, 1)["width"] == pytest.approx(fd[1] - fd[0])


def test_chi2_zusammenlegen_und_ndf():
    r = np.random.default_rng(5).normal(0, 1, 2000)
    mu, sd = r.mean(), r.std(ddof=1)
    e = R.hist_edges(r, "50")
    c = R.chi2_gof(r, e, mu, sd)
    assert c["ndf"] == c["bins_used"] - 3
    assert c["p"] > 0.001  # echte Normalverteilung soll nicht klar verworfen werden
    # Nachrechnung der Zusammenlegung: alle erwarteten Haeufigkeiten >= 5
    obs, _ = np.histogram(r, bins=e)
    assert c["bins_used"] < len(obs)
    # zu wenige Bins -> n/a
    c2 = R.chi2_gof(r[:12], R.hist_edges(r[:12], "25"), mu, sd)
    assert c2["p"] is None or c2["ndf"] >= 1


def test_anderson_darling_entscheidung_und_p():
    rng = np.random.default_rng(7)
    norm, fat = rng.normal(size=1000), rng.standard_t(2, size=1000)
    a, b = R.anderson(norm), R.anderson(fat)
    assert a["normal"]["0.95"] and a["p"] > 0.05
    assert not b["normal"]["0.95"] and not b["normal"]["0.99"] and b["p"] < 0.001
    assert a["a2_star"] == pytest.approx(a["a2"] * (1 + 0.75 / 1000 + 2.25 / 1000 ** 2))
    # p-Wert an den kritischen Werten ~ Niveau
    assert R.ad_pvalue(0.752) == pytest.approx(0.05, abs=0.002)
    assert R.ad_pvalue(1.035) == pytest.approx(0.01, abs=0.001)
    # sehr grosse A*^2 (z. B. S&P 500 taeglich seit 1927): kein Ueberlauf, p faellt monoton auf 0
    ps = [R.ad_pvalue(v) for v in (0.1, 0.3, 0.5, 1, 5, 50, 152.9, 153, 400, 5000)]
    assert all(a >= b for a, b in zip(ps, ps[1:])) and ps[-1] == 0.0


def test_shapiro_nur_bis_5000():
    r = pd.Series(np.random.default_rng(1).normal(size=5001) * 0.01, index=pd.date_range("2000-01-01", periods=5001))
    t = R.describe(r, 252)["tests"]["shapiro"]
    assert t["w"] is None and t["na"] == "n/a (N > 5000)"


def test_walk_forward_fenster_rolling_expanding():
    assert R.wf_windows(10, 4, 2, "rolling") == [(0, 4, 4, 6), (2, 6, 6, 8), (4, 8, 8, 10)]
    assert R.wf_windows(10, 4, 2, "expanding") == [(0, 4, 4, 6), (0, 6, 6, 8), (0, 8, 8, 10)]
    assert R.wf_windows(11, 4, 3, "rolling") == [(0, 4, 4, 7), (3, 7, 7, 10), (6, 10, 10, 11)]  # letzter partial
    assert R.wf_windows(4, 4, 2, "rolling") == []


def test_walk_forward_kein_look_ahead():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2000-01-07", periods=300, freq="W-FRI")
    base = pd.Series(rng.normal(0.001, 0.02, 300), index=idx)
    a = R.walk_forward(base, 156, 52, "rolling", "simple", 52)
    for k, step in enumerate(a["steps"]):
        s, e = step["is_idx"]
        bad = base.copy()
        bad.iloc[e:] = -0.5           # alles ab OOS-Start extrem
        b = R.walk_forward(bad, 156, 52, "rolling", "simple", 52)
        sb = b["steps"][k]
        assert sb["is_mu"] == step["is_mu"] and sb["is_sigma"] == step["is_sigma"]
        for c in ("0.95", "0.99"):
            for m in ("hist", "param"):
                assert sb["by"][c][m]["var"] == step["by"][c][m]["var"]
                assert sb["by"][c][m]["es"] == step["by"][c][m]["es"]
        assert s == (k * 52) and e == 156 + k * 52
    assert a["steps"][-1]["partial"] is True and a["steps"][-1]["oos_n"] == 300 - 156 - 2 * 52


def test_walk_forward_verletzungen_von_hand():
    r = pd.Series([0.0] * 20 + [-0.1, 0.0, 0.0, -0.2], index=pd.date_range("2000-01-01", periods=24))
    w = R.walk_forward(r, 20, 4, "rolling", "simple", 252)
    st0 = w["steps"][0]
    assert st0["by"]["0.95"]["hist"]["var"] == 0.0      # IS nur Nullen
    assert st0["by"]["0.95"]["hist"]["viol"] == 2       # -0.1 und -0.2 < -0
    assert st0["by"]["0.95"]["hist"]["oos_es"] == pytest.approx(0.15)
    assert w["total"]["hist"]["0.95"]["expected"] == pytest.approx(0.05 * 4)


def test_kupiec():
    k = R.kupiec(5, 250, 0.01)
    x, t, a = 5, 250, 0.01
    lr = -2 * ((t - x) * math.log(1 - a) + x * math.log(a)) + 2 * ((t - x) * math.log(1 - x / t) + x * math.log(x / t))
    assert k["lr"] == pytest.approx(lr) and k["lr"] == pytest.approx(1.9568, abs=1e-3)
    assert k["p"] == pytest.approx(st.chi2.sf(lr, 1))
    z = R.kupiec(0, 250, 0.01)
    assert z["lr"] == pytest.approx(-2 * 250 * math.log(0.99)) and 0 < z["p"] < 1
    assert R.kupiec(250, 250, 0.01)["p"] < 1e-10


def _fake_px(n=1500, seed=2, start="2015-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def test_payload_schema_ohne_nan():
    px = _fake_px()
    today = px.index[-1]  # letzter Tag = heute -> Tag laeuft noch
    pay = R.instrument_payload("TST", "Test", "Aktien", "yf:TST", "", px, today)
    txt = R.ds.to_json(pay)
    assert "NaN" not in txt and "Infinity" not in txt
    j = json.loads(txt)
    assert j["meta"]["daily_live"] is True
    assert len(j["series"]["d"]) == len(px) - 1   # laufender Tag nicht im Strategie-Tester
    assert j["series"]["d"][0] == 0 and j["series"]["d0"] == str(px.index[0].date())
    assert set(j["period_ends"]) == {"1wk", "1mo"}
    c = j["combos"]["1wk|simple|max"]
    for key in ("distribution", "quantiles", "var_es", "tails", "tests", "hist", "qq", "live_rank", "wf"):
        assert key in c
    for b in R.BINS:
        assert {"edges", "counts", "colors", "normal_curve_x", "normal_curve_y", "chi2"} <= set(c["hist"][b])
    assert c["wf"]["rolling"]["total"]["hist"]["0.99"]["t"] > 0
    assert "1mo|simple|1y" not in j["combos"]   # N < 30
    assert "wf" not in j["combos"]["1wk|simple|5y"]  # Walk-Forward nur ueber die volle Historie
    r = R.round_sig({"a": [0.123456789, 1e-40 / 3, float("nan")], "b": 3})
    assert r["a"][0] == 0.123457 and r["a"][1] == pytest.approx(3.33333e-41) and math.isnan(r["a"][2]) and r["b"] == 3
    # Wochen-Enden: nur abgeschlossene, Index zeigt auf den letzten Handelstag der Woche
    d = pd.Timestamp(j["series"]["d0"]) + pd.to_timedelta(j["series"]["d"], unit="D")
    ends = d[j["period_ends"]["1wk"]]
    assert all(e.weekday() == 4 or (e + pd.Timedelta(days=1)).weekday() != 0 for e in ends)
    assert ends[-1] < R.period_end_date(today, "1wk") or not j["live"]["1wk"]["is_live"]


def test_live_periode_nicht_in_historie():
    px = _fake_px(800)
    today = px.index[-1]
    if today.weekday() == 4:
        today = today - pd.Timedelta(days=1)
    w = R.resample_last(px, "1wk")
    per, lv = R.split_live(w, "1wk", today)
    assert lv
    c = R.combo(per, lv, "1wk", "simple", None, False)
    assert c["distribution"]["n"] == len(w) - 2   # erste Periode entfaellt, laufende Woche nicht dabei
    rstar = per.iloc[-1] / per.iloc[-2] - 1
    assert c["live_rank"]["r"] == pytest.approx(rstar)
    assert c["live_rank"]["live"] is True


def test_veraltete_kurse_werden_erkannt():
    mo, fr = pd.Timestamp("2026-11-23"), pd.Timestamp("2026-11-20")   # Montag / Freitag davor
    assert not R.is_stale(fr, mo, "yf:SPY", False)                    # normales Wochenende
    assert not R.is_stale(pd.Timestamp("2026-12-24"), pd.Timestamp("2026-12-29"), "yf:SPY", False)  # Feiertage
    assert R.is_stale(pd.Timestamp("2026-11-13"), mo, "yf:SPY", False)  # eine Woche alt
    assert not R.is_stale(pd.Timestamp("2026-11-22"), mo, "yf:BTC-USD", True)
    assert R.is_stale(pd.Timestamp("2026-11-19"), mo, "yf:BTC-USD", True)
    assert not R.is_stale(pd.Timestamp("2026-11-17"), mo, "fx:DEXUSEU:EUR", False)
    assert R.is_stale(pd.Timestamp("2026-11-13"), mo, "fx:DEXUSEU:EUR", False)  # nur Fed-Stand ohne EZB


def test_standardfehler_und_intervall(ra):
    D = R.describe(ra, 52)["distribution"]
    se = 0.023191397307857693 / math.sqrt(12)
    assert D["se_mean"] == pytest.approx(se, abs=1e-12)
    lo, hi = st.t.interval(0.95, 11, loc=-0.00175, scale=se)
    assert D["ci95"][0] == pytest.approx(lo, abs=1e-12) and D["ci95"][1] == pytest.approx(hi, abs=1e-12)
    # t-Test passt dazu: t = Mittel / SE
    assert R.describe(ra, 52)["tests"]["ttest"]["t"] == pytest.approx(-0.00175 / se)
