"""Tests fuer den Strategie-Tester: makro_ml/strategie.js gegen die unabhaengige Python-Gegenrechnung
makro_ml/strategie_ref.py plus Handrechnungen. Offline (synthetische Kurse). Braucht Node; ohne Node -> skip.
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "makro_ml"))
import returns_stats as R  # noqa: E402
import strategie_ref as SR  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js fehlt")


def run_js(dates, close, ends, configs, crypto=False, **extra):
    d0 = dates[0]
    inp = {"series": {"d0": str(d0.date()), "d": [(d - d0).days for d in dates], "c": [float(v) for v in close]},
           "ends": ends, "crypto": crypto, "configs": configs} | extra
    out = subprocess.run([NODE, str(ROOT / "tests" / "run_strategie.js")], input=json.dumps(inp),
                         capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout)


def synth(n=2600, seed=1, crypto=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-04", periods=int(n * 1.5), freq="D" if crypto else "B")
    if not crypto:  # ein paar Feiertage
        idx = idx[rng.random(idx.size) > 0.02]
    idx = idx[:n]
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    return idx, px


def ends_for(idx, px, today):
    s = pd.Series(px, index=idx)
    out = {}
    for iv in ("1wk", "1mo"):
        per = R.resample_last(s, iv)
        _, live = R.split_live(per, iv, today)
        e = per.iloc[:-1] if live else per
        out[iv] = s.index.get_indexer(e.index).tolist()
    return out


CONFIGS = [
    {"interval": "1d", "dir": "long", "conds": [{"type": "always"}]},
    {"interval": "1wk", "dir": "long", "conds": [{"type": "ret", "op": "lt", "th": -0.02}], "exit": {"maxHold": 3},
     "cost": {"fee_bp": 2, "spread_bp": 3, "slip_bp": 1, "fin_pa": 4}},
    {"interval": "1mo", "dir": "long", "conds": [{"type": "price_sma", "n": 6, "op": "gt"}], "exit": {"signal": True},
     "cost": {"fee_bp": 5}},
    {"interval": "1wk", "dir": "short", "conds": [{"type": "sma_sma", "a": 4, "b": 12, "op": "lt"}],
     "exit": {"signal": True, "sl": 4, "tp": 6}, "cost": {"spread_bp": 2}, "lag": 0},
    {"interval": "1d", "dir": "ls", "conds": [{"type": "mom", "n": 60, "op": "gt", "th": 0}], "exit": {"maxHold": 20},
     "lev": 2, "size": 0.5, "cost": {"fee_bp": 1, "fin_pa": 3}},
    {"interval": "1wk", "dir": "long", "conds": [{"type": "z", "n": 26, "op": "lt", "th": -1.5},
                                                  {"type": "vol", "n": 13, "op": "lt", "th": 0.3}],
     "exit": {"trail": 3, "maxHold": 8}, "start": "2011-06-01", "end": "2018-12-31", "split": "2016-01-01"},
    {"interval": "1d", "dir": "long", "conds": [{"type": "rsi", "n": 14, "op": "lt", "th": 35}],
     "exit": {"tp": 3, "sl": 2, "maxHold": 15}, "cost": {"fee_bp": 3, "slip_bp": 2}},
    {"interval": "1mo", "dir": "long", "conds": [{"type": "hilo", "n": 12, "side": "high", "th": 0.03},
                                                  {"type": "calendar", "months": [1, 2, 3, 4, 10, 11, 12]}],
     "exit": {"signal": True}},
    {"interval": "1d", "dir": "short", "conds": [{"type": "hilo", "n": 50, "side": "low", "th": 0.02},
                                                 {"type": "calendar", "weekdays": [1, 2, 3]}],
     "exit": {"maxHold": 5, "trail": 2}, "lag": 0, "cost": {"fee_bp": 1}},
]


@pytest.fixture(scope="module")
def market():
    idx, px = synth()
    today = idx[-1] + pd.Timedelta(days=10)
    return idx, px, ends_for(idx, px, today), today


@pytest.fixture(scope="module")
def js_results(market):
    idx, px, ends, today = market
    return run_js(idx, px, ends, CONFIGS, today=str(today.date()))


@pytest.mark.parametrize("k", range(len(CONFIGS)))
def test_js_gleich_python(market, js_results, k):
    idx, px, ends, _ = market
    cfg = CONFIGS[k]
    js = js_results["results"][k]
    assert "error" not in js, js.get("error")
    py = SR.backtest(idx, px, ends, cfg)
    assert js["days"] == py["days"].tolist()
    assert np.allclose(js["g"], py["g"], atol=1e-12, rtol=0), "Tagesrenditen weichen ab"
    assert len(js["trades"]) == len(py["trades"])
    for a, b in zip(js["trades"], py["trades"]):
        assert (a["entryIdx"], a["exitIdx"], a["dir"], a["reason"]) == (b["entryIdx"], b["exitIdx"], b["dir"], b["reason"])
        assert a["ret"] == pytest.approx(b["ret"], abs=1e-12)
    assert js["exposure"] == pytest.approx(py["exposure"], abs=1e-12)
    for part, key in (("all", "all"), ("bh", "bh"), ("is", "is"), ("oos", "oos")):
        mj, mp = js["metrics"][part], py[key]
        for f in ("n", "total", "cagr", "vol", "sharpe", "sortino", "maxdd", "dd_days", "calmar", "t", "years"):
            if mp[f] is None:
                assert mj[f] is None, f
            else:
                assert mj[f] == pytest.approx(mp[f], rel=1e-9, abs=1e-12), f"{part}.{f}"
        assert mj["p"] == pytest.approx(mp["p"], abs=1e-7), f"{part}.p"
    assert len(py["trades"]) > 0 or k == 0


def test_regeln_wurden_wirklich_ausgeloest(js_results):
    """Damit der Vergleich etwas prueft: jede Konfiguration handelt, Stops/Ziele/Wechsel kommen vor."""
    reasons = {t["reason"] for r in js_results["results"] for t in r["trades"]}
    assert {"Stop", "Ziel", "Trailing", "Signal", "Haltedauer", "Wechsel", "Ende"} <= reasons
    assert all(len(r["trades"]) >= 1 for r in js_results["results"])


def test_periodenenden_js_gleich_python(market, js_results):
    _, _, ends, _ = market
    assert js_results["ends"] == ends


def test_periodenenden_letzte_periode_laufend():
    idx = pd.bdate_range("2024-01-01", "2024-03-13")
    px = np.linspace(100, 120, idx.size)
    today = pd.Timestamp("2024-03-14")
    ends = ends_for(idx, px, today)
    js = run_js(idx, px, ends, [], today="2024-03-14")
    assert js["ends"] == ends
    assert idx[ends["1mo"][-1]] == pd.Timestamp("2024-02-29")   # Maerz laeuft noch
    assert idx[ends["1wk"][-1]] == pd.Timestamp("2024-03-08")   # Woche bis Fr 15.3. laeuft noch


def test_handrechnung_kosten_finanzierung_lag():
    # 6 Handelstage, Mo-Fr + Mo; Signal immer, Long, lag 1 -> Einstieg Di-Schluss
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    px = np.array([100.0, 100.0, 110.0, 99.0, 99.0, 108.9])
    cfg = {"interval": "1d", "dir": "long", "conds": [{"type": "always"}],
           "cost": {"fee_bp": 10, "spread_bp": 20, "slip_bp": 5, "fin_pa": 36.5}}
    side = (10 + 10 + 5) / 1e4
    js = run_js(idx, px, {"1wk": [], "1mo": []}, [cfg])["results"][0]
    py = SR.backtest(idx, px, {"1wk": [], "1mo": []}, cfg)
    # g ab Tag 2 (Tag 1 = Bewertungsbeginn); Finanzierung 36,5 % p. a. = 0,1 %/Kalendertag, Montag: 3 Tage
    exp = [-side, 0.10 - 0.001, -0.10 - 0.001, 0.0 - 0.001, 0.10 - 0.003]
    assert np.allclose(js["g"], exp, atol=1e-12) and np.allclose(py["g"], exp, atol=1e-12)
    assert js["trades"][0]["reason"] == "Ende" and js["trades"][0]["entryIdx"] == 1
    assert js["trades"][0]["ret"] == pytest.approx((1 - side) * np.prod([1 + v for v in exp[1:]]) - 1)
    # lag 0: am ersten Tag keine Ausfuehrung, Einstieg am Schluss von Tag 2 (Kosten zaehlen)
    js0 = run_js(idx, px, {"1wk": [], "1mo": []}, [cfg | {"lag": 0}])["results"][0]
    assert js0["g"][0] == pytest.approx(-side) and js0["trades"][0]["entryIdx"] == 1
    assert js0["metrics"]["all"]["total"] == pytest.approx(np.prod([1 + v for v in js0["g"]]) - 1)


def test_handrechnung_stop_mit_kursluecke():
    idx = pd.bdate_range("2024-01-01", periods=6)
    px = np.array([100.0, 100.0, 97.0, 90.0, 95.0, 96.0])
    cfg = {"interval": "1d", "dir": "long", "conds": [{"type": "always"}], "exit": {"sl": 5}, "lag": 0}
    js = run_js(idx, px, {"1wk": [], "1mo": []}, [cfg])["results"][0]
    t0 = js["trades"][0]
    assert (t0["entryIdx"], t0["exitIdx"], t0["reason"]) == (1, 3, "Stop")
    assert t0["ret"] == pytest.approx(-0.10)   # Stop bei -5 %, aber Schluss erst bei -10 % (Kursluecke)
    assert js["trades"][1]["entryIdx"] == 4     # Wiedereinstieg erst am naechsten Periodenende


def test_kein_look_ahead_signale(market):
    idx, px, ends, today = market
    cut = 1500
    px2 = px.copy()
    px2[cut + 1:] *= np.linspace(0.3, 3.0, px.size - cut - 1)   # Zukunft voellig anders
    a = run_js(idx, px, ends, CONFIGS)["results"]
    b = run_js(idx, px2, ends, CONFIGS)["results"]
    for k, cfg in enumerate(CONFIGS):
        if cfg.get("end") and pd.Timestamp(cfg["end"]) < idx[cut]:
            continue
        ga, gb = np.array(a[k]["g"]), np.array(b[k]["g"])
        days = np.array(a[k]["days"])
        m = days <= cut
        assert np.array_equal(ga[m], gb[m]), f"Konfiguration {k}: Vergangenheit haengt von der Zukunft ab"


def test_ruin_bei_hohem_hebel():
    idx = pd.bdate_range("2024-01-01", periods=5)
    px = np.array([100.0, 100.0, 60.0, 61.0, 62.0])
    cfg = {"interval": "1d", "dir": "long", "conds": [{"type": "always"}], "lev": 3, "lag": 0}
    js = run_js(idx, px, {"1wk": [], "1mo": []}, [cfg])["results"][0]
    assert js["trades"][0]["reason"] == "Ruin"
    assert js["metrics"]["all"]["total"] == pytest.approx(-1.0)
    assert js["trades"][0]["ret"] == pytest.approx(-1.0) and js["g"][1:] == [-1.0, 0.0, 0.0]
    py = SR.backtest(idx, px, {"1wk": [], "1mo": []}, cfg)
    assert np.allclose(py["g"], js["g"]) and py["trades"][0]["reason"] == "Ruin"


def test_mathe_hilfen():
    ps = [1e-10, 0.001, 0.02, 0.5, 0.975, 0.999999]
    ts = [(2.0, 100), (-3.5, 10), (0.1, 5000), (8.0, 30)]
    ems = [(100, 30), (2, 10), (1000, 5)]
    js = run_js(*synth(50)[0:1], np.ones(50) * 100, {"1wk": [], "1mo": []}, [],
                math={"p": ps, "t": ts, "ems": ems})["math"]
    assert np.allclose(js["normInv"], st.norm.ppf(ps), atol=1e-9)
    assert np.allclose(js["t"], [2 * st.t.sf(abs(t), df) for t, df in ts], rtol=1e-8, atol=1e-14)
    g = 0.5772156649
    exp = [math.sqrt(1 / y) * ((1 - g) * st.norm.ppf(1 - 1 / m) + g * st.norm.ppf(1 - 1 / (m * math.e))) for m, y in ems]
    assert np.allclose(js["ems"], exp, atol=1e-9)


def test_placebo_reproduzierbar_und_sinnvoll(market):
    idx, px, ends, _ = market
    cfg = CONFIGS[1]
    a = run_js(idx, px, ends, [cfg], placebo=True)["results"][0]["placebo"]
    b = run_js(idx, px, ends, [cfg], placebo=True)["results"][0]["placebo"]
    assert a["sharpes"] == b["sharpes"]              # fester Seed
    assert 0 < a["p"] <= 1 and len(a["sharpes"]) == 50
    # Zufallsdaten ohne Vorteil: p soll nicht extrem klein sein
    assert a["p"] > 0.01
