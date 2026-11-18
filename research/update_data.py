"""Lädt die aktuellen AAII- und CNN-Daten und baut alle abgeleiteten Dateien neu.

Aufruf:  python3 research/update_data.py
Benötigt Netz-Zugriff auf www.aaii.com und production.dataviz.cnn.io.
BofA-Werte (monatlich) werden von Hand in data/bofa_sent.csv und data/bofa_cash.csv ergänzt.
"""
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DASH = os.path.join(HERE, "dashboard")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_aaii():
    raw = get("https://www.aaii.com/files/surveys/sentiment.xls")
    x = pd.read_excel(io.BytesIO(raw), sheet_name="SENTIMENT", header=None, skiprows=5)
    x = x[[0, 1, 2, 3, 12]]
    x.columns = ["date", "bull", "neutral", "bear", "spx"]
    x["date"] = pd.to_datetime(x.date, errors="coerce")
    x = x.dropna(subset=["date", "bull", "bear"])
    x["spread"] = x.bull - x.bear
    x[["date", "bull", "neutral", "bear", "spread"]].to_csv(os.path.join(DATA, "aaii.csv"), index=False)
    y = pd.read_excel(io.BytesIO(raw), sheet_name="SENTIMENT", header=None, skiprows=5)[[0, 10, 11, 12]]
    y.columns = ["date", "hi", "lo", "spx"]
    y["date"] = pd.to_datetime(y.date, errors="coerce")
    y.dropna().to_csv(os.path.join(DATA, "spx_weekly.csv"), index=False)
    print("AAII bis", x.date.max().date())


def fetch_cnn():
    raw = get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-07-15",
              {"Referer": "https://edition.cnn.com/", "Accept": "application/json"})
    d = json.loads(raw)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(v.get("data"), list):
            out[k] = pd.Series({pd.to_datetime(p["x"], unit="ms").normalize(): p["y"] for p in v["data"]})
    df = pd.DataFrame(out)
    df.index.name = "date"
    df.to_csv(os.path.join(DATA, "cnn.csv"))
    print("CNN bis", df.index.max().date())


def rsi(s, n):
    d = s.diff()
    u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn)


def weekly_start(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("W-SUN").start_time
    return s.groupby(level=0).last()


def build_dashboard():
    os.makedirs(DASH, exist_ok=True)
    a = pd.read_csv(os.path.join(DATA, "aaii.csv"), parse_dates=["date"]).set_index("date")["spread"] * 100
    spxf = pd.read_csv(os.path.join(DATA, "spx_weekly.csv"), parse_dates=["date"]).set_index("date")
    spx = spxf["spx"]
    cnn = pd.read_csv(os.path.join(DATA, "cnn.csv"), index_col=0, parse_dates=True)
    cnn_fg = cnn["fear_and_greed_historical"].dropna()
    # S&P-Wochenkurs: AAII-Datei, ab 2020 ergänzt um CNNs täglichen S&P-Schlusskurs (aktueller)
    spx_w = weekly_start(spx)
    cnn_spx = weekly_start(cnn["market_momentum_sp500"].dropna())
    spx_w = spx_w.combine_first(cnn_spx)
    spx_w.update(cnn_spx)

    def mon(name, col):
        s = pd.read_csv(os.path.join(DATA, name), index_col=0, parse_dates=True)[col]
        s.index = s.index + pd.Timedelta(days=19)
        return weekly_start(s)

    w = pd.DataFrame({"spx": spx_w, "aaii": weekly_start(a), "cnn": weekly_start(cnn_fg),
                      "bofa": mon("bofa_sent.csv", "bofa_sent"), "cash": mon("bofa_cash.csv", "bofa_cash")})
    w = w.sort_index()
    a_w = weekly_start(a)
    w["aaii4"] = a_w.rolling(4).mean().reindex(w.index)
    w[["aaii", "aaii4", "bofa", "cash"]] = w[["aaii", "aaii4", "bofa", "cash"]].ffill()
    w = w[w.index >= "1999-01-01"].dropna(subset=["spx"])
    # Wochen-Oszillator auf dem S&P (nur Schlusskurse verfügbar → Spanne aus Schlusskursen)
    c = w.spx
    hi = weekly_start(spxf["hi"]).reindex(w.index) if "hi" in spxf else None
    lo = weekly_start(spxf["lo"]).reindex(w.index) if "lo" in spxf else None
    hi = c if hi is None else hi.fillna(c).clip(lower=c)
    lo = c if lo is None else lo.fillna(c).clip(upper=c)
    st = (c - lo.rolling(13).min()) / (hi.rolling(13).max() - lo.rolling(13).min()) * 100
    z = ((c - c.rolling(20).mean()) / c.rolling(20).std() * 20 + 50).clip(0, 100)
    w["osc"] = (rsi(c, 9) + st + z) / 3
    s_aaii = (w.aaii4.clip(-30, 30) + 30) / 60 * 100
    num = 2 * w.osc + s_aaii.fillna(0) + w.cnn.fillna(0)
    den = 2 + s_aaii.notna() + w.cnn.notna()
    w["score"] = num / den
    w["zone"] = np.select([w.score >= 80, w.score >= 75, w.score <= 20, w.score <= 25],
                          ["Extreme Gier", "Gier", "Extreme Angst", "Angst"], "Neutral")
    w.index.name = "week"
    out = w.reset_index()
    out["week"] = out.week.dt.strftime("%Y-%m-%d")
    out = out[out.week >= "2005-01-01"].round(2)
    out.to_csv(os.path.join(DASH, "sentiment_weekly.csv"), index=False)
    print("Dashboard-Daten:", len(out), "Wochen bis", out.week.iloc[-1], "Score", out.score.iloc[-1])


if __name__ == "__main__":
    import sys
    if "--offline" not in sys.argv:
        fetch_aaii()
        fetch_cnn()
    build_dashboard()
    import make_pine
    make_pine.main()
