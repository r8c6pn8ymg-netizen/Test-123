"""Wendepunkt-Analyse: Stehen Makro ML, Faktor-Modell, CFTC und Zinsdifferenzen an grossen
Hoch-/Tiefpunkten (Wochencharts, ±26 bzw. ±13 Wochen) auf der richtigen Seite?

Aufruf: python3 research/turning_points.py <Ordner mit den TradingView-CSV-Exporten> [Fensterwochen]
Erwartet die Dateien aus FILES (CME-Futures 6E/6B/6C/6N/6S, OANDA USDJPY und AUDCHF, Wochenbasis).
Ergebnis (Stand 20.11.2026): kein Signal erkennt Wenden im Voraus; das CFTC-Gegensignal ist an
Wenden nur scheinbar stark (Positionierung folgt dem Trend), im Vorwaertstest nach starken Trends
erhoeht es die Wendewahrscheinlichkeit nur von 50,4 % auf 53,5 % (nicht signifikant).
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
import sys
U = (sys.argv[1].rstrip("/") + "/") if len(sys.argv) > 1 else "./"
FILES = {  # Name: (Datei, Waehrung/Paar, invertieren?)
    "EUR (6E)": ("92bea8c2-CME_DL_6E1_1W_3.csv", "EUR", False),
    "GBP (6B)": ("1a2f6db8-CME_DL_6B1_1W.csv", "GBP", False),
    "CAD (6C)": ("51929a8a-CME_DL_6C1_1W.csv", "CAD", False),
    "NZD (6N)": ("9fc02c75-CME_DL_6N1_1W.csv", "NZD", False),
    "CHF (6S)": ("d5d426e7-CME_DL_6S1_1W.csv", "CHF", False),
    "JPY (USDJPY invertiert)": ("28e2ba91-OANDA_USDJPY_1W_2.csv", "JPY", True),
    "AUD/CHF": ("13abb202-OANDA_AUDCHF_1W.csv", "AUD/CHF", False),
}
def load(fn, inv):
    d = pd.read_csv(U + fn)
    d["date"] = pd.to_datetime(d["time"], unit="s").dt.normalize()
    d = d.set_index("date")[["open", "high", "low", "close"]].astype(float)
    if inv:  # USDJPY -> JPY in USD: Hoch und Tief tauschen
        d = pd.DataFrame({"open": 1 / d.open, "high": 1 / d.low, "low": 1 / d.high, "close": 1 / d.close}, index=d.index)
    return d
import json
from scipy.stats import mannwhitneyu, fisher_exact
import pipeline as P, factors as Fm, data_sources as ds
px = P.load_fx(False); core = Fm.core(px); Z = core["z"]
OUT = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "output") + "/"
MS = json.load(open(OUT + "macro_score_alfred.json"))
def macro(c):
    d = MS.get(c); return pd.Series(d["v"], index=pd.to_datetime(d["d"])).sort_index() if d else None
def cot_contra(c):
    s = ds.cftc_net(P.COT_CODES[c], False)["net_pct"].dropna()
    pct = s.rolling(156, min_periods=104).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    out = (50 - pct) / 50; out.index = out.index + pd.Timedelta(days=3); return out.dropna()
def cot_trend(c):
    s = ds.cftc_net(P.COT_CODES[c], False)["net_pct"].dropna(); s.index = s.index + pd.Timedelta(days=3)
    return (s - s.rolling(156, min_periods=104).mean()) / s.rolling(156, min_periods=104).std()
# Zinsdifferenz 10J (Land minus USA), Veraenderung 3 Monate: Monatsmittel, bekannt zum Monatsende
sipri = ds.sipri_share_of_gdp(False)
def y10(cur):
    cc = {"EUR": ("XM", "EZ"), "GBP": ("GB", "GB"), "CAD": ("CA", "CA"), "NZD": ("NZ", "NZ"), "CHF": ("CH", "CH"), "JPY": ("JP", "JP"), "AUD": ("AU", "AU"), "USD": ("US", "US")}[cur]
    if cc[1] == "EZ": s = P.monthly(ds.ecb("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", False))
    else: s = ds.fred(f"IRLTLT01{cc[1]}M156N", False)
    s.index = s.index + pd.offsets.MonthEnd(0); return s
US10 = y10("USD")
def rate_mom(c):
    d = (y10(c) - US10).dropna(); return (d - d.shift(3)).dropna()
def sig_for(cur):
    """Alle Signale als 'bullish fuer die Waehrung (gegen USD bzw. AUD gegen CHF)'."""
    if cur == "AUD/CHF":
        a, b = sig_for("AUD"), sig_for("CHF")
        return {k: (a[k] - b[k].reindex(a[k].index, method="ffill")).dropna() for k in a if a[k] is not None and b[k] is not None}
    s = {}
    for k in ("kombi", "carry", "value", "momentum"):
        s["Faktor " + k] = (Z[k][cur] - Z[k]["USD"]).dropna()
    m = macro(cur); s["Makro ML"] = m
    s["CFTC Gegensignal"] = cot_contra(cur); s["CFTC Trend"] = cot_trend(cur)
    s["Zinsdiff. 10J Δ3M"] = rate_mom(cur)
    return s
def pivots(d, w):
    hi = d.high[(d.high == d.high.rolling(2 * w + 1, center=True).max())]
    lo = d.low[(d.low == d.low.rolling(2 * w + 1, center=True).min())]
    ev = [(t, "Hoch") for t in hi.index] + [(t, "Tief") for t in lo.index]
    return sorted(ev)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 26
rng = np.random.default_rng(3)
SIGS = ["Faktor kombi", "Faktor carry", "Faktor value", "Faktor momentum", "Makro ML", "CFTC Gegensignal", "CFTC Trend", "Zinsdiff. 10J Δ3M"]
rows = []           # je Ereignis und Signal
series_store = {}   # fuer Permutation
for name, (fn, cur, inv) in FILES.items():
    d = load(fn, inv); d = d[d.index >= "2007-01-01"]
    ev = pivots(d, W); sg = sig_for(cur)
    weeks = d.index
    for k in SIGS:
        s = sg.get(k)
        if s is None or len(s) == 0: continue
        s = s.sort_index(); s = s[~s.index.duplicated()]
        z = (s - s.mean()) / s.std()
        at = z.reindex(weeks, method="ffill")           # Stand zu Wochenbeginn (damals bekannt)
        series_store[(name, k)] = (at, ev)
        for t, kind in ev:
            v = at.get(t)
            if v is None or pd.isna(v): continue
            want = -1 if kind == "Hoch" else 1
            i = weeks.get_loc(t)
            pre = at.iloc[max(0, i - 13):i].dropna(); post = at.iloc[i + 1:i + 14].dropna()
            rows.append(dict(serie=name, signal=k, t=t, art=kind, v=float(v), ok=float(np.sign(v) == want),
                             pre=float((np.sign(pre) == want).mean()) if len(pre) else np.nan,
                             post=float((np.sign(post) == want).mean()) if len(post) else np.nan))
R = pd.DataFrame(rows)
def perm_p(k, n=2000):
    """Zufalls-Wendepunkte: Ereignisse je Serie zufaellig verschoben (Struktur bleibt erhalten)."""
    obs = R[R.signal == k].ok.mean()
    sims = []
    for _ in range(n):
        hits = []
        for (nm, kk), (at, ev) in series_store.items():
            if kk != k: continue
            valid = at.dropna(); 
            if len(valid) < 60: continue
            off = rng.integers(26, len(at) - 26)
            for t, kind in ev:
                j = (at.index.get_loc(t) + off) % len(at); v = at.iloc[j]
                if pd.notna(v): hits.append(float(np.sign(v) == (-1 if kind == "Hoch" else 1)))
        sims.append(np.mean(hits))
    return obs, float((np.array(sims) >= obs).mean())
print(f"Wendepunkt-Fenster ±{W} Wochen | Ereignisse gesamt: {R.drop_duplicates(['serie','t','art']).shape[0]}")
print(f"{'Signal':22s} {'N':>4s} {'richtig@Wende':>13s} {'Zufall-p':>8s} {'Wert Tief':>9s} {'Wert Hoch':>9s} {'MWU p':>7s} {'vorher(-13..-1)':>15s} {'nachher(+1..+13)':>16s}")
for k in SIGS:
    g = R[R.signal == k]
    if g.empty: continue
    obs, pp = perm_p(k)
    lo, hi = g[g.art == "Tief"].v, g[g.art == "Hoch"].v
    mw = mannwhitneyu(lo, hi, alternative="greater").pvalue
    print(f"{k:22s} {len(g):4d} {obs*100:12.1f}% {pp:8.3f} {lo.mean():9.2f} {hi.mean():9.2f} {mw:7.3f} {g.pre.mean()*100:14.1f}% {g.post.mean()*100:15.1f}%")
R.to_csv(f"events_w{W}.csv", index=False)
