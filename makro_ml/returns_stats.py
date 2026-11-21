"""Renditestatistik fuer den Cockpit-Reiter "Statistik" (Definitionen: SPEC_STATISTIK.md).

Reine Rechenfunktionen (ohne Netz) plus Datenabruf und Ausgabe nach ``output/``:
  output/stats_index.json      Instrumentliste, Makro-Overlay
  output/stats/<id>.json       je Instrument alle Varianten (Intervall x Rendite-Typ x Zeitraum)

Aufruf aus makro_ml/:  python3 returns_stats.py [--cache]
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

import data_sources as ds

OUT = Path(__file__).parent / "output"
ALPHAS = (0.05, 0.01)
QS = (0.01, 0.05, 0.95, 0.99)
INTERVALS = ("1d", "1wk", "1mo")
TYPES = ("simple", "log")
PERIODS = {"max": None, "20y": 20, "10y": 10, "5y": 5, "1y": 1}
BINS = ("auto", "50")
WF_DEFAULT = {"1d": (756, 252), "1wk": (156, 52), "1mo": (36, 12)}
MIN_N = 30
# Stephens (D'Agostino/Stephens 1986, mu und sigma geschaetzt), fuer A*^2 = A^2 (1 + 0.75/N + 2.25/N^2)
AD_CRIT = {0.05: 0.752, 0.01: 1.035}

# id, Name, Gruppe, Quelle:Ticker, Hinweis
INSTRUMENTS = [
    ("GSPC", "S&P 500 (Kursindex)", "Aktien", "yf:^GSPC", "Kursindex ohne Dividenden"),
    ("SPY", "SPDR S&P 500 ETF", "Aktien", "yf:SPY", ""),
    ("QQQ", "Invesco QQQ (Nasdaq 100)", "Aktien", "yf:QQQ", ""),
    ("NDX", "Nasdaq 100 (Kursindex)", "Aktien", "yf:^NDX", "Kursindex ohne Dividenden"),
    ("DJI", "Dow Jones Industrial", "Aktien", "yf:^DJI", "Kursindex ohne Dividenden"),
    ("RUT", "Russell 2000", "Aktien", "yf:^RUT", "Kursindex ohne Dividenden"),
    ("GDAXI", "DAX", "Aktien", "yf:^GDAXI", "Performanceindex (mit Dividenden)"),
    ("STOXX50E", "Euro Stoxx 50", "Aktien", "yf:^STOXX50E", "Kursindex ohne Dividenden"),
    ("FTSE", "FTSE 100", "Aktien", "yf:^FTSE", "Kursindex ohne Dividenden"),
    ("FCHI", "CAC 40", "Aktien", "yf:^FCHI", "Kursindex ohne Dividenden"),
    ("SSMI", "SMI", "Aktien", "yf:^SSMI", "Kursindex ohne Dividenden"),
    ("N225", "Nikkei 225", "Aktien", "yf:^N225", "Kursindex ohne Dividenden"),
    ("HSI", "Hang Seng", "Aktien", "yf:^HSI", "Kursindex ohne Dividenden"),
    ("EEM", "iShares MSCI Emerging Markets", "Aktien", "yf:EEM", ""),
    ("URTH", "iShares MSCI World", "Aktien", "yf:URTH", ""),
    ("TLT", "US-Staatsanleihen 20J+ (TLT)", "Anleihen", "yf:TLT", ""),
    ("IEF", "US-Staatsanleihen 7–10J (IEF)", "Anleihen", "yf:IEF", ""),
    ("SHY", "US-Staatsanleihen 1–3J (SHY)", "Anleihen", "yf:SHY", ""),
    ("TIP", "Inflationsgeschützte US-Anleihen (TIP)", "Anleihen", "yf:TIP", ""),
    ("LQD", "US-Unternehmensanleihen IG (LQD)", "Anleihen", "yf:LQD", ""),
    ("HYG", "US-Hochzinsanleihen (HYG)", "Anleihen", "yf:HYG", ""),
    ("AGG", "US-Anleihen gesamt (AGG)", "Anleihen", "yf:AGG", ""),
    ("GC_F", "Gold (Future)", "Rohstoffe", "yf:GC=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("SI_F", "Silber (Future)", "Rohstoffe", "yf:SI=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("HG_F", "Kupfer (Future)", "Rohstoffe", "yf:HG=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("CL_F", "WTI-Öl (Future)", "Rohstoffe", "yf:CL=F",
     "Front-Monat, Roll-Sprünge nicht bereinigt; April 2020 negativer Preis (entfällt)"),
    ("BZ_F", "Brent-Öl (Future)", "Rohstoffe", "yf:BZ=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("NG_F", "Erdgas (Future)", "Rohstoffe", "yf:NG=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("ZW_F", "Weizen (Future)", "Rohstoffe", "yf:ZW=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("ZC_F", "Mais (Future)", "Rohstoffe", "yf:ZC=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("ZS_F", "Sojabohnen (Future)", "Rohstoffe", "yf:ZS=F", "Front-Monat, Roll-Sprünge nicht bereinigt"),
    ("GLD", "SPDR Gold Shares (GLD)", "Rohstoffe", "yf:GLD", ""),
    ("SLV", "iShares Silver (SLV)", "Rohstoffe", "yf:SLV", ""),
    ("USO", "US Oil Fund (USO)", "Rohstoffe", "yf:USO", "enthält Roll-Kosten"),
    ("DBC", "Invesco Rohstoff-Korb (DBC)", "Rohstoffe", "yf:DBC", "enthält Roll-Kosten"),
    ("EURUSD", "EUR/USD", "Währungen", "fx:DEXUSEU:EUR", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("GBPUSD", "GBP/USD", "Währungen", "fx:DEXUSUK:GBP", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("AUDUSD", "AUD/USD", "Währungen", "fx:DEXUSAL:AUD", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("NZDUSD", "NZD/USD", "Währungen", "fx:DEXUSNZ:NZD", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("USDJPY", "USD/JPY", "Währungen", "fx:DEXJPUS:JPY", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("USDCHF", "USD/CHF", "Währungen", "fx:DEXSZUS:CHF", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("USDCAD", "USD/CAD", "Währungen", "fx:DEXCAUS:CAD", "Fed H.10 (12 Uhr New York), verlängert mit EZB-Kursen"),
    ("DXY", "US-Dollar-Index", "Währungen", "yf:DX-Y.NYB",
     "gehandelter ICE-Index (Yahoo); in den anderen Reitern aus Fed-Kursen berechnet, leichte Abweichungen möglich"),
    ("BTC_USD", "Bitcoin", "Krypto", "yf:BTC-USD", "Handel an 7 Tagen"),
    ("ETH_USD", "Ethereum", "Krypto", "yf:ETH-USD", "Handel an 7 Tagen"),
    ("VIX", "VIX (Volatilitätsindex)", "Volatilität", "yf:^VIX", "nicht direkt handelbar"),
]


# ----------------------------------------------------------------------------- Daten

def round_sig(o, sig: int = 6):
    """Alle Gleitkommazahlen auf ``sig`` signifikante Stellen (nur fuer die Ausgabe, Dateigroesse)."""
    if isinstance(o, dict):
        return {k: round_sig(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_sig(v, sig) for v in o]
    if isinstance(o, float) and math.isfinite(o):
        return float(f"{o:.{sig}g}")
    return o


def is_crypto(source: str) -> bool:
    t = source.split(":")[1]
    return t.endswith(("-USD", "-EUR", "-USDT"))


def clean_prices(s: pd.Series) -> pd.Series:
    """Duplikate (letzter Wert), Sortierung, Zeitzone weg, Kurse <= 0 und NaN verwerfen."""
    s = pd.Series(pd.to_numeric(s, errors="coerce"), index=pd.DatetimeIndex(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[np.isfinite(s) & (s > 0)]
    return s.astype(float)


def _yf(ticker: str, cid: str, refresh: bool) -> pd.Series:
    path = ds.CACHE / f"yf_{cid}.csv"
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < 20 * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        import yfinance as yf
        d = None
        for i in range(4):
            d = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
            if d is not None and len(d):
                break
            time.sleep(3 * (i + 1))
        if d is None or not len(d):
            raise RuntimeError(f"keine Daten fuer {ticker}")
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.name = "close"
        close.to_csv(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        return close
    except Exception:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        raise


def _fx(fred_id: str, cur: str, refresh: bool, ecb: pd.DataFrame | None) -> pd.Series:
    s = ds.fred(fred_id, refresh)
    if ecb is not None and cur in ecb:
        e = ecb[cur]  # USD je Einheit
        if fred_id in ("DEXJPUS", "DEXSZUS", "DEXCAUS"):  # FRED-Reihe ist Einheit je USD
            e = 1 / e
        s, _ = ds.live_extend(s, e)
    return s


def load_instrument(source: str, cid: str, refresh: bool, ecb: pd.DataFrame | None = None) -> pd.Series:
    kind, rest = source.split(":", 1)
    if kind == "yf":
        raw = _yf(rest, cid, refresh)
    else:
        fid, cur = rest.split(":")
        raw = _fx(fid, cur, refresh, ecb)
    return clean_prices(raw)


# ----------------------------------------------------------------------------- Perioden

def period_end_date(d: pd.Timestamp, interval: str) -> pd.Timestamp:
    if interval == "1d":
        return d
    if interval == "1wk":
        return d + pd.Timedelta(days=(4 - d.weekday()) % 7)  # Freitag derselben Sa-Fr-Woche
    return d + pd.offsets.MonthEnd(0)


def resample_last(px: pd.Series, interval: str) -> pd.Series:
    """Letzter Kurs je Periode, Index = tatsaechlicher letzter Handelstag der Periode."""
    if interval == "1d":
        return px.copy()
    key = px.index.to_period("W-FRI" if interval == "1wk" else "M")
    return px.groupby(key).tail(1)


def split_live(per: pd.Series, interval: str, today: pd.Timestamp) -> tuple[pd.Series, bool]:
    """(Perioden-Schlusskurse, letzte Periode laeuft noch?). Laufend, wenn Periodenende >= heute."""
    if per.empty:
        return per, False
    return per, bool(period_end_date(per.index[-1], interval) >= today.normalize())


def to_returns(p: pd.Series, rtype: str) -> pd.Series:
    r = np.log(p / p.shift(1)) if rtype == "log" else p / p.shift(1) - 1
    return r.iloc[1:]


def ann_factor(interval: str, crypto: bool) -> int:
    return {"1d": 365 if crypto else 252, "1wk": 52, "1mo": 12}[interval]


# ----------------------------------------------------------------------------- Kennzahlen

def var_es_hist(r: np.ndarray, alpha: float) -> tuple[float, float]:
    q = float(np.quantile(r, alpha, method="linear"))
    tail = r[r <= q]
    if tail.size == 0:  # kann bei linearer Interpolation nicht vorkommen, Absicherung
        tail = np.array([r.min()])
    return -q, -float(tail.mean())


def var_es_param(mu: float, sd: float, alpha: float) -> tuple[float, float]:
    z = st.norm.ppf(alpha)
    return -(mu + sd * z), -(mu - sd * st.norm.pdf(z) / alpha)


def ad_pvalue(a2_star: float) -> float:
    a = a2_star
    if a >= 153.0:  # Formel hat bei 5.709/(2*0.0186) ~ 153.5 ihr Minimum (p ~ 1e-190), danach ungueltig
        return 0.0
    if a >= 0.6:
        p = math.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
    elif a >= 0.34:
        p = math.exp(0.9177 - 4.279 * a - 1.38 * a * a)
    elif a >= 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * a - 59.938 * a * a)
    else:
        p = 1 - math.exp(-13.436 + 101.14 * a - 223.73 * a * a)
    return min(max(p, 0.0), 1.0)


def anderson(r: np.ndarray) -> dict:
    n = r.size
    a2 = float(st.anderson(r, dist="norm", method="interpolate").statistic)
    a2s = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    return {"a2": a2, "a2_star": a2s, "p": ad_pvalue(a2s),
            "normal": {str(1 - a): bool(a2s <= c) for a, c in AD_CRIT.items()}}


def hist_edges(r: np.ndarray, bins: str) -> np.ndarray:
    lo, hi = float(r.min()), float(r.max())
    if bins == "auto":
        e = np.histogram_bin_edges(r, bins="fd")
        h = float(e[1] - e[0]) if e.size > 1 else 0.0
        if not np.isfinite(h) or h <= 0 or e.size > 2000:  # IQR = 0 -> Sturges
            e = np.histogram_bin_edges(r, bins="sturges")
            h = float(e[1] - e[0])
    else:
        h = (hi - lo) / int(bins)
    if h <= 0:
        h = max(abs(hi), 1e-6)
    k0 = math.floor(lo / h + 1e-9)
    k1 = math.ceil(hi / h - 1e-9)
    if k1 <= k0:
        k1 = k0 + 1
    return np.arange(k0, k1 + 1) * h


def chi2_gof(r: np.ndarray, edges: np.ndarray, mu: float, sd: float) -> dict:
    n = r.size
    obs, _ = np.histogram(r, bins=edges)
    cdf = st.norm.cdf(edges[1:-1], mu, sd)
    p = np.diff(np.concatenate([[0.0], cdf, [1.0]]))  # aeussere Bins nach +-unendlich offen
    o, e = list(obs.astype(float)), list(p * n)
    # von links zusammenlegen
    while len(e) > 1 and e[0] < 5:
        e[1] += e[0]; o[1] += o[0]; e.pop(0); o.pop(0)
    while len(e) > 1 and e[-1] < 5:
        e[-2] += e[-1]; o[-2] += o[-1]; e.pop(); o.pop()
    i = 0
    while i < len(e) and len(e) > 1:  # innere Bins mit e < 5 mit rechtem (zuletzt linkem) Nachbarn
        if e[i] < 5:
            j = i + 1 if i + 1 < len(e) else i - 1
            e[j] += e[i]; o[j] += o[i]; e.pop(i); o.pop(i)
            i = max(0, min(i, j))
            continue
        i += 1
    eo, ee = np.array(o), np.array(e)
    chi = float(((eo - ee) ** 2 / ee).sum())
    ndf = len(ee) - 3
    if ndf < 1:
        return {"chi2": chi, "ndf": ndf, "chi2_ndf": None, "p": None, "bins_used": len(ee)}
    return {"chi2": chi, "ndf": ndf, "chi2_ndf": chi / ndf, "p": float(st.chi2.sf(chi, ndf)), "bins_used": len(ee)}


def histogram(r: np.ndarray, bins: str, mu: float, sd: float) -> dict:
    edges = hist_edges(r, bins)
    counts, _ = np.histogram(r, bins=edges)
    n, h = r.size, float(edges[1] - edges[0])
    expct = n * np.diff(st.norm.cdf(edges, mu, sd))
    colors = ["neg" if b <= 1e-15 else ("pos" if a >= -1e-15 else "mix") for a, b in zip(edges[:-1], edges[1:])]
    x = np.linspace(edges[0], edges[-1], 120)
    y = n * h * st.norm.pdf((x - mu) / sd) / sd
    return {"edges": edges.tolist(), "counts": counts.tolist(), "colors": colors, "expected": expct.tolist(),
            "normal_curve_x": x.tolist(), "normal_curve_y": y.tolist(), "width": h,
            "chi2": chi2_gof(r, edges, mu, sd)}


def qq_points(r: np.ndarray, max_pts: int = 400, tail: int = 20) -> dict:
    n = r.size
    y = np.sort(r)
    x = st.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    if n > max_pts:
        mid = np.linspace(tail, n - tail - 1, max_pts - 2 * tail).round().astype(int)
        idx = np.unique(np.concatenate([np.arange(tail), mid, np.arange(n - tail, n)]))
        x, y = x[idx], y[idx]
    return {"x": x.tolist(), "y": y.tolist()}


def live_rank(hist: np.ndarray, r_star: float, mu: float, sd: float, first: pd.Timestamp, last: pd.Timestamp) -> dict:
    n = hist.size
    z = (r_star - mu) / sd
    worse = int((hist < r_star).sum())
    better = int((hist > r_star).sum())
    return {"r": r_star, "n": n, "worse_count": worse, "at_least_as_bad": int((hist <= r_star).sum()) + 1,
            "better_count": better, "at_least_as_good": int((hist >= r_star).sum()) + 1,
            "percentile": 100.0 * worse / n, "z": z,
            "normal_expect": n * float(st.norm.cdf(z)) if r_star <= mu else n * float(st.norm.sf(z)),
            "years": int(round((last - first).days / 365.25))}


def describe(r: pd.Series, k: int) -> dict:
    """Alle Kennzahlen aus SPEC Abschnitt 3 (ohne Histogramm/Live)."""
    x = r.to_numpy(dtype=float)
    n = x.size
    mu, md, sd = float(np.mean(x)), float(np.median(x)), float(np.std(x, ddof=1))
    ib, iw = int(np.argmax(x)), int(np.argmin(x))
    up, dn = int((x > 0).sum()), int((x < 0).sum())
    out = {
        "distribution": {"n": n, "mean": mu, "median": md, "sigma": sd, "ann_vol": sd * math.sqrt(k), "k": k,
                         # Standardfehler des Mittelwerts und 95-%-Intervall (t-Verteilung, df = N - 1)
                         "se_mean": sd / math.sqrt(n),
                         "ci95": [mu - float(st.t.ppf(0.975, n - 1)) * sd / math.sqrt(n),
                                  mu + float(st.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)],
                         "skew": float(st.skew(x, bias=False)),
                         "kurt": float(st.kurtosis(x, fisher=True, bias=False)),
                         "best": x[ib], "best_date": str(r.index[ib].date()),
                         "worst": x[iw], "worst_date": str(r.index[iw].date()),
                         "up": up, "down": dn, "zero": n - up - dn, "up_pct": 100 * up / n, "down_pct": 100 * dn / n,
                         "start": str(r.index[0].date()), "end": str(r.index[-1].date())},
        "quantiles": {str(q): {"obs": float(np.quantile(x, q, method="linear")),
                               "normal": mu + sd * float(st.norm.ppf(q))} for q in QS},
        "var_es": {},
        "tails": {},
    }
    for a in ALPHAS:
        hv, he = var_es_hist(x, a)
        pv, pe = var_es_param(mu, sd, a)
        out["var_es"][str(1 - a)] = {"var_hist": hv, "es_hist": he, "var_param": pv, "es_param": pe}
    for kk in (1, 2, 3):
        c = int((np.abs(x - mu) > kk * sd).sum())
        out["tails"][str(kk)] = {"count": c, "pct": 100 * c / n, "normal_pct": 100 * 2 * float(st.norm.sf(kk))}
    jb = st.jarque_bera(x)
    tt = st.ttest_1samp(x, 0)
    sw = st.shapiro(x) if n <= 5000 else None
    out["tests"] = {
        "jarque_bera": {"stat": float(jb.statistic), "p": float(jb.pvalue), "n": n},
        "shapiro": {"w": float(sw.statistic), "p": float(sw.pvalue), "n": n} if sw is not None else
                   {"w": None, "p": None, "n": n, "na": "n/a (N > 5000)"},
        "anderson": anderson(x) | {"n": n},
        "ttest": {"t": float(tt.statistic), "p": float(tt.pvalue), "df": n - 1},
    }
    return out


# ----------------------------------------------------------------------------- Walk-Forward

def wf_windows(n: int, is_len: int, oos_len: int, mode: str) -> list[tuple[int, int, int, int]]:
    """Liste (is_start, is_end, oos_start, oos_end), Enden exklusiv."""
    out, k = [], 0
    while True:
        e = is_len + k * oos_len
        if e >= n:
            break
        s = k * oos_len if mode == "rolling" else 0
        out.append((s, e, e, min(e + oos_len, n)))
        k += 1
    return out


def kupiec(x: int, t: int, alpha: float) -> dict:
    if t <= 0:
        return {"lr": None, "p": None}

    def xlogy(a, b):
        return 0.0 if a == 0 else a * math.log(b)
    lr = -2 * (xlogy(t - x, 1 - alpha) + xlogy(x, alpha)) + 2 * (xlogy(t - x, 1 - x / t) + xlogy(x, x / t))
    lr = max(lr, 0.0)
    return {"lr": lr, "p": float(st.chi2.sf(lr, 1))}


def _thin(n: int, max_pts: int) -> np.ndarray:
    if n <= max_pts:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_pts).round().astype(int))


def walk_forward(r: pd.Series, is_len: int, oos_len: int, mode: str, rtype: str, k: int) -> dict | None:
    x = r.to_numpy(dtype=float)
    d = r.index
    wins = wf_windows(x.size, is_len, oos_len, mode)
    if not wins:
        return None
    steps = []
    agg = {m: {str(1 - a): {"x": 0, "t": 0} for a in ALPHAS} for m in ("hist", "param")}
    viol_marks = {str(1 - a): [] for a in ALPHAS}
    oos_r, oos_d = [], []
    for s, e, os_, oe in wins:
        ins, oos = x[s:e], x[os_:oe]
        mu, sd = float(ins.mean()), float(ins.std(ddof=1))
        row = {"is_start": str(d[s].date()), "is_end": str(d[e - 1].date()), "oos_start": str(d[os_].date()),
               "oos_end": str(d[oe - 1].date()), "is_idx": [s, e], "oos_idx": [os_, oe], "partial": oe - os_ < oos_len,
               "is_mu": mu, "is_sigma": sd, "oos_n": int(oos.size), "oos_mu": float(oos.mean()),
               "oos_sigma": float(oos.std(ddof=1)) if oos.size > 1 else None, "by": {}}
        for a in ALPHAS:
            c = str(1 - a)
            hv, he = var_es_hist(ins, a)
            pv, pe = var_es_param(mu, sd, a)
            res = {}
            for m, v, es in (("hist", hv, he), ("param", pv, pe)):
                vi = oos[oos < -v]
                res[m] = {"var": v, "es": es, "viol": int(vi.size), "hit": vi.size / oos.size,
                          "oos_es": -float(vi.mean()) if vi.size else None}
                agg[m][c]["x"] += int(vi.size)
                agg[m][c]["t"] += int(oos.size)
                if m == "hist":
                    viol_marks[c] += [[str(d[os_ + i].date()), float(oos[i])] for i in np.where(oos < -v)[0]]
            row["by"][c] = res
        steps.append(row)
        oos_r.append(oos)
        oos_d += list(d[os_:oe])
    total = {}
    for m in agg:
        total[m] = {}
        for a in ALPHAS:
            c = str(1 - a)
            xx, tt = agg[m][c]["x"], agg[m][c]["t"]
            total[m][c] = {"viol": xx, "t": tt, "expected": a * tt, "hit": xx / tt if tt else None} | kupiec(xx, tt, a)
    rr = np.concatenate(oos_r)
    eq = np.exp(np.cumsum(rr)) if rtype == "log" else np.cumprod(1 + rr)
    rv = pd.Series(rr).rolling(oos_len, min_periods=oos_len).std(ddof=1).to_numpy() * math.sqrt(k)
    idx = _thin(rr.size, 400)
    mus = np.array([s_["is_mu"] for s_ in steps])
    sds = np.array([s_["is_sigma"] for s_ in steps])
    return {"is": is_len, "oos": oos_len, "mode": mode, "steps": steps, "total": total,
            "stability": {"mu_mean": float(mus.mean()), "mu_range": float(mus.max() - mus.min()),
                          "sigma_mean": float(sds.mean()), "sigma_range": float(sds.max() - sds.min())},
            "chart": {"d": [str(oos_d[i].date()) for i in idx], "equity": eq[idx].tolist(),
                      "roll_vol": [None if not np.isfinite(v) else float(v) for v in rv[idx]],
                      "viol": viol_marks}}


# ----------------------------------------------------------------------------- Varianten je Instrument

def combo(per_px: pd.Series, live: bool, interval: str, rtype: str, years: int | None, crypto: bool) -> dict | None:
    r_all = to_returns(per_px, rtype)
    if live:
        hist, r_live = r_all.iloc[:-1], r_all.iloc[-1:]
    else:
        hist, r_live = r_all, None
    if years is not None and len(hist):
        hist = hist[hist.index > hist.index[-1] - pd.DateOffset(years=years)]
    if len(hist) < MIN_N:
        return None
    k = ann_factor(interval, crypto)
    out = describe(hist, k)
    mu, sd = out["distribution"]["mean"], out["distribution"]["sigma"]
    x = hist.to_numpy(dtype=float)
    out["hist"] = {b: histogram(x, b, mu, sd) for b in BINS}
    out["qq"] = qq_points(x)
    if r_live is not None:
        lr = live_rank(x, float(r_live.iloc[0]), mu, sd, hist.index[0], hist.index[-1])
        lr |= {"live": True, "date": str(r_live.index[0].date())}
    else:  # letzte abgeschlossene Periode, fuer den Vergleich aus der Historie genommen
        h2 = x[:-1]
        m2, s2 = float(h2.mean()), float(h2.std(ddof=1))
        lr = live_rank(h2, float(x[-1]), m2, s2, hist.index[0], hist.index[-2])
        lr |= {"live": False, "date": str(hist.index[-1].date())}
    out["live_rank"] = lr
    if years is None:  # Walk-Forward nur ueber die volle Historie (SPEC 4)
        is_len, oos_len = WF_DEFAULT[interval]
        out["wf"] = {m: walk_forward(hist, is_len, oos_len, m, rtype, k) for m in ("rolling", "expanding")}
    return out


def quality(px: pd.Series, crypto: bool) -> list:
    r = px.pct_change().iloc[1:]
    lim = 0.40 if crypto else 0.25
    bad = r[r.abs() > lim]
    return [[str(d.date()), float(v)] for d, v in bad.items()][:50]


def instrument_payload(cid: str, name: str, group: str, source: str, note: str, px: pd.Series,
                       today: pd.Timestamp) -> dict:
    crypto = is_crypto(source)
    daily_live = bool(px.index[-1] >= today.normalize())
    closed = px.iloc[:-1] if daily_live else px  # Strategie-Tester nur mit abgeschlossenen Tagen
    pe = {}
    for iv in ("1wk", "1mo"):
        per = resample_last(closed, iv)
        _, lv = split_live(per, iv, today)
        ends = per.iloc[:-1] if lv else per
        pos = closed.index.get_indexer(ends.index)
        pe[iv] = pos.tolist()
    combos, live = {}, {}
    for iv in INTERVALS:
        per, lv = split_live(resample_last(px, iv), iv, today)
        live[iv] = {"is_live": lv, "date": str(per.index[-1].date()), "price": float(per.iloc[-1])}
        for rt in TYPES:
            for pk, yrs in PERIODS.items():
                c = combo(per, lv, iv, rt, yrs, crypto)
                if c is not None:
                    combos[f"{iv}|{rt}|{pk}"] = c
    return {"meta": {"id": cid, "name": name, "group": group, "ticker": source.split(":")[1], "source": source,
                     "note": note, "crypto": crypto, "investable": cid != "VIX",
                     "start": str(px.index[0].date()), "end": str(px.index[-1].date()),
                     "data_as_of": str(px.index[-1].date()), "daily_live": daily_live, "quality": quality(px, crypto)},
            "series": {"d0": str(closed.index[0].date()), "d": (closed.index - closed.index[0]).days.tolist(),
                       "c": [float(f"{v:.6g}") for v in closed.values]},
            "period_ends": pe, "live": live, "combos": combos}


def macro_overlay(refresh: bool) -> dict:
    out = {}
    for sid, label, yoy in (("DGS10", "US-Rendite 10 J. (%)", False), ("VIXCLS", "VIX", False),
                            ("CPIAUCSL", "US-Inflation (CPI, % ggü. Vorjahr)", True)):
        try:
            s = ds.fred(sid, refresh)
            m = s.groupby(s.index.to_period("M")).last()
            if yoy:
                m = (m / m.shift(12) - 1) * 100
            m = m.dropna()
            out[sid] = {"label": label, "d": [str(p.to_timestamp(how="end").date()) for p in m.index],
                        "v": [round(float(v), 4) for v in m.values]}
        except Exception as e:  # Overlay ist Zusatz: Fehler -> "nicht verfuegbar"
            out[sid] = {"label": label, "error": type(e).__name__}
    return out


def is_stale(last: pd.Timestamp, today: pd.Timestamp, source: str, crypto: bool) -> bool:
    """Letzter Kurs zu alt? Faellt ein Abruf aus, wird still der Cache genutzt -> das muss auffallen.
    Erlaubt sind Wochenende und Feiertage: Boerse bis 3 Werktage dazwischen, Waehrungen (Fed + EZB) bis 4,
    Krypto (7 Tage Handel) bis 2 Kalendertage."""
    last, today = pd.Timestamp(last).normalize(), pd.Timestamp(today).normalize()
    if crypto:
        return (today - last).days > 2
    gap = int(np.busday_count((last + pd.Timedelta(days=1)).date(), today.date()))
    return gap > (4 if source.startswith("fx:") else 3)


def main(refresh: bool = True) -> int:
    today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
    (OUT / "stats").mkdir(parents=True, exist_ok=True)
    try:
        ecb = ds.ecb_fx_daily()
    except Exception:
        ecb = None
    index, failed, stale = [], [], []
    for cid, name, group, source, note in INSTRUMENTS:
        try:
            px = load_instrument(source, cid, refresh, ecb)
            if len(px) < 300:
                raise ValueError("zu wenig Daten")
            pay = instrument_payload(cid, name, group, source, note, px, today)
            (OUT / "stats" / f"{cid}.json").write_text(ds.to_json(round_sig(pay)), encoding="utf-8")
            m = pay["meta"]
            old = is_stale(px.index[-1], today, source, m["crypto"])
            if old:
                stale.append(cid)
            index.append({k: m[k] for k in ("id", "name", "group", "ticker", "note", "crypto", "investable",
                                             "start", "end")} | {"n_quality": len(m["quality"]), "stale": old})
            print(f"{'VERALTET' if old else 'OK':8s} {cid:9s} {m['start']} – {m['end']}  {len(pay['combos'])} Varianten")
        except Exception as e:
            failed.append(cid)
            print(f"FEHLER   {cid:9s} {type(e).__name__}: {e}")
        if source.startswith("yf:") and refresh:
            time.sleep(0.5)  # Yahoo bremst bei schnellen Abfragen (429)
    payload = {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
               "today": str(today.date()), "instruments": index, "failed": failed, "stale": stale,
               "macro": macro_overlay(refresh), "defaults": {"wf": WF_DEFAULT}}
    (OUT / "stats_index.json").write_text(ds.to_json(payload), encoding="utf-8")
    print(f"{len(index)} Instrumente, {len(failed)} Fehler" + (f": {', '.join(failed)}" if failed else "")
          + f", {len(stale)} mit veralteten Kursen" + (f": {', '.join(stale)}" if stale else ""))
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main(refresh="--cache" not in sys.argv))

