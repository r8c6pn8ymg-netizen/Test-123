"""Unabhaengiger Nachbau der restlichen Modell-Arten (Ergaenzung zu nachbau_stichprobe.py):
Profi-Faktoren (Momentum, REER, CFTC, Rohstoffe, VIX u. a.), alle Laender-Differenzen und die ALFRED-Variante.
Eigener Datenaufbau aus den Rohdaten, Bewertung mit demselben unabhaengigen Kern wie nachbau_stichprobe.py.
Aufruf aus makro_ml/:  python3 research/nachbau_rest.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import data_sources as ds  # nur Datenabruf  # noqa: E402
from nachbau_stichprobe import price  # Kurse (eigene Umsetzung aus dem ersten Nachbau)  # noqa: E402

BIS = {"DXY": "US", "EUR": "XM", "JPY": "JP", "GBP": "GB", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
OECD = {"EUR": "EZ", "JPY": "JP", "GBP": "GB", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
SIPRI = {"EUR": "Germany", "JPY": "Japan", "GBP": "United Kingdom", "CHF": "Switzerland", "CAD": "Canada",
         "AUD": "Australia", "NZD": "New Zealand", "US": "United States of America"}
COT = {"DXY": "098662", "EUR": "099741", "JPY": "097741", "GBP": "096742", "CHF": "092741", "CAD": "090741",
       "AUD": "232741", "NZD": "112741"}
US_SIMPLE = {"FEDFUNDS": ("FEDFUNDS", "lvl", 32), "DGS2": ("DGS2", "lvl", 3), "T10Y2Y": ("T10Y2Y", "lvl", 3),
             "INDPRO": ("INDPRO", "yoy", 55), "UMCSENT": ("UMCSENT", "lvl", 30), "RRSFS": ("RRSFS", "yoy", 40),
             "PPIACO": ("PPIACO", "yoy", 40), "BOPGSTB": ("BOPGSTB", "k", 70), "VIX": ("VIXCLS", "lvl", 3),
             "CREDIT": ("BAA10Y", "lvl", 3)}
COMMOD = {"OIL-WTI": ("DCOILWTICO", 3), "IRONORE": ("PIORECRUSDM", 45), "COPPER": ("PCOPPUSDM", 45),
          "FOOD-IDX": ("PFOODINDEXM", 45)}
ALFRED = {"PAYEMS": "diff3", "UNRATE": "level", "CPIAUCSL": "yoy", "CPILFESL": "yoy", "INDPRO": "yoy",
          "OPHNFB": "yoy_q", "BOPGSTB": "k", "PCEPILFE": "yoy", "RRSFS": "yoy", "PPIACO": "yoy"}


def mmean(s):
    s = s.dropna()
    g = s.groupby(s.index.to_period("M")).mean()
    g.index = g.index.to_timestamp()
    return g


def mlast(s):
    s = s.dropna()
    g = s.groupby(s.index.to_period("M")).last()
    g.index = g.index.to_timestamp()
    return g


def pct(s, n=12):
    return (100 * (s / s.shift(n) - 1)).dropna()


def q_to_m(s):  # Quartalswert auf die Monate des Quartals (ab Quartalsbeginn)
    s = s.dropna()
    return s.resample("MS").ffill()


def cpi_bis(cc):
    s = ds.bis("WS_LONG_CPI", f"M.{cc}.771").dropna()
    s.index = s.index.to_period("M").to_timestamp()
    recent = s[s.index >= s.index[-1] - pd.DateOffset(years=3)]
    if recent.nunique() <= 0.6 * len(recent):
        q = s.groupby(s.index.to_period("Q")).last()
        s = pd.Series(q.values, index=q.index.to_timestamp(how="start") + pd.DateOffset(months=2)).resample("MS").ffill()
    return s


def country(cc_bis, cc_oecd, sipri_name):
    out = {"POL": (mmean(ds.bis("WS_CBPOL", f"M.{cc_bis}")), 32), "CPI": (cpi_bis(cc_bis), 50)}
    if cc_oecd == "EZ":
        out["Y10"] = (mmean(ds.ecb("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")), 3)
        out["R3M"] = (ds.ecb("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA").dropna(), 32)
        out["UNR"] = (ds.ecb("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T").dropna(), 45)
    else:
        out["Y10"] = (ds.fred(f"IRLTLT01{cc_oecd}M156N").dropna(), 62)
        out["R3M"] = (ds.fred(f"IR3TIB01{cc_oecd}M156N").dropna(), 65)
        if cc_oecd in ("CH", "NZ"):
            out["UNR"] = (q_to_m(ds.fred(f"LRHUTTTT{cc_oecd}Q156S")), 210 if cc_oecd == "CH" else 150)
        elif cc_oecd == "US":
            out["UNR"] = (ds.fred("UNRATE").dropna(), 37)
        else:
            out["UNR"] = (ds.fred(f"LRHUTTTT{cc_oecd}M156S").dropna(), 125 if cc_oecd == "GB" else 65)
        x, m = ds.fred(f"XTEXVA01{cc_oecd}M667S").dropna(), ds.fred(f"XTIMVA01{cc_oecd}M667S").dropna()
        out["TRD"] = ((100 * (x - m) / (x + m)).dropna(), 95)
        out["EXP"] = (pct(x), 95)
        out["EQ"] = (pct(ds.fred(f"SPASTT01{cc_oecd}M661N").dropna()), 35)
    sp = ds.sipri_share_of_gdp()
    if sipri_name in sp.columns:
        a = sp[sipri_name].dropna() * 100
        a.index = pd.to_datetime([f"{y}-01-01" for y in a.index])
        out["MIL"] = (a.resample("MS").ffill(), 485)
    return out


def series_for(basis, asset):
    """Monatsreihe (Index = Beobachtungsmonat) und Frist in Tagen fuer eine Basis."""
    if basis in US_SIMPLE:
        sid, how, lag = US_SIMPLE[basis]
        m = mmean(ds.fred(sid))
        return (pct(m) if how == "yoy" else m / 1000 if how == "k" else m), lag
    if basis in COMMOD:
        sid, lag = COMMOD[basis]
        return pct(mmean(ds.fred(sid))), lag
    if basis in ("MOM-12M", "MOM-3M"):
        pm = mlast(price(asset))
        return pct(pm, 12 if basis == "MOM-12M" else 3), 1
    if basis == "REER-GAP":
        r = ds.bis("WS_EER", f"M.R.B.{BIS[asset]}").dropna()
        return (100 * (r / r.rolling(120, min_periods=60).mean() - 1)).dropna(), 45
    if basis in ("COT-NET", "COT-COMM"):
        c = ds.cftc_net(COT[asset])
        return mmean(c["net_pct" if basis == "COT-NET" else "comm_pct"]), 7
    cc = country(BIS[asset], OECD[asset], SIPRI[asset])
    us = country("US", "US", SIPRI["US"])
    key = {"CBPOL-Diff": "POL", "IR3M-Diff": "R3M", "Y10-Diff": "Y10", "UNR-Diff": "UNR", "EQUITY-Rel": "EQ",
           "MILEX-Diff": "MIL", "CPI-Diff": "CPI"}.get(basis)
    if key:
        (a, la), (b, lb) = cc[key], us[key]
        return (a - b).dropna(), max(la, lb)
    if basis in ("TRADE", "EXPORTS"):
        return cc["TRD" if basis == "TRADE" else "EXP"]
    if basis == "CURVE-Diff":
        s = (cc["Y10"][0] - cc["POL"][0]) - (us["Y10"][0] - us["POL"][0])
        return s.dropna(), max(cc["Y10"][1], cc["POL"][1], us["Y10"][1], us["POL"][1])
    if basis == "REALRATE-Diff":
        return ((cc["POL"][0] - cc["CPI"][0]) - (us["POL"][0] - us["CPI"][0])).dropna(), 50
    raise KeyError(basis)


def feats_from_monthly(m, lag):
    f = pd.DataFrame({"lvl": m, "d3": m - m.shift(3), "d12": m - m.shift(12)}).dropna()
    f["avail"] = f.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    return f.sort_values("avail").drop_duplicates("avail", keep="last")


def feats_alfred(basis, today):
    """Je Datenstand (Monatsende seit 1998 + heute) die Reihe, wie sie damals bekannt war."""
    vint = list(pd.date_range("1998-01-31", today, freq="ME"))
    if vint[-1] != today:
        vint.append(today)
    V = ds.alfred_vintages(basis, vint, False)
    how, rows = ALFRED[basis], []
    for v in V.columns:
        s = V[v].dropna()
        if len(s) < 30:
            continue
        s.index = pd.to_datetime(s.index)
        if how == "yoy_q":
            x = pct(q_to_m(s))
        elif how == "yoy":
            x = pct(mmean(s))
        elif how == "diff3":
            x = mmean(s).diff(3).dropna()
        elif how == "k":
            x = mmean(s) / 1000
        else:
            x = mmean(s)
        if len(x) < 15:
            continue
        t = x.index[-1]
        if (t - pd.DateOffset(months=3)) not in x.index or (t - pd.DateOffset(months=12)) not in x.index:
            continue
        lv = float(x.iloc[-1])
        rows.append(dict(avail=pd.Timestamp(v), lvl=lv, d3=lv - float(x[t - pd.DateOffset(months=3)]),
                         d12=lv - float(x[t - pd.DateOffset(months=12)])))
    return pd.DataFrame(rows).sort_values("avail")


def evaluate(f, asset, h, today):
    px = price(asset).sort_index()
    weeks = px.resample("W-FRI").last().dropna().index
    weeks = weeks[weeks >= "1999-01-08"]
    if weeks[-1] < today:
        weeks = weeks.append(pd.DatetimeIndex([today]))
    X = pd.merge_asof(pd.DataFrame({"date": weeks}), f, left_on="date", right_on="avail", direction="backward")
    pxdf = pd.DataFrame({"date": px.index.values, "p": px.values})
    now = pd.merge_asof(pd.DataFrame({"date": weeks}), pxdf, on="date", direction="backward")["p"].values
    fut_dates = weeks + pd.Timedelta(days=h)
    fut = pd.merge_asof(pd.DataFrame({"date": fut_dates}), pxdf, on="date", direction="backward")["p"].values
    up = np.where(fut_dates <= px.index[-1], (fut / now - 1) > 0, np.nan)
    F = X[["lvl", "d3", "d12"]].values
    ok = ~np.isnan(F).any(axis=1) & ~np.isnan(now)
    start = max(pd.Timestamp("2008-01-04"), weeks[ok][0] + pd.Timedelta(weeks=260))
    refits = pd.date_range(start, weeks[-1] + pd.Timedelta(days=1), freq="26W-FRI")
    pred = np.full(len(weeks), np.nan)
    for i, R in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else weeks[-1] + pd.Timedelta(days=7)
        tr = ok & (fut_dates <= R) & ~np.isnan(up)
        te = ok & (weeks >= R) & (weeks < nxt)
        if tr.sum() < 130 or not te.any():
            continue
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        pred[te] = LogisticRegression(C=0.5, max_iter=1000).fit((F[tr] - mu) / sd, up[tr].astype(int)).predict_proba((F[te] - mu) / sd)[:, 1]
    ev = ~np.isnan(pred) & ~np.isnan(up)
    tr = ok & ~np.isnan(up)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    last = np.where(ok)[0][-1]
    p_now = LogisticRegression(C=0.5, max_iter=1000).fit((F[tr] - mu) / sd, up[tr].astype(int)).predict_proba(((F[last] - mu) / sd)[None, :])[0, 1]
    return round(float(((pred[ev] > 0.5) == (up[ev] == 1)).mean()) * 100, 1), round(float(p_now), 3)


def main():
    today = pd.Timestamp(pd.Timestamp.today().date())
    rng = np.random.default_rng(11)
    jobs = []
    Df = json.load(open(HERE.parent / "output" / "data_fred.json"))["models"]
    Da = json.load(open(HERE.parent / "output" / "data_alfred.json"))["models"]
    first_sample = set(["CPILFESL", "PAYEMS", "ICSA", "PCEPILFE", "UNRATE", "CREDIT", "CPIAUCSL", "CPI-Diff"])
    for name, pool, alfred in (("Profi + übrige Länder (FRED)", [m for m in Df if m["basis"] not in first_sample and m["basis"] not in ALFRED], False),
                               ("ALFRED", [m for m in Da if m["basis"] in ALFRED], True)):
        by = {}
        for m in pool:
            by.setdefault(m["basis"], []).append(m)
        for _b, lst in sorted(by.items()):
            for i in rng.choice(len(lst), size=min(2, len(lst)), replace=False):
                jobs.append((name, lst[i], alfred))
    print(f"{'Bereich':28s} {'Modell':30s} {'DA Cockpit':>10s} {'Nachbau':>8s} {'Roh C':>7s} {'Nachbau':>8s}  Ergebnis")
    bad = 0
    cache = {}
    for name, m, alfred in jobs:
        try:
            key = (m["basis"], m["asset"], alfred)
            if key not in cache:
                cache[key] = feats_alfred(m["basis"], today) if alfred else feats_from_monthly(*series_for(m["basis"], m["asset"]))
            da, pr = evaluate(cache[key], m["asset"], m["h"], today)
            ok = abs(da - m["da"]) <= 0.2 and abs(pr - m["prob_raw"]) <= 0.01
        except Exception as e:
            da, pr, ok = float("nan"), float("nan"), False
            print("   Fehler im Nachbau:", m["basis"], m["asset"], type(e).__name__, str(e)[:80])
        bad += not ok
        print(f"{name:28s} {m['asset']+' · '+m['basis']+' · '+str(m['h'])+'d':30s} {m['da']:9.1f}% {da:7.1f}% {m['prob_raw']*100:6.1f}% {pr*100:7.1f}%  {'OK' if ok else 'ABWEICHUNG'}")
    print(f"\nGESAMT: {len(jobs) - bad} von {len(jobs)} stimmen überein (DA ±0,2 Punkte, Roh-Ausgabe ±1 Punkt)")


if __name__ == "__main__":
    main()
