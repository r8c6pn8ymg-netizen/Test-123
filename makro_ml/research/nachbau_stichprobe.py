"""Unabhaengiger Nachbau einer Stichprobe der angezeigten Makro-ML-Modelle (FRED-Variante).
Eigener Code: Rohdaten direkt, eigene Merkmale (zeitlich passende Verknuepfung mit merge_asof), eigenes Walk-Forward,
Modell aus scikit-learn (L2, C = 1/lam = 0,5). Verglichen werden Trefferquote (DA) und heutige Roh-Ausgabe mit dem Cockpit.
Aufruf aus makro_ml/:  python3 research/nachbau_stichprobe.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data_sources as ds  # nur Datenabruf (Rohdaten), keine Logik aus pipeline.py  # noqa: E402

FX = {"EUR": ("DEXUSEU", 1), "JPY": ("DEXJPUS", -1), "GBP": ("DEXUSUK", 1), "CHF": ("DEXSZUS", -1),
      "CAD": ("DEXCAUS", -1), "AUD": ("DEXUSAL", 1), "NZD": ("DEXUSNZ", 1)}
BIS_CC = {"EUR": "XM", "JPY": "JP", "GBP": "GB", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
# Basis -> (FRED-ID, Umformung, Frist in Tagen ab Monatsende)
US = {"CPILFESL": ("CPILFESL", "yoy", 45), "CPIAUCSL": ("CPIAUCSL", "yoy", 45), "PCEPILFE": ("PCEPILFE", "yoy", 62),
      "UNRATE": ("UNRATE", "lvl", 37), "PAYEMS": ("PAYEMS", "d3", 37), "ICSA": ("ICSA", "lvl", 7),
      "CREDIT": ("BAA10Y", "lvl", 3)}


def price(asset: str) -> pd.Series:
    if asset == "DXY":  # offizielle ICE-Formel aus den sechs Fed-Kursen
        k = {"DEXUSEU": -0.576, "DEXJPUS": 0.136, "DEXUSUK": -0.119, "DEXCAUS": 0.091, "DEXSDUS": 0.042, "DEXSZUS": 0.036}
        df = pd.concat({s: ds.fred(s) for s in k}, axis=1).dropna()
        out = pd.Series(50.14348112, index=df.index)
        for s, e in k.items():
            out = out * df[s] ** e
        return out
    sid, sign = FX[asset]
    s = ds.fred(sid).dropna()
    return s if sign > 0 else 1 / s


def monthly_basis(basis: str, asset: str) -> tuple[pd.Series, int]:
    if basis == "CPI-Diff":  # Inflations-Differenz Land minus USA (BIS, YoY), Quartalswerte erst ab Quartalsende
        def cpi(cc):
            s = ds.bis("WS_LONG_CPI", f"M.{cc}.771").dropna()
            s.index = s.index.to_period("M").to_timestamp()
            recent = s[s.index >= s.index[-1] - pd.DateOffset(years=3)]
            if recent.nunique() <= 0.6 * len(recent):          # drei gleiche Monatswerte je Quartal
                q = s.groupby(s.index.to_period("Q")).last()
                s = pd.Series(q.values, index=q.index.to_timestamp(how="start") + pd.DateOffset(months=2))
                s = s.resample("MS").ffill()
            return s
        return (cpi(BIS_CC[asset]) - cpi("US")).dropna(), 50
    sid, how, lag = US[basis]
    m = ds.fred(sid).dropna()
    m = m.groupby(m.index.to_period("M")).mean()
    m.index = m.index.to_timestamp()
    if how == "yoy":
        m = 100 * (m / m.shift(12) - 1)
    elif how == "d3":
        m = m.diff(3)
    return m.dropna(), lag


def rebuild(basis: str, asset: str, h: int, today: pd.Timestamp) -> dict:
    px = price(asset)
    weeks = px.resample("W-FRI").last().dropna().index
    weeks = weeks[weeks >= "1999-01-08"]
    if weeks[-1] < today:
        weeks = weeks.append(pd.DatetimeIndex([today]))
    m, lag = monthly_basis(basis, asset)
    f = pd.DataFrame({"lvl": m, "d3": m - m.shift(3), "d12": m - m.shift(12)}).dropna()
    f["avail"] = f.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)   # gilt erst ab Veroeffentlichung
    f = f.sort_values("avail").drop_duplicates("avail", keep="last")
    X = pd.merge_asof(pd.DataFrame({"date": weeks}), f, left_on="date", right_on="avail", direction="backward")
    pxd = px.sort_index()
    pxdf = pd.DataFrame({"date": pxd.index.values, "p": pxd.values})
    now = pd.merge_asof(pd.DataFrame({"date": weeks}), pxdf, on="date", direction="backward")["p"].values
    fut_dates = weeks + pd.Timedelta(days=h)
    fut = pd.merge_asof(pd.DataFrame({"date": fut_dates}), pxdf, on="date", direction="backward")["p"].values
    known = fut_dates <= pxd.index[-1]
    up = np.where(known, (fut / now - 1) > 0, np.nan)
    feats = X[["lvl", "d3", "d12"]].values
    okf = ~np.isnan(feats).any(axis=1) & ~np.isnan(now)
    first = weeks[okf][0]
    start = max(pd.Timestamp("2008-01-04"), first + pd.Timedelta(weeks=260))
    refits = pd.date_range(start, weeks[-1] + pd.Timedelta(days=1), freq="26W-FRI")
    pred = np.full(len(weeks), np.nan)
    for i, R in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else weeks[-1] + pd.Timedelta(days=7)
        tr = okf & (fut_dates <= R) & ~np.isnan(up)
        te = okf & (weeks >= R) & (weeks < nxt)
        if tr.sum() < 130 or not te.any():
            continue
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        clf = LogisticRegression(C=0.5, max_iter=1000).fit((feats[tr] - mu) / sd, up[tr].astype(int))
        pred[te] = clf.predict_proba((feats[te] - mu) / sd)[:, 1]
    ev = ~np.isnan(pred) & ~np.isnan(up)
    da = float(((pred[ev] > 0.5) == (up[ev] == 1)).mean())
    tr = okf & ~np.isnan(up)
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
    clf = LogisticRegression(C=0.5, max_iter=1000).fit((feats[tr] - mu) / sd, up[tr].astype(int))
    last = np.where(okf)[0][-1]
    p_now = float(clf.predict_proba(((feats[last] - mu) / sd)[None, :])[0, 1])
    return dict(da=round(da * 100, 1), prob_raw=round(p_now, 3), n=int(ev.sum()))


def main():
    D = json.load(open(HERE.parent / "output" / "data_fred.json"))
    today = pd.Timestamp(pd.Timestamp.today().date())
    want = set(US) | {"CPI-Diff"}
    pool = [m for m in D["models"] if m["basis"] in want]
    rng = np.random.default_rng(7)
    sample = []
    for a in ["DXY", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]:
        cand = [m for m in pool if m["asset"] == a and not (a == "DXY" and m["basis"] == "CPI-Diff")]
        for i in rng.choice(len(cand), size=min(3, len(cand)), replace=False):
            sample.append(cand[i])
    print(f"{'Modell':32s} {'DA Cockpit':>10s} {'DA Nachbau':>10s} {'Roh heute C':>11s} {'Nachbau':>8s}  Ergebnis")
    bad = 0
    for m in sample:
        r = rebuild(m["basis"], m["asset"], m["h"], today)
        dd, dp = abs(r["da"] - m["da"]), abs(r["prob_raw"] - m["prob_raw"])
        ok = dd <= 0.2 + 1e-9 and dp <= 0.01 + 1e-9  # Toleranz inklusive (Gleitkomma)
        bad += not ok
        print(f"{m['asset']+' · '+m['basis']+' · '+str(m['h'])+'d':32s} {m['da']:9.1f}% {r['da']:9.1f}% {m['prob_raw']*100:10.1f}% {r['prob_raw']*100:7.1f}%  {'OK' if ok else 'ABWEICHUNG'}")
    print(f"\nGESAMT: {len(sample) - bad} von {len(sample)} stimmen überein (DA ±0,2 Punkte, Roh-Ausgabe ±1 Punkt)")


if __name__ == "__main__":
    main()
