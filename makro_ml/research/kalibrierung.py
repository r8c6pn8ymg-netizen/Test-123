"""Kalibrierung der Makro-ML-Wahrscheinlichkeiten (P(steigt)): Wenn ein Modell X % sagt, stieg die Waehrung dann
in X von 100 Faellen? Vergleich: Cockpit-Einstellung (L2, lam=2) gegen dieselben Modelle ohne Regularisierung
(lam ~ 0, Punkte breiter gestreut). Nur Out-of-Sample-Prognosen (Walk-Forward wie im Cockpit), FRED-Variante,
Makro-Basen (US + Laender-Differenzen). Die exakte Kalibrierung ALLER Cockpit-Kandidaten (je Ansicht) rechnet
pipeline.py taeglich (Feld "calibration" in data_*.json); dieses Skript dient nur dem Vergleich der Regularisierung.
Aufruf aus makro_ml/:  python3 research/kalibrierung.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pipeline as P  # noqa: E402

HS = (20, 60, 90, 180)


def fit_predict(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A1 = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    B1 = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    w = np.zeros(A1.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        g = A1.T @ (p - ytr) + lam * np.r_[0, w[1:]]
        H = (A1 * (p * (1 - p))[:, None]).T @ A1 + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H + 1e-8 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return 1 / (1 + np.exp(-np.clip(B1 @ w, -30, 30)))


def oos(X, price, h, lam):
    """Wie pipeline.run_model: Walk-Forward alle 26 Wochen, nur abgeschlossene Faelle im Training."""
    dates = X.index
    fwd = price.reindex(dates + pd.Timedelta(days=h), method="ffill").values
    now = price.reindex(dates, method="ffill").values
    ret = np.where((dates + pd.Timedelta(days=h)) <= price.index[-1], fwd / now - 1, np.nan)
    ok = X.notna().all(axis=1).values & ~np.isnan(now)
    if not ok.any():
        return None
    Xv, y = X.values, (ret > 0).astype(float)
    start = max(P.OOS_START, dates[ok][0] + pd.Timedelta(weeks=P.MIN_TRAIN))
    refits = pd.date_range(start, dates[-1] + pd.Timedelta(days=1), freq=f"{P.REFIT_WEEKS}W-FRI")
    if len(refits) < 4:  # dieselben Filter wie pipeline.run_model
        return None
    probs = np.full(len(dates), np.nan)
    for i, R in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else dates[-1] + pd.Timedelta(days=7)
        tr = ok & ((dates + pd.Timedelta(days=h)) <= R) & ~np.isnan(ret)
        te = ok & (dates >= R) & (dates < nxt)
        if tr.sum() < P.MIN_TRAIN // 2 or not te.any():
            continue
        probs[te] = P._fit_predict(Xv[tr], y[tr], Xv[te]) if lam == 2.0 else fit_predict(Xv[tr], y[tr], Xv[te], lam)
    ev = ~np.isnan(probs) & ~np.isnan(ret)
    if int(ev.sum()) < 150:
        return None
    return probs[ev], y[ev]


def main():
    today = pd.Timestamp(pd.Timestamp.today().date())
    px = P.load_fx(False)
    sipri = P.ds.sipri_share_of_gdp(False)
    us_cc = P.country_series("US", "US", "United States of America", sipri, False)
    ub = P.us_bases(False)
    tasks = []
    for asset in P.ASSET_ORDER:
        bases = list(ub)
        if asset != "DXY":
            cb, co, sn, _, _ = P.ASSETS[asset]
            bases += P.diff_bases(P.country_series(cb, co, sn, sipri, False), us_cc)
        price = px[asset].dropna()
        dates = pd.DatetimeIndex(price.resample("W-FRI").last().dropna().index)
        dates = dates[dates >= "1999-01-08"]
        for b in bases:
            X = P.features(b, dates)
            for h in HS:
                tasks.append((X, price, h))
    print(f"{len(tasks)} Modelle (echte Makro-Basen, Horizonte {HS})")
    out = {}
    for lam, name in ((2.0, "Cockpit (L2, lam=2)"), (1e-6, "ohne Regularisierung")):
        res = Parallel(n_jobs=-1, batch_size=16)(delayed(oos)(X, pr, h, lam) for X, pr, h in tasks)
        p = np.concatenate([r[0] for r in res if r is not None])
        y = np.concatenate([r[1] for r in res if r is not None])
        bins = [0, .3, .4, .45, .5, .55, .6, .7, 1.0001]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() > 200:
                rows.append((f"{lo*100:.0f}–{min(hi,1)*100:.0f} %", int(m.sum()), round(100 * p[m].mean(), 1),
                             round(100 * y[m].mean(), 1)))
        brier = float(np.mean((p - y) ** 2))
        base = float(np.mean((y.mean() - y) ** 2))
        da = float(((p > .5) == (y == 1)).mean())
        spread = float(np.mean(np.abs(p - .5) > .1))
        out[name] = dict(rows=rows, brier=brier, brier_base=base, da=da, spread=spread)
        print(f"\n== {name}: Prognosen weiter als 10 Punkte von 50 % entfernt: {spread*100:.1f} %  | "
              f"Trefferquote {da*100:.1f} %  | Brier {brier:.4f} (immer Durchschnitt sagen: {base:.4f}, kleiner = besser)")
        print(f"{'Modell sagt':>12s} {'Anzahl':>9s} {'Ø gesagt':>9s} {'tatsächlich gestiegen':>22s}")
        for r in rows:
            print(f"{r[0]:>12s} {r[1]:9d} {r[2]:8.1f}% {r[3]:21.1f}%")
    return out


if __name__ == "__main__":
    main()
