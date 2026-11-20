"""Test nach Vorab-Registrierung (VORAB_REGISTRIERUNG_52w.md): 52-Wochen-Hoch/-Tief bei Waehrungen.
Aufruf aus makro_ml/:  python3 research/hoch_tief_52w.py   -> output/research_52w.json"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pipeline as P  # noqa: E402
from data_sources import to_json  # noqa: E402

CUR7 = ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]
H = (5, 20, 65)
WIN, GAP, NEAR = 260, 20, 0.02
PERIODS = [("1971–1999", "1971", "1999"), ("2000–2014", "2000", "2014"), ("2015–heute", "2015", "2100")]
rng = np.random.default_rng(52)


def events(s: pd.Series, lookahead: bool = False) -> dict[str, pd.DatetimeIndex]:
    """Ereignistage. Hoch/Tief aus den 260 Handelstagen DAVOR (lookahead=True nur fuer die Timing-Kontrolle)."""
    if lookahead:  # falsch mit Absicht: Fenster reicht in die Zukunft
        hi = s.rolling(WIN + 1, center=True, min_periods=WIN).max()
        lo = s.rolling(WIN + 1, center=True, min_periods=WIN).min()
        raw = {"new_high": s >= hi, "new_low": s <= lo}
    else:
        hi = s.shift(1).rolling(WIN, min_periods=WIN).max()
        lo = s.shift(1).rolling(WIN, min_periods=WIN).min()
        raw = {"new_high": s > hi, "new_low": s < lo,
               "near_high": (s >= hi * (1 - NEAR)) & (s <= hi),
               "near_low": (s <= lo * (1 + NEAR)) & (s >= lo)}
    out = {}
    for k, m in raw.items():
        pos = np.flatnonzero(m.fillna(False).values)
        keep, last = [], -10**9
        for p in pos:  # Haeufung: naechstes Ereignis erst nach mindestens 20 Handelstagen
            if p - last >= GAP:
                keep.append(p)
                last = p
        out[k] = s.index[keep]
    return out


def table(px: dict, lookahead: bool = False) -> pd.DataFrame:
    """Eine Zeile je Ereignis: Waehrung, Art, Datum, Ueber-Bewegung je Horizont (in Ereignis-Richtung, in %)."""
    rows = []
    for c in CUR7 + ["DXY"]:
        s = px[c].dropna()
        fr = {h: s.shift(-h) / s - 1 for h in H}
        mu = {h: fr[h].mean() for h in H}
        upbase = {h: float((fr[h].dropna() > 0).mean()) for h in H}
        for kind, idx in events(s, lookahead).items():
            d = 1 if "high" in kind else -1
            for t in idx:
                r = dict(cur=c, kind=kind, date=t)
                for h in H:
                    v = fr[h].get(t)
                    r[f"x{h}"] = np.nan if pd.isna(v) else 100 * d * (v - mu[h])        # Ueber-Bewegung
                    r[f"hit{h}"] = np.nan if pd.isna(v) else float(d * v > 0)
                    r[f"base{h}"] = upbase[h] if d > 0 else 1 - upbase[h]
                rows.append(r)
    return pd.DataFrame(rows)


def boot_p(df: pd.DataFrame, col: str, n: int = 2000) -> tuple[float, float]:
    """Mittelwert und zweiseitiger p-Wert per Block-Bootstrap ueber Kalenderjahre (Ereignisse eines Jahres zusammen)."""
    x = df[["date", col]].dropna()
    if len(x) < 10:
        return np.nan, np.nan
    yrs = x["date"].dt.year.values
    uy = np.unique(yrs)
    sums = np.array([x[col].values[yrs == y].sum() for y in uy])
    cnts = np.array([(yrs == y).sum() for y in uy])
    idx = rng.integers(0, len(uy), size=(n, len(uy)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    m = x[col].mean()
    # zentrierte Verteilung (so, als gaebe es keinen Effekt): wie oft liegt sie so weit weg wie das beobachtete?
    centred = means - m
    p = float(((np.abs(centred) >= abs(m)).mean()))
    return float(m), p


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [None] * len(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def main() -> dict:
    px = P.load_fx(False)
    T = table(px)
    T7 = T[T["cur"].isin(CUR7)]
    res, keys, ps = {}, [], []
    for kind in ("new_high", "new_low", "near_high", "near_low"):
        for h in H:
            sub = T7[T7["kind"] == kind]
            m, p = boot_p(sub, f"x{h}")
            per = {lab: round(float(sub[(sub.date >= a) & (sub.date <= b + "-12-31")][f"x{h}"].mean()), 3)
                   for lab, a, b in PERIODS}
            by_cur = {c: round(float(sub[sub.cur == c][f"x{h}"].mean()), 3) for c in CUR7}
            # Je Waehrung einzeln (nur zur Anzeige; wenige Faelle, einzeln nicht signifikant), DXY eingeschlossen
            detail = {}
            for c in CUR7 + ["DXY"]:
                d_ = T[(T.cur == c) & (T.kind == kind)]
                n_ = int(d_[f"x{h}"].notna().sum())
                detail[c] = dict(n=n_, hit=round(float(d_[f"hit{h}"].mean()) * 100, 1) if n_ else None,
                                 base=round(float(d_[f"base{h}"].mean()) * 100, 1) if n_ else None,
                                 move=round(float(d_[f"x{h}"].mean()), 3) if n_ else None,
                                 since=str(d_["date"].min().year) if n_ else None)
            dx = T[(T.cur == "DXY") & (T.kind == kind)][f"x{h}"]
            res[f"{kind}_{h}"] = dict(n=int(sub[f"x{h}"].notna().sum()), mean=round(m, 3), p=round(p, 4),
                                      hit=round(float(sub[f"hit{h}"].mean()) * 100, 1),
                                      base=round(float(sub[f"base{h}"].mean()) * 100, 1),
                                      periods=per, by_cur=by_cur, dxy=round(float(dx.mean()), 3), detail=detail)
            keys.append(f"{kind}_{h}")
            ps.append(p)
    for k, a in zip(keys, holm(ps)):
        r = res[k]
        r["p_holm"] = round(a, 4)
        sg = np.sign(r["mean"])
        same_periods = all(np.sign(v) == sg for v in r["periods"].values() if not np.isnan(v))
        same_cur = sum(np.sign(v) == sg for v in r["by_cur"].values() if not np.isnan(v))
        r["pass"] = bool(a < 0.05 and same_periods and same_cur >= 5)
        r["same_cur"] = int(same_cur)

    # Placebo: gleiche Anzahl zufaelliger Tage je Waehrung
    pl = []
    for c in CUR7:
        s = px[c].dropna()
        n = int((T7.cur == c).sum() / 4)
        days = s.index[rng.choice(len(s) - 70, size=max(n, 1), replace=False)]
        fr = s.shift(-20) / s - 1
        pl.append(100 * (fr.reindex(days) - fr.mean()))
    placebo = float(pd.concat(pl).mean())
    # Timing-Kontrolle: Hoch/Tief inkl. Zukunft -> muss auffaellig "drehen" (Rueckwaertswissen)
    L = table(px, lookahead=True)
    look = {k: round(float(L[(L.cur.isin(CUR7)) & (L.kind == k)]["x20"].mean()), 3) for k in ("new_high", "new_low")}

    # Aktueller Stand je Waehrung: Abstand zum 52-Wochen-Hoch/-Tief
    now = {}
    for c in CUR7 + ["DXY"]:
        s = px[c].dropna()
        hi, lo, last = s.iloc[-WIN - 1:-1].max(), s.iloc[-WIN - 1:-1].min(), s.iloc[-1]
        now[c] = dict(date=str(s.index[-1].date()), to_high=round(100 * (last / hi - 1), 2),
                      to_low=round(100 * (last / lo - 1), 2))
    out = dict(results=res, placebo_x20=round(placebo, 3), lookahead_x20=look, now=now)
    (HERE.parent / "output" / "research_52w.json").write_text(to_json(out))
    return out


if __name__ == "__main__":
    o = main()
    print(f"{'Test':16s} {'n':>5s} {'Ø %':>7s} {'Treffer':>8s} {'normal':>7s} {'p':>7s} {'p Holm':>7s}  Zeiträume (71–99 / 00–14 / 15–)   Währungen gleich  Regel")
    for k, r in o["results"].items():
        per = " / ".join(f"{v:+.2f}" for v in r["periods"].values())
        print(f"{k:16s} {r['n']:5d} {r['mean']:+7.3f} {r['hit']:7.1f}% {r['base']:6.1f}% {r['p']:7.4f} {r['p_holm']:7.4f}  {per:32s}  {r['same_cur']}/7  {'JA' if r['pass'] else 'nein'}")
    print("Placebo (zufällige Tage, 20 Tage):", o["placebo_x20"], "| Timing-Kontrolle mit Zukunft:", o["lookahead_x20"])
    print("Heute:", {c: (v["to_high"], v["to_low"]) for c, v in o["now"].items()})
