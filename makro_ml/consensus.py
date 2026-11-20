"""Konsens-Test: Makro ML + Faktor-Rangliste + CFTC kombiniert.

Alle Regeln sind VOR dem Test festgelegt (keine Optimierung):

Signale je Waehrung c gegen USD, jeweils am Monatsende, nur mit damals bekannten Daten:
  F (Faktor)  : sign(Kombi-Score c - Kombi-Score USD)                         [ab 1977]
  M (Makro)   : sign(Makro-Score c), wenn |Score| >= 0,1, sonst 0              [ab 2008]
                Makro-Score = alle echten Kandidaten-Modelle (keine Top-50-Auswahl), je Basis
                eine Stimme, OHNE Basen, die schon in F oder C stecken (Zinsdifferenz, Momentum,
                Bewertung, CFTC) -> keine doppelte Information.
  C (CFTC)    : Gegensignal bei Extremen: Spekulanten-Perzentil (3 Jahre) > 90 -> -1, < 10 -> +1
                (Stichtag Dienstag, nutzbar ab Freitag)                       [ab 1989]

Varianten: F, M, C einzeln; Mehrheit = sign(Summe); Konsens = nur wenn alle Signale ungleich 0
in dieselbe Richtung zeigen (mind. 2).
Portfolio: gleichgewichtet ueber alle Waehrungen mit Position, Halten 1 Monat, Ertrag inkl.
Zinsdifferenz, Kosten 5 Bp je Umschichtung (Sensitivitaet 20 Bp).

Zwei Sichtweisen:
  1) Gegen USD : jede Waehrung einzeln long/short gegen den Dollar.
  2) Relativ   : alle 8 Waehrungen (inkl. USD) werden je Signal in eine Rangliste uebersetzt;
                 gekauft werden die 3 staerksten, verkauft die 3 schwaechsten. Damit entstehen
                 automatisch Kreuzpaare (z. B. AUD/JPY), der USD ist nur eine von 8 Waehrungen.
                 Makro- und CFTC-Signal des USD = 0 (Referenz), Faktor = eigener Score.
                 Kombi = Mittel der Rang-Scores; Konsens = nur Waehrungen, die in ALLEN Signalen
                 in der oberen (long) bzw. unteren Haelfte (short) liegen.

Walk-Forward: Jedes Jahr wird die Variante mit der besten Sharpe-Ratio im In-Sample-Fenster
(5 J., 10 J., wachsend) gewaehlt und im Folgejahr out-of-sample gehandelt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import data_sources as ds
import factors as Fm

OUT = Path(__file__).parent / "output"
CURS = ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]
COST = 0.0005


def cot_signal(code: str, index: pd.DatetimeIndex) -> pd.Series:
    c = ds.cftc_net(code, False)["net_pct"].dropna()
    pct = c.rolling(156, min_periods=104).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    sig = pd.Series(0.0, index=pct.index)
    sig[pct > 90] = -1.0
    sig[pct < 10] = 1.0
    sig[pct.isna()] = np.nan
    sig.index = sig.index + pd.Timedelta(days=3)  # Veroeffentlichung freitags
    return sig.reindex(index, method="ffill")


def macro_signal(path: Path, asset: str, index: pd.DatetimeIndex) -> pd.Series:
    d = json.loads(path.read_text()).get(asset)
    if not d:
        return pd.Series(np.nan, index=index)
    s = pd.Series(d["v"], index=pd.to_datetime(d["d"])).sort_index()
    s = s.reindex(index, method="ffill", tolerance=pd.Timedelta(days=10))
    out = np.sign(s).where(s.abs() >= 0.1, 0.0)
    out[s.isna()] = np.nan
    return out


def portfolio(pos: pd.DataFrame, excess: pd.DataFrame, cost: float = COST) -> pd.Series:
    n = pos.abs().sum(axis=1)
    w = pos.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    gross = (w * excess.reindex(w.index)[w.columns]).sum(axis=1)
    turn = Fm.turnover(w)
    net = gross - turn * cost
    exw = excess.reindex(w.index)[[c for c in w.columns if c != "USD"]]
    net = net[exw.notna().any(axis=1)]   # USD allein (immer 0) zaehlt nicht als vorhandener Monat
    return pd.Series(net.values, index=net.index + pd.offsets.MonthEnd(1))  # Ertrag t->t+1 auf t+1


def nw(r: pd.Series, lags: int = 3):
    x = r.values - r.values.mean()
    T = len(x)
    lrv = x @ x / T
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (x[L:] @ x[:-L]) / T
    t = float(r.mean() / np.sqrt(lrv / T)) if lrv > 0 else np.nan
    return t, float(1 - norm.cdf(t))


def perf(r: pd.Series, pos: pd.DataFrame | None = None) -> dict | None:
    r = r.dropna()
    if len(r) < 24 or r.std() == 0:
        return None
    ann, vol = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    t, p = nw(r)
    out = dict(ret=round(float(ann) * 100, 2), vol=round(float(vol) * 100, 2), sharpe=round(float(ann / vol), 2),
               dd=round(float((eq / eq.cummax() - 1).min()) * 100, 1), t=round(t, 2), p=float(f"{p:.3g}"),
               months=len(r), active=round(float((r != 0).mean()) * 100))
    return out


def sign_permutation(pos: pd.DataFrame, excess: pd.DataFrame, n: int = 2000, seed: int = 11,
                     cost: float = COST) -> np.ndarray:
    """Zufalls-Richtungen bei gleicher Positionsgroesse: prueft, ob die RICHTUNG Information enthaelt.
    Gleich gerechnet wie portfolio(): Kosten je Umschichtung, dieselben Monate (ohne reine USD-Monate)."""
    rng = np.random.default_rng(seed)
    act = pos.abs().values  # Positionsgroesse bleibt (auch Teilpositionen bei Gleichstand), nur die Richtung wird zufaellig
    exdf = excess.reindex(pos.index)[pos.columns]
    ex = exdf.fillna(0).values
    keep = exdf[[c for c in pos.columns if c != "USD"]].notna().any(axis=1).values
    cnt = act.sum(axis=1)
    out = np.empty(n)
    for i in range(n):
        s = rng.choice([-1.0, 1.0], size=act.shape) * act
        w = np.divide(s, cnt[:, None], out=np.zeros_like(s), where=cnt[:, None] > 0)
        turn = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(axis=1)
        r = ((w * ex).sum(axis=1) - turn * cost)[keep]
        out[i] = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    return out


def walk_forward(rets: dict[str, pd.Series], is_years: int | None, oos_start: str) -> pd.Series:
    """Jedes Jahr: Variante mit bester In-Sample-Sharpe waehlen, im Folgejahr handeln."""
    df = pd.DataFrame(rets).dropna(how="all")
    out, picks = [], {}
    for y in range(pd.Timestamp(oos_start).year, df.index[-1].year + 1):
        a = pd.Timestamp(f"{y}-01-01")
        is_ = df[(df.index < a) & ((df.index >= a - pd.DateOffset(years=is_years)) if is_years else True)]
        if len(is_) < 36:
            continue
        sr = is_.mean() / is_.std()
        best = sr.idxmax()
        picks[y] = best
        out.append(df.loc[(df.index >= a) & (df.index < pd.Timestamp(f"{y + 1}-01-01")), best])
    return (pd.concat(out) if out else pd.Series(dtype=float)), picks


def run_universe(name: str, sig: dict[str, pd.DataFrame], excess: pd.DataFrame, start: str, oos_start: str):
    keys = list(sig)
    S = pd.concat(sig, axis=1)
    idx = S.index[S.index >= start]
    sig = {k: v.reindex(idx) for k, v in sig.items()}
    avail = [sig[k] for k in keys]
    total = sum(s.fillna(0) for s in avail)
    nonzero = sum((s.fillna(0) != 0).astype(int) for s in avail)
    agree = sum((s.fillna(0) == np.sign(total)).astype(int) * (s.fillna(0) != 0) for s in avail)
    variants = {k: sig[k].fillna(0) for k in keys}
    variants["Mehrheit"] = np.sign(total)
    variants["Konsens"] = np.sign(total).where((nonzero >= 2) & (agree == nonzero), 0.0)
    rets, stats, perms, rets20 = {}, {}, {}, {}
    for k, pos in variants.items():
        r = portfolio(pos, excess)
        rets[k] = r
        rets20[k] = portfolio(pos, excess, 0.002)
        stats[k] = perf(r)
        if stats[k]:
            stats[k]["sharpe_20bp"] = (perf(rets20[k]) or {}).get("sharpe")
            stats[k]["positions"] = round(float(pos.abs().sum(axis=1).mean()), 2)  # Teilpositionen anteilig
            ex_ = excess.reindex(pos.index)[pos.columns]
            hit = (np.sign(ex_) == np.sign(pos))[(pos != 0) & (ex_ != 0)].stack().mean()  # Teilpositionen (Gleichstand) zaehlen nach Richtung
            stats[k]["hit"] = round(float(hit) * 100, 1)
            sims = sign_permutation(pos, excess)
            perms[k] = float(f"{(sims >= stats[k]['sharpe']).mean():.3g}")
            stats[k]["perm_p"] = perms[k]
    # Holm-Korrektur ueber alle Varianten (Mehrfachtests)
    ps = sorted(((s["p"], k) for k, s in stats.items() if s), key=lambda x: x[0])
    m = len(ps)
    for i, (p, k) in enumerate(ps):
        stats[k]["p_holm"] = float(f"{min(1.0, max(p * (m - j) for j, (p, _) in enumerate(ps[:i + 1]))):.3g}")
    # 5-Jahres-Bloecke
    blocks = {}
    last_y = max(r.index.max().year for r in rets.values() if len(r))
    yrs = list(range(pd.Timestamp(start).year, last_y + 1, 5))
    for k, r in rets.items():
        blocks[k] = {}
        for y0 in yrs:
            seg = r[(r.index >= f"{y0}-01-01") & (r.index < f"{y0 + 5}-01-01")]
            pf = perf(seg)
            blocks[k][f"{y0}–{min(y0 + 4, last_y)}"] = pf["sharpe"] if pf else None
    # Walk-Forward (Variante nach In-Sample-Sharpe waehlen)
    wf = {}
    for label, yrs_ in (("IS 5 Jahre", 5), ("IS 10 Jahre", 10), ("IS wachsend", None)):
        r, picks = walk_forward(rets, yrs_, oos_start)
        r = r[r.index >= oos_start]
        pf = perf(r)
        if pf:
            pf["picks"] = {str(y): k for y, k in picks.items()}
            wf[label] = pf
    static_oos = {k: (perf(r[r.index >= oos_start]) or {}).get("sharpe") for k, r in rets.items()}
    eq_idx = rets[keys[0]].index
    equity = {k: [round(float(v), 4) for v in (1 + r.reindex(eq_idx).fillna(0)).cumprod().values] for k, r in rets.items()}
    equity["d"] = [str(d.date()) for d in eq_idx]
    return dict(name=name, start=start, oos_start=oos_start, stats=stats, blocks=blocks, wf=wf,
                static_oos=static_oos, equity=equity), variants


def run_relative(name: str, scores: dict[str, pd.DataFrame], excess: pd.DataFrame, start: str, oos_start: str):
    """Relative Sicht: Rangliste aller 8 Waehrungen je Signal, 3 long / 3 short."""
    keys = list(scores)
    zs = {k: Fm.zrank(v[v.index >= start]) for k, v in scores.items()}
    idx = zs[keys[0]].index
    zs = {k: v.reindex(idx) for k, v in zs.items()}
    combo = sum(zs[k] for k in keys) / len(keys)
    raw = {k: v[v.index >= start].reindex(idx) for k, v in scores.items()}

    def halves(df):
        """+1 = strikt ueber dem Median der Waehrungen, -1 = strikt darunter, 0 = Median/Gleichstand."""
        med = df.median(axis=1)
        out = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        out[df.gt(med, axis=0)] = 1.0
        out[df.lt(med, axis=0)] = -1.0
        out[df.isna()] = np.nan
        return out

    variants = {}
    for k in keys:
        discrete = raw[k].stack().nunique() <= 3  # CFTC: nur -1/0/+1 -> keine kuenstliche 3/3-Auswahl
        variants[k] = halves(raw[k]) if discrete else Fm.ls_weights(zs[k]) * Fm.N_LS
    variants["Kombi"] = Fm.ls_weights(combo) * Fm.N_LS
    H = [halves(raw[k]) for k in keys]
    allpos = pd.concat([(h == 1).astype(float) for h in H]).groupby(level=0).min()
    allneg = pd.concat([(h == -1).astype(float) for h in H]).groupby(level=0).min()
    variants["Konsens"] = (allpos - allneg).reindex(idx)
    rets, stats = {}, {}
    for k, pos in variants.items():
        pos = pos.fillna(0)
        r = portfolio(pos, excess)
        rets[k] = r
        st = perf(r)
        if st:
            st["sharpe_20bp"] = (perf(portfolio(pos, excess, 0.002)) or {}).get("sharpe")
            st["positions"] = round(float(pos.abs().sum(axis=1).mean()), 2)  # Teilpositionen anteilig
            ex_ = excess.reindex(pos.index)[pos.columns]
            hit = (np.sign(ex_) == np.sign(pos))[(pos != 0) & (ex_ != 0)].stack().mean()  # Teilpositionen (Gleichstand) zaehlen nach Richtung
            st["hit"] = round(float(hit) * 100, 1)
            sims = sign_permutation(pos, excess)
            st["perm_p"] = float(f"{(sims >= st['sharpe']).mean():.3g}")
        stats[k] = st
    ps = sorted(((s_["p"], k) for k, s_ in stats.items() if s_), key=lambda x: x[0])
    m = len(ps)
    for i, (p, k) in enumerate(ps):
        stats[k]["p_holm"] = float(f"{min(1.0, max(pp * (m - j) for j, (pp, _) in enumerate(ps[:i + 1]))):.3g}")
    blocks = {}
    for k, r in rets.items():
        blocks[k] = {}
        last_y = r.index.max().year if len(r) else pd.Timestamp.today().year
        for y0 in range(pd.Timestamp(start).year, last_y + 1, 5):
            pf = perf(r[(r.index >= f"{y0}-01-01") & (r.index < f"{y0 + 5}-01-01")])
            blocks[k][f"{y0}–{min(y0 + 4, last_y)}"] = pf["sharpe"] if pf else None
    wf = {}
    for label, yrs_ in (("IS 5 Jahre", 5), ("IS 10 Jahre", 10), ("IS wachsend", None)):
        r, picks = walk_forward(rets, yrs_, oos_start)
        r = r[r.index >= oos_start]
        pf = perf(r)
        if pf:
            pf["picks"] = {str(y): k for y, k in picks.items()}
            wf[label] = pf
    static_oos = {k: (perf(r[r.index >= oos_start]) or {}).get("sharpe") for k, r in rets.items()}
    eq_idx = rets["Kombi"].index
    equity = {k: [round(float(v), 4) for v in (1 + r.reindex(eq_idx).fillna(0)).cumprod().values] for k, r in rets.items()}
    equity["d"] = [str(d.date()) for d in eq_idx]
    rank_now = {k: {c: round(float(v), 2) for c, v in zs[k].iloc[-1].items() if pd.notna(v)} for k in keys}
    rank_now["Kombi"] = {c: round(float(v), 2) for c, v in combo.iloc[-1].items() if pd.notna(v)}
    return dict(name=name, start=start, oos_start=oos_start, stats=stats, blocks=blocks, wf=wf,
                static_oos=static_oos, equity=equity, rank_now=rank_now)


def build(px: dict[str, pd.Series], mode: str = "alfred") -> dict:
    core = Fm.core(px)
    z, excess = core["z"]["kombi"], core["excess"]
    idx = z.index
    F = pd.DataFrame({c: np.sign(z[c] - z["USD"]) for c in CURS}, index=idx)
    import pipeline as P
    C = pd.DataFrame({c: cot_signal(P.COT_CODES[c], idx) for c in CURS}, index=idx)
    M = pd.DataFrame({c: macro_signal(OUT / f"macro_score_{mode}.json", c, idx) for c in CURS}, index=idx)

    # Relative Sicht: Scores fuer alle 8 Waehrungen (USD = Referenz 0 bei Makro und CFTC)
    ALL = ["USD"] + CURS
    zf = z[ALL]
    Mraw = pd.DataFrame({c: macro_signal(OUT / f"macro_score_{mode}.json", c, idx) for c in CURS}, index=idx)
    Mrel = Mraw.copy()
    for c in CURS:  # fuer die Rangliste den stetigen Score nutzen (nicht nur das Vorzeichen)
        d_ = json.loads((OUT / f"macro_score_{mode}.json").read_text()).get(c)
        if d_:
            Mrel[c] = pd.Series(d_["v"], index=pd.to_datetime(d_["d"])).sort_index().reindex(
                idx, method="ffill", tolerance=pd.Timedelta(days=10))
    Mrel["USD"] = 0.0
    Mrel.loc[Mrel[CURS].isna().all(axis=1), "USD"] = np.nan
    Crel = C.copy()
    Crel["USD"] = 0.0
    Crel.loc[Crel[CURS].isna().all(axis=1), "USD"] = np.nan
    rel_long = run_relative("Relativ: Faktor + CFTC (ab 1990)", {"Faktor": zf, "CFTC": Crel[ALL]}, excess,
                            "1990-01-31", "2000-01-01")
    rel_full = run_relative("Relativ: Faktor + Makro ML + CFTC (ab 2008)",
                            {"Faktor": zf, "Makro": Mrel[ALL], "CFTC": Crel[ALL]}, excess, "2008-01-31", "2013-01-01")

    long_u, _ = run_universe("Faktor + CFTC (ab 1990)", {"Faktor": F, "CFTC": C}, excess, "1990-01-31", "2000-01-01")
    full_u, var_full = run_universe("Faktor + Makro ML + CFTC (ab 2008)", {"Faktor": F, "Makro": M, "CFTC": C},
                                    excess, "2008-01-31", "2013-01-01")
    # Aktueller Stand je Waehrung (letztes Monatsende + aktuelle Signale)
    last = idx[-1]
    now = []
    for c in CURS:
        f_, m_, c_ = F[c].iloc[-1], M[c].dropna().iloc[-1] if M[c].notna().any() else np.nan, C[c].iloc[-1]
        sigs = [x for x in (f_, m_, c_) if pd.notna(x)]
        tot = np.sign(sum(sigs))
        nz = [x for x in sigs if x != 0]
        kons = tot if len(nz) >= 2 and all(x == tot for x in nz) else 0
        now.append(dict(cur=c, faktor=float(f_), makro=None if pd.isna(m_) else float(m_), cftc=float(c_),
                        mehrheit=float(tot), konsens=float(kons)))
    out = dict(generated=str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), asof=str(last.date()), mode=mode,
               long=long_u, full=full_u, rel_long=rel_long, rel_full=rel_full, now=now)
    (OUT / "consensus.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    import sys
    import pipeline as P
    mode = sys.argv[1] if len(sys.argv) > 1 else "alfred"
    o = build(P.load_fx(False), mode)
    for u in ("long", "full", "rel_long", "rel_full"):
        U = o[u]
        print(f"\n===== {U['name']} =====")
        print(f"{'Variante':10s} {'Sharpe':>6s} {'20Bp':>5s} {'NW p':>7s} {'Holm p':>7s} {'Zufall p':>8s} {'Treffer':>7s} {'aktiv%':>6s} {'Pos.':>5s} {'DD':>6s}")
        for k, s in U["stats"].items():
            if s:
                print(f"{k:10s} {s['sharpe']:6.2f} {s['sharpe_20bp']:5.2f} {s['p']:7.3g} {s['p_holm']:7.3g} {s['perm_p']:8.3g} {s['hit']:7.1f} {s['active']:6d} {s['positions']:5.2f} {s['dd']:6.1f}")
        print("5-Jahres-Bloecke:", json.dumps(U["blocks"], ensure_ascii=False))
        print(f"Walk-Forward OOS ab {U['oos_start']}: " + " | ".join(f"{k}: Sharpe {v['sharpe']} p {v['p']}" for k, v in U["wf"].items()))
        print("Statisch gleicher OOS-Zeitraum:", U["static_oos"])
        for k, v in U["wf"].items():
            print("  Auswahl", k, v["picks"])
    print("\nAktuell gegen USD:", o["now"])
    print("Aktuelle relative Rangliste (Kombi):", sorted(o["rel_full"]["rank_now"]["Kombi"].items(), key=lambda x: -x[1]))
