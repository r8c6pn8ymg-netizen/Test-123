"""Test nach Vorab-Registrierung (VORAB_REGISTRIERUNG_trend_carry.md):
T = Trendfolge (Time-Series-Momentum), V = Carry mit Volatilitaets-Bremse. Aufruf aus makro_ml/:
    python3 research/trend_volcarry.py
Schreibt output/research_trend_carry.json und gibt die Entscheidung nach der festen Regel aus."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import factors as F  # noqa: E402
import pipeline as P  # noqa: E402
from data_sources import to_json  # noqa: E402

START = pd.Timestamp("1976-12-31")
HOLD = pd.Timestamp("2022-01-01")
CUR7 = [c for c in F.CUR if c != "USD"]
rng = np.random.default_rng(20261120)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(12)) if len(r) > 11 and r.std() > 0 else float("nan")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    r = r[r.index >= START]
    t, p = F.nw_test(r)
    return dict(sharpe=round(sharpe(r), 2), pre2000=round(sharpe(r[r.index < "2000-01-01"]), 2),
                post2000=round(sharpe(r[r.index >= "2000-01-01"]), 2), holdout=round(sharpe(r[r.index >= HOLD]), 2),
                t=round(t, 2), p=float(f"{p:.3g}"), months=len(r),
                ret=round(r.mean() * 1200, 1), vol=round(r.std() * np.sqrt(12) * 100, 1),
                dd=round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) * 100, 1))


def net_of(w: pd.DataFrame, ex: pd.DataFrame, cost: float = F.COST) -> pd.Series:
    w = w[w.index >= START]
    gross = (w * ex.reindex(w.index)[w.columns]).sum(axis=1, min_count=1).dropna()
    net = gross - F.turnover(w).reindex(gross.index).fillna(0.0) * cost
    # Ertrag t -> t+1 auf den Monat datieren, in dem er anfaellt (wie im Faktor-Portfolio), damit die
    # Zeitraum-Grenzen (vor 2000, ab 2022) stimmen
    return pd.Series(net.values, index=net.index + pd.offsets.MonthEnd(1))


def load():
    px = P.load_fx(False)
    C = F.core(px, False)
    return px, C


# ---------------- Strategie T: Trendfolge ----------------
def ewma_vol(px: dict) -> pd.DataFrame:
    """Annualisierte EWMA-Schwankung der taeglichen Kursaenderungen (Schwerpunkt 60 Tage), am Monatsende."""
    out = {}
    for c in CUR7:
        r = np.log(px[c].dropna()).diff().dropna()
        out[c] = (r.ewm(com=60, min_periods=60).std() * np.sqrt(261)).resample("ME").last()
    return pd.DataFrame(out)


def trend_weights(C: dict, vol: pd.DataFrame, shift_signal: int = 0) -> pd.DataFrame:
    ex = C["excess"][CUR7]
    # Ueberschussertrag der letzten 12 Monate bis Monatsende t: Zeilen t-12..t-1 (Zeile s = Ertrag s -> s+1)
    past12 = (1 + ex.shift(1)).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1
    sig = np.sign(past12).shift(shift_signal)          # shift_signal nur fuer den Timing-Test
    tradable = C["spot"][CUR7].notna()
    v = vol.reindex(ex.index).where(tradable)
    raw = (sig * 0.10 / v).where(tradable)
    n = raw.notna().sum(axis=1).replace(0, np.nan)
    w = raw.div(n, axis=0)
    w["USD"] = np.where(n.notna(), 0.0, np.nan)  # Monate ohne jedes Signal zaehlen nicht als 0 %-Monat
    return w[F.CUR]


# ---------------- Strategie V: Carry mit Volatilitaets-Bremse ----------------
def carry_daily_rv(px: dict, wC: pd.DataFrame) -> pd.Series:
    """Realisierte Varianz der taeglichen Ertraege des Carry-Portfolios je Kalendermonat.
    Im Monat m gilt das Gewicht vom Monatsende davor."""
    dr = pd.DataFrame({c: px[c].dropna().pct_change() for c in CUR7}).dropna(how="all").fillna(0.0)
    me = dr.index + pd.offsets.MonthEnd(0)
    w_prev = wC[CUR7].shift(1).reindex(me).fillna(0.0)
    w_prev.index = dr.index
    port = (dr * w_prev).sum(axis=1)
    return (port ** 2).groupby(port.index + pd.offsets.MonthEnd(0)).sum()


def main() -> dict:
    px, C = load()
    ex = C["excess"]
    out: dict = {}

    # ---- T ----
    vol = ewma_vol(px)
    wT = trend_weights(C, vol)
    T = net_of(wT, ex)
    gT = (wT[wT.index >= START] * ex.reindex(wT[wT.index >= START].index)).sum(axis=1, min_count=1).dropna()
    out["T"] = stats(T)
    out["T"]["costs"] = {f"{b} Bp": round(sharpe(net_of(wT, ex, b / 1e4)[lambda s: s.index >= START]), 2) for b in (5, 10, 20)}
    # Zufallsportfolios: zufaelliges Vorzeichen, gleiche Positionsgroessen (brutto vs. brutto)
    W = wT[wT.index >= START][CUR7].reindex(gT.index)
    X = ex[CUR7].reindex(gT.index)
    absW, Xv = W.abs().values, X.values
    sims = np.empty(2000)
    for i in range(2000):
        sgn = rng.choice([-1.0, 1.0], size=absW.shape)
        r = np.nansum(sgn * absW * Xv, axis=1)
        sims[i] = r.mean() / r.std() * np.sqrt(12)
    out["T"]["random_p"] = round(float((sims >= sharpe(gT)).mean()), 4)
    out["T"]["random_p95"] = round(float(np.quantile(sims, 0.95)), 2)
    # Selbstpruefung Timing + Placebo
    out["T"]["check_future"] = round(sharpe(net_of(trend_weights(C, vol, shift_signal=-1), ex)), 2)
    out["T"]["check_lag1"] = round(sharpe(net_of(trend_weights(C, vol, shift_signal=1), ex)), 2)
    perm = wT.copy()
    idx = perm.index[perm.index >= START]
    perm.loc[idx, CUR7] = perm.loc[idx, CUR7].values[rng.permutation(len(idx))]
    out["T"]["check_placebo"] = round(sharpe(net_of(perm, ex)), 2)

    # ---- Carry (Basis) und V ----
    wC = F.ls_weights(C["z"]["carry"][C["z"]["carry"].index >= START])
    Cn = net_of(wC, ex)
    out["Carry"] = stats(Cn)
    rv = carry_daily_rv(px, wC)
    s = (1.0 / rv).reindex(wC.index)                    # Bremse fuer Zeile t (Ertrag t -> t+1) aus Monat t
    s = s.where(rv.reindex(wC.index) > 0)
    wV = wC.mul(s, axis=0)
    V1 = net_of(wV, ex)
    common = V1.index.intersection(Cn.index)
    c = Cn.loc[common].std() / V1.loc[common].std()    # Konstante: gleiche Gesamt-Schwankung (Sharpe unveraendert)
    V = V1 * c
    out["V"] = stats(V)
    s_paid = pd.Series(s.values, index=s.index + pd.offsets.MonthEnd(1))  # gleiche Datierung wie die Ertraege
    out["V"]["max_leverage"] = round(float((s_paid * c).reindex(common).max()), 1)
    out["V"]["costs"] = {f"{b} Bp": round(sharpe(net_of(wV, ex, b / 1e4)), 2) for b in (5, 10, 20)}
    import statsmodels.api as sm
    y, xC = V.loc[common], Cn.loc[common]
    m = sm.OLS(y.values, sm.add_constant(xC.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    out["V"]["alpha_pa"] = round(float(m.params[0]) * 1200, 2)
    out["V"]["alpha_t"] = round(float(m.tvalues[0]), 2)
    out["V"]["alpha_p"] = float(f"{(1 - __import__('scipy').stats.norm.cdf(m.tvalues[0])):.3g}")
    out["V"]["beta"] = round(float(m.params[1]), 2)

    # ---- Entscheidung nach fester Regel ----
    a = 0.025
    passT = out["T"]["p"] < a and out["T"]["post2000"] > 0 and out["T"]["holdout"] > 0
    passV = (out["V"]["alpha_p"] < a and out["V"]["post2000"] > out["Carry"]["post2000"]
             and out["V"]["holdout"] >= out["Carry"]["holdout"])
    out["decision"] = dict(T=passT, V=passV)

    # ---- Kombination (nur mit bestandenen Strategien) ----
    K = net_of(F.ls_weights(C["z"]["kombi"][C["z"]["kombi"].index >= START]), ex)
    out["Kombi"] = stats(K)
    parts = {"Kombi": K}
    if passT:
        parts["T"] = T
    if passV:
        parts["V"] = V
    if len(parts) > 1:
        df = pd.DataFrame(parts).dropna()
        scale = 1 / df.rolling(60, min_periods=60).std().shift(1)  # gleiches Risiko, nur Vergangenheit
        blend = (df * scale).mean(axis=1, skipna=False).dropna()
        Kb = df["Kombi"].reindex(blend.index)
        out["Blend"] = dict(parts=list(parts), **stats(blend),
                            kombi_same_period=stats(Kb))
        out["decision"]["Blend"] = (sharpe(blend[blend.index < HOLD]) > sharpe(Kb[Kb.index < HOLD])
                                    and sharpe(blend[blend.index >= HOLD]) >= sharpe(Kb[Kb.index >= HOLD]))
    out["series"] = {"d": [str(d.date()) for d in T.index],
                     "T": [round(float(x), 6) for x in T.values]}
    (HERE.parent / "output" / "research_trend_carry.json").write_text(to_json(out))
    return out


if __name__ == "__main__":
    o = main()
    for k in ("T", "Carry", "V", "Kombi", "Blend"):
        if k in o:
            print(k, {kk: vv for kk, vv in o[k].items() if kk not in ("series",)})
    print("ENTSCHEIDUNG:", o["decision"])
