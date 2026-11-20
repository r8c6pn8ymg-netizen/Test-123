"""Faktor-Portfolio: Carry, Value, Momentum ueber 8 Waehrungen (inkl. USD).

Jeden Monatsende werden die 8 Waehrungen nach drei in der Forschung gut belegten
Faktoren sortiert. Das Portfolio kauft die 3 staerksten und verkauft die 3 schwaechsten
(je 1/3 Gewicht) und haelt bis zum naechsten Monatsende.

* Carry     : Leitzins des Landes minus US-Leitzins (BIS). Hoch = kaufen.
* Value     : realer effektiver Wechselkurs vs. 10-Jahres-Schnitt (BIS), 2 Monate verzoegert
              (Veroeffentlichung). Billig = kaufen.
* Momentum  : Kursentwicklung gegen USD ueber 3 und 12 Monate. Stark = kaufen.
* Kombi     : Mittel der drei Rang-Scores.
* Adaptiv   : Walk-Forward. Jeden Monat werden die Faktoren nach ihrer Sharpe-Ratio der
              letzten 60 Monate gewichtet (negative = 0), nur mit Daten bis zum Vormonat.

Historie ab 1977 (Wechselkurse ab 1971, reale effektive Wechselkurse der BIS ab 1964).
Der Euro ist erst ab 1999 enthalten (davor 7 Waehrungen).

Alle Parameter (3 long / 3 short, 3+12 Monate, 10 Jahre) stammen aus der Literatur und
wurden NICHT auf diese Daten optimiert, es gibt also keine Auswahl-Verzerrung.
Ertrag je Monat = Kursveraenderung gegen USD + Zinsdifferenz (Carry wird verdient).
Kosten: 5 Bp je umgeschichteter Einheit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import data_sources as ds

OUT = Path(__file__).parent / "output"
CUR = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]
BIS = {"USD": "US", "EUR": "XM", "JPY": "JP", "GBP": "GB", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
COST = 0.0005
N_LS = 3
PERIODS = [("1977–1989", "1977-01-01", "1990-01-01"), ("1990–1999", "1990-01-01", "2000-01-01"),
           ("2000–2007", "2000-01-01", "2008-01-01"), ("2008–2012", "2008-01-01", "2013-01-01"),
           ("2013–2017", "2013-01-01", "2018-01-01"), ("2018–2021", "2018-01-01", "2022-01-01"),
           ("2022–heute", "2022-01-01", "2100-01-01")]


def zrank(df: pd.DataFrame) -> pd.DataFrame:
    """Querschnitts-Rang je Monat, standardisiert (Mittel 0, Std 1)."""
    r = df.rank(axis=1)
    return r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1).replace(0, np.nan), axis=0)


def ls_weights(score: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for t, row in score.iterrows():
        row = row.dropna()
        if len(row) < 2 * N_LS:
            w.loc[t] = np.nan
            continue
        s = row.sort_values()
        w.loc[t, s.index[-N_LS:]] = 1 / N_LS
        w.loc[t, s.index[:N_LS]] = -1 / N_LS
    return w


def nw_test(r: pd.Series, lags: int = 3):
    x = r.values - r.values.mean()
    T = len(x)
    lrv = x @ x / T
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (x[L:] @ x[:-L]) / T
    t = float(r.mean() / np.sqrt(lrv / T)) if lrv > 0 else np.nan
    return t, float(1 - norm.cdf(t))


def perf(r: pd.Series) -> dict | None:
    r = r.dropna()
    if len(r) < 12:
        return None
    ann, vol = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    t, p = nw_test(r)
    return dict(ret=round(ann * 100, 1), vol=round(vol * 100, 1), sharpe=round(ann / vol, 2) if vol > 0 else None,
                dd=round(float((eq / eq.cummax() - 1).min()) * 100, 1), hit=round(float((r > 0).mean()) * 100, 1),
                t=round(t, 2), p=float(f"{p:.3g}"), months=len(r))


def build(px: dict[str, pd.Series], refresh: bool = False) -> dict:
    # Monatsend-Kurse (USD je Waehrung), USD = 1
    spot = pd.DataFrame({c: px[c].resample("ME").last() for c in CUR if c != "USD"})
    spot["USD"] = 1.0
    spot = spot[CUR].dropna(how="all")
    # Leitzinsen (BIS, Monatsende; Zinsentscheide sind sofort oeffentlich)
    rate = pd.DataFrame({c: ds.bis("WS_CBPOL", f"M.{BIS[c]}", refresh).resample("ME").last() for c in CUR}).ffill()
    rate = rate.reindex(spot.index, method="ffill")
    # Realer effektiver Wechselkurs: Abweichung vom 10-Jahres-Schnitt, 2 Monate Veroeffentlichungsverzug
    # "narrow"-Index der BIS: lange Historie ab 1964, durchgaengig dieselbe Methodik
    reer = pd.DataFrame({c: ds.bis("WS_EER", f"M.R.N.{BIS[c]}", refresh).resample("ME").last() for c in CUR})
    gap = 100 * (reer / reer.rolling(120, min_periods=60).mean() - 1)
    gap = gap.shift(2).reindex(spot.index, method="ffill")

    carry = rate.sub(rate["USD"], axis=0)
    mom = 0.5 * (spot / spot.shift(3) - 1) + 0.5 * (spot / spot.shift(12) - 1)
    value = -gap  # billig = hoch
    z = {"carry": zrank(carry), "value": zrank(value), "momentum": zrank(mom)}
    z["kombi"] = (z["carry"] + z["value"] + z["momentum"]) / 3

    # Ueberschussrendite Monat t -> t+1 gegen USD inkl. Zinsdifferenz
    excess = (spot.shift(-1) / spot - 1) + carry / 1200
    excess["USD"] = 0.0
    start = pd.Timestamp("1976-12-31")
    res, weights = {}, {}

    def run(sc):
        w = ls_weights(sc[sc.index >= start])
        gross = (w * excess.reindex(w.index)).sum(axis=1, min_count=1).dropna()
        turn = w.diff().abs().sum(axis=1).reindex(gross.index).fillna(1.0)
        return w, dict(gross=gross, net=gross - turn * COST)

    for k, sc in z.items():
        weights[k], res[k] = run(sc)
    # Walk-Forward: Faktor-Gewichte aus der Sharpe-Ratio der letzten 60 Monate (bis Vormonat)
    fr = pd.DataFrame({k: res[k]["net"] for k in ("carry", "value", "momentum")})
    roll = (fr.rolling(60, min_periods=60).mean() / fr.rolling(60, min_periods=60).std()).clip(lower=0)
    roll = roll.shift(1)  # Rendite t->t+1 ist auf t datiert -> nur bis t-1 bekannt
    wf = roll.div(roll.sum(axis=1), axis=0)
    wf[roll.sum(axis=1) == 0] = 1 / 3
    wf = wf.dropna()
    sc_ad = sum(z[k].reindex(wf.index).mul(wf[k], axis=0) for k in ("carry", "value", "momentum"))
    weights["adaptiv"], res["adaptiv"] = run(sc_ad)
    factor_weights = {k: [round(float(v), 2) for v in wf[k].resample("YE").mean().values] for k in wf.columns}
    factor_weights["y"] = [int(d.year) for d in wf.resample("YE").mean().index]
    # Rendite des Monats t->t+1 auf den Monat t+1 datieren
    for k in res:
        for g in ("gross", "net"):
            s = res[k][g]
            res[k][g] = pd.Series(s.values, index=s.index + pd.offsets.MonthEnd(1))

    # Permutationstest: jeden Monat zufaellig 3 der VERFUEGBAREN Waehrungen long, 3 short
    ex = excess.reindex(weights["kombi"].index)
    ex = ex[ex.notna().sum(axis=1) >= 2 * N_LS].iloc[:-1]
    rng = np.random.default_rng(7)
    NS = 2000
    sim_r = np.zeros((NS, len(ex)))
    for t, (_, row) in enumerate(ex.iterrows()):
        v = row.dropna().values
        keys = rng.random((NS, len(v)))
        o = np.argsort(keys, axis=1)
        sim_r[:, t] = v[o[:, -N_LS:]].mean(axis=1) - v[o[:, :N_LS]].mean(axis=1)
    sims = sim_r.mean(axis=1) / sim_r.std(axis=1) * np.sqrt(12)
    perm_idx = pd.Index(ex.index + pd.offsets.MonthEnd(1))

    stats, periods, perm, rolling = {}, {}, {}, {}
    for k in res:
        n_ = res[k]["net"]
        stats[k] = {g: perf(res[k][g]) for g in ("gross", "net")}
        stats[k]["holdout"] = perf(n_[n_.index >= "2022-01-01"])
        stats[k]["pre2000"] = perf(n_[n_.index < "2000-01-01"])
        stats[k]["post2000"] = perf(n_[n_.index >= "2000-01-01"])
        rs = (n_.rolling(60).mean() / n_.rolling(60).std() * np.sqrt(12)).dropna()
        rolling[k] = [round(float(v), 2) for v in rs.values]
        rolling.setdefault("d_" + k, [str(d.date()) for d in rs.index])
        periods[k] = {n: (perf(res[k]["net"][(res[k]["net"].index >= a) & (res[k]["net"].index < b)]) or {}).get("sharpe")
                      for n, a, b in PERIODS}
        g = res[k]["gross"].reindex(perm_idx)
        sr = g.mean() / g.std() * np.sqrt(12)
        perm[k] = dict(sharpe=round(float(sr), 2), p=float(f"{(sims >= sr).mean():.3g}"),
                       rand_p95=round(float(np.percentile(sims, 95)), 2), rand_mean=round(float(sims.mean()), 2))

    eq = {}
    base_idx = res["kombi"]["net"].index
    for k in res:
        e = (1 + res[k]["net"].reindex(base_idx).fillna(0)).cumprod()
        eq[k] = [round(float(v), 4) for v in e.values]
    eq["d"] = [str(d.date()) for d in base_idx]
    eq["kombi_gross"] = [round(float(v), 4) for v in (1 + res["kombi"]["gross"].reindex(base_idx).fillna(0)).cumprod().values]

    # Aktuelle Rangliste (letztes Monatsende mit vollstaendigen Daten) + live mit heutigen Kursen
    last = z["kombi"].dropna(how="all").index[-1]
    live_spot = pd.Series({c: (px[c].dropna().iloc[-1] if c != "USD" else 1.0) for c in CUR})
    live_mom = 0.5 * (live_spot / spot.iloc[-4] - 1) + 0.5 * (live_spot / spot.iloc[-13] - 1)
    live_carry = carry.iloc[-1]
    live_value = value.iloc[-1]
    zl = {"carry": zrank(live_carry.to_frame().T).iloc[0], "value": zrank(live_value.to_frame().T).iloc[0],
          "momentum": zrank(live_mom.to_frame().T).iloc[0]}
    zl["kombi"] = (zl["carry"] + zl["value"] + zl["momentum"]) / 3
    ranking = []
    for c in CUR:
        ranking.append(dict(cur=c, kombi=round(float(zl["kombi"][c]), 2), z_carry=round(float(zl["carry"][c]), 2),
                            z_value=round(float(zl["value"][c]), 2), z_mom=round(float(zl["momentum"][c]), 2),
                            carry=round(float(live_carry[c]), 2), value_gap=round(float(-live_value[c]), 1),
                            mom3=round(float(live_spot[c] / spot[c].iloc[-4] - 1) * 100, 1),
                            mom12=round(float(live_spot[c] / spot[c].iloc[-13] - 1) * 100, 1),
                            rate=round(float(rate[c].iloc[-1]), 2)))
    ranking.sort(key=lambda r: -r["kombi"])
    longs = [r["cur"] for r in ranking[:N_LS]]
    shorts = [r["cur"] for r in ranking[-N_LS:]]
    pairs = sorted(((a, b, next(r["kombi"] for r in ranking if r["cur"] == a) - next(r["kombi"] for r in ranking if r["cur"] == b))
                    for a in longs for b in shorts), key=lambda x: -x[2])

    # Rang-Historie der letzten 24 Monate (Kombi)
    zk = z["kombi"].dropna(how="all").iloc[-24:]
    hist_rank = dict(d=[str(d.date())[:7] for d in zk.index],
                     ranks={c: [int(v) for v in zk.rank(axis=1, ascending=False)[c].fillna(0).values] for c in CUR})

    corr = pd.DataFrame({k: res[k]["net"] for k in ("carry", "value", "momentum")}).corr().round(2)
    out = dict(generated=str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), asof=str(last.date()),
               live_fx=str(min(px[c].dropna().index[-1] for c in CUR if c != "USD").date()),
               ranking=ranking, longs=longs, shorts=shorts,
               pairs=[dict(long=a, short=b, diff=round(float(d), 2)) for a, b, d in pairs],
               stats=stats, periods=periods, perm=perm, equity=eq, hist_rank=hist_rank,
               rolling=rolling, factor_weights=factor_weights,
               corr={k: corr[k].to_dict() for k in corr.columns}, cost_bp=COST * 1e4, n_ls=N_LS)
    (OUT / "factors.json").write_text(json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    import sys
    import pipeline as P
    o = build(P.load_fx("--cache" not in sys.argv), "--cache" not in sys.argv)
    print("Rangliste:", [(r["cur"], r["kombi"]) for r in o["ranking"]])
    for k, v in o["stats"].items():
        print(k, "netto", v["net"], "| Schlusstest", v["holdout"], "| Perm", o["perm"][k])
    print("Zeiträume", o["periods"])
    print("Korrelation", o["corr"])
