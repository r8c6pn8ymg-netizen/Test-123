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


def turnover(w: pd.DataFrame) -> pd.Series:
    """Umschichtung je Monat inkl. Einstieg (erster Monat und nach Pausen zaehlt die volle Position)."""
    w0 = w.fillna(0.0)
    t = w0.diff().abs().sum(axis=1)
    t.iloc[0] = w0.iloc[0].abs().sum()
    return t


def ls_weights(score: pd.DataFrame, n_ls: int = N_LS) -> pd.DataFrame:
    """n_ls Plaetze long (hoechste Werte) und n_ls short (niedrigste), je 1/n_ls.
    Gleichstand: Eine Gruppe gleicher Werte belegt gemeinsam die Raenge first..last und bekommt den Anteil
    dieser Raenge, der in die Long- bzw. Short-Plaetze faellt, gleichmaessig verteilt. Reicht eine Gruppe ueber
    beide Grenzen, werden Long- und Short-Anteil verrechnet. Unabhaengig von der Spaltenreihenfolge."""
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for t, row in score.iterrows():
        row = row.dropna()
        if len(row) < 2 * n_ls:
            w.loc[t] = np.nan
            continue
        for sign, v in ((1, row), (-1, -row)):
            first = v.rank(ascending=False, method="min")        # 1 = groesster Wert
            size = v.groupby(v).transform("count")               # Groesse der Gleichstands-Gruppe
            share = (np.minimum(first + size - 1, n_ls) - first + 1).clip(lower=0) / size
            w.loc[t, v.index] += sign * share / n_ls
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


def complete_months(spot: pd.DataFrame, px: dict) -> pd.DataFrame:
    """Kurse nur fuer abgeschlossene Monate: Der laufende Monat (Kurse erst bis z. B. zum 13.) ist kein
    voller Monat und darf nicht als Monatsrendite in Backtest und Statistik zaehlen. Die Live-Rangliste
    nutzt weiter die aktuellsten Kurse."""
    last = max(s.dropna().index[-1] for c, s in px.items() if c in spot.columns)
    out = spot.copy()
    if last < spot.index[-1] - pd.Timedelta(days=3):  # Monatsende noch nicht erreicht
        out.loc[spot.index[-1], [c for c in out.columns if c != "USD"]] = np.nan
    return out


def core(px: dict[str, pd.Series], refresh: bool = False):
    """Gemeinsame Bausteine (auch fuer den Konsens-Test): Kurse, Zinsen, Faktor-Scores, Ueberschussrenditen."""
    spot = pd.DataFrame({c: px[c].resample("ME").last() for c in CUR if c != "USD"})
    spot["USD"] = 1.0
    spot = spot[CUR].dropna(how="all")
    rate = pd.DataFrame({c: ds.bis("WS_CBPOL", f"M.{BIS[c]}", refresh).resample("ME").last() for c in CUR}).ffill()
    rate = rate.reindex(spot.index, method="ffill")
    reer = pd.DataFrame({c: ds.bis("WS_EER", f"M.R.N.{BIS[c]}", refresh).resample("ME").last() for c in CUR})
    gap = (100 * (reer / reer.rolling(120, min_periods=60).mean() - 1)).shift(2).reindex(spot.index, method="ffill")
    carry = rate.sub(rate["USD"], axis=0)
    mom = 0.5 * (spot / spot.shift(3) - 1) + 0.5 * (spot / spot.shift(12) - 1)
    tradable = spot.notna()  # EUR erst ab 1999 handelbar
    z = {"carry": zrank(carry.where(tradable)), "value": zrank((-gap).where(tradable)), "momentum": zrank(mom.where(tradable))}
    z["kombi"] = (z["carry"] + z["value"] + z["momentum"]) / 3
    sp = complete_months(spot, px)
    excess = (sp.shift(-1) / sp - 1) + carry / 1200
    excess["USD"] = 0.0
    excess.loc[excess.drop(columns="USD").isna().all(axis=1), "USD"] = np.nan  # kein Phantom-Monat am Ende
    return dict(spot=spot, rate=rate, carry=carry, z=z, excess=excess)


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
    tradable = spot.notna()
    z = {"carry": zrank(carry.where(tradable)), "value": zrank(value.where(tradable)), "momentum": zrank(mom.where(tradable))}
    z["kombi"] = (z["carry"] + z["value"] + z["momentum"]) / 3

    # Ueberschussrendite Monat t -> t+1 gegen USD inkl. Zinsdifferenz
    sp = complete_months(spot, px)
    excess = (sp.shift(-1) / sp - 1) + carry / 1200
    excess["USD"] = 0.0
    excess.loc[excess.drop(columns="USD").isna().all(axis=1), "USD"] = np.nan  # kein Phantom-Monat am Ende
    start = pd.Timestamp("1976-12-31")
    res, weights = {}, {}

    def run(sc):
        w = ls_weights(sc[sc.index >= start])
        gross = (w * excess.reindex(w.index)).sum(axis=1, min_count=1).dropna()
        turn = turnover(w).reindex(gross.index).fillna(0.0)
        return w, dict(gross=gross, net=gross - turn * COST)

    for k, sc in z.items():
        weights[k], res[k] = run(sc)
    # Walk-Forward: Faktor-Gewichte aus der Sharpe-Ratio der letzten 60 Monate (bis Vormonat)
    fr = pd.DataFrame({k: res[k]["net"] for k in ("carry", "value", "momentum")})
    roll = (fr.rolling(60, min_periods=60).mean() / fr.rolling(60, min_periods=60).std()).clip(lower=0)
    roll = roll.shift(1)  # Rendite t->t+1 ist auf t datiert -> nur bis t-1 bekannt
    have = roll.notna().all(axis=1)
    wf = roll.div(roll.sum(axis=1), axis=0)
    wf[have & (roll.sum(axis=1) == 0)] = 1 / 3  # alle Faktoren zuletzt negativ -> gleich gewichten
    wf = wf[have].dropna()
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
    # Leitzinsen fuer die Live-Rangliste tagesaktuell (BIS Tageswerte); Backtest bleibt bei Monatsende-Werten
    live_rate = rate.iloc[-1].copy()
    # Rueckfall-Datum = Monat des letzten echten BIS-Monatswerts (nicht des aufgefuellten Monatsendes)
    rate_date = {c: str(ds.bis("WS_CBPOL", f"M.{BIS[c]}", False).index[-1].date())[:7] for c in CUR}
    for c in CUR:
        try:
            dser = ds.bis_daily_recent("WS_CBPOL", f"D.{BIS[c]}", refresh=refresh)
            if len(dser):
                live_rate[c] = float(dser.iloc[-1]); rate_date[c] = str(dser.index[-1].date())
        except Exception as e:  # Rueckfall: Monatswert
            print(f"  ! Tages-Leitzins {c}: {str(e)[:60]}")
    live_carry = live_rate - live_rate["USD"]
    live_value = value.iloc[-1]

    def pct_hist(hist, v):
        h = hist[hist.index >= "1977-01-01"].dropna()
        return round(float((h < v).mean() * 100)) if len(h) > 24 and pd.notna(v) else None
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
                            rate=round(float(live_rate[c]), 2), rate_date=rate_date[c],
                            # Einordnung gegen die eigene Historie seit 1977 (Perzentil, 100 = staerkster Wert je)
                            p_carry=None if c == "USD" else pct_hist(carry[c], live_carry[c]),
                            p_value=pct_hist(value[c], live_value[c]),
                            p_mom=None if c == "USD" else pct_hist(mom[c], live_mom[c])))
    ranking.sort(key=lambda r: -r["kombi"])
    longs = [r["cur"] for r in ranking[:N_LS]]
    shorts = [r["cur"] for r in ranking[-N_LS:]]
    pairs = sorted(((a, b, next(r["kombi"] for r in ranking if r["cur"] == a) - next(r["kombi"] for r in ranking if r["cur"] == b))
                    for a in longs for b in shorts), key=lambda x: -x[2])

    # Rang-Historie der letzten 24 Monate (Kombi)
    zk = z["kombi"].dropna(how="all").iloc[-24:]
    hist_rank = dict(d=[str(d.date())[:7] for d in zk.index],
                     ranks={c: [int(v) for v in zk.rank(axis=1, ascending=False)[c].fillna(0).values] for c in CUR})

    # ---- Robustheit: andere (ebenfalls uebliche) Einstellungen, Kosten, Ausreisser, einzelne Waehrungen
    def variant(n_ls=N_LS, m_s=3, m_l=12, vy=10, cost=COST, drop=None):
        cols = [c for c in CUR if c != drop]
        g_ = (100 * (reer[cols] / reer[cols].rolling(vy * 12, min_periods=vy * 6).mean() - 1)).shift(2).reindex(spot.index, method="ffill")
        m_ = 0.5 * (spot[cols] / spot[cols].shift(m_s) - 1) + 0.5 * (spot[cols] / spot[cols].shift(m_l) - 1)
        tr_ = spot[cols].notna()
        sc = (zrank(carry[cols].where(tr_)) + zrank((-g_).where(tr_)) + zrank(m_.where(tr_))) / 3
        sc = sc[sc.index >= start]
        w = ls_weights(sc, n_ls)  # dieselbe Gleichstands-Regel wie im Haupt-Backtest
        gr = (w * excess[cols].reindex(w.index)).sum(axis=1, min_count=1).dropna()
        return gr - turnover(w).reindex(gr.index).fillna(0.0) * cost

    def shp(r):
        r = r.dropna()
        return round(float(r.mean() / r.std() * np.sqrt(12)), 2) if len(r) > 24 and r.std() > 0 else None

    grid = []
    for n_ls in (2, 3):  # 4 je Seite braucht 8 Waehrungen -> erst ab 1999 moeglich, darum nicht im Vergleich
        for ms, ml in ((1, 12), (3, 12), (6, 12), (3, 6)):
            for vy in (5, 10):
                r_ = variant(n_ls, ms, ml, vy)
                grid.append(dict(n=n_ls, mom=f"{ms}+{ml}M", val=f"{vy}J", full=shp(r_), post=shp(r_[r_.index >= "2000"]),
                                 std=(n_ls, ms, ml, vy) == (N_LS, 3, 12, 10)))
    fulls = [g["full"] for g in grid]
    posts = [g["post"] for g in grid]
    robust = dict(grid=grid, full_min=min(fulls), full_med=float(np.median(fulls)), full_max=max(fulls),
                  post_min=min(posts), post_med=float(np.median(posts)), post_max=max(posts),
                  std_rank=1 + sum(f > next(g["full"] for g in grid if g["std"]) for f in fulls), n=len(grid))
    robust["costs"] = [dict(bp=bp, full=shp(r_), post=shp(r_[r_.index >= "2000"]))
                       for bp, r_ in ((bp, variant(cost=bp / 1e4)) for bp in (0, 5, 10, 20, 40))]
    base = res["kombi"]["net"]
    robust["turnover"] = round(float(turnover(weights["kombi"]).mean() * 12), 1)
    robust["no_top12"] = shp(base.drop(base.nlargest(12).index))
    robust["loo"] = {c: shp(variant(drop=c)) for c in CUR}

    # ---- Alle 7 Faktor-Auswahlen (fuer die Auswahl im Cockpit) -------------------------
    # C/V/M = Carry/Value/Momentum. Nur CVM (Standard) und die Einzelfaktoren sind vorab festgelegt;
    # Zweier-Kombinationen sind eine nachtraegliche Auswahl des Nutzers.
    combos = {}
    zmap = {"C": z["carry"], "V": z["value"], "M": z["momentum"]}
    for key in ("CVM", "C", "V", "M", "CV", "CM", "VM"):
        sc = sum(zmap[ch] for ch in key) / len(key)
        _, rr = run(sc)
        n_ = pd.Series(rr["net"].values, index=rr["net"].index + pd.offsets.MonthEnd(1))
        pf = perf(n_) or {}
        per = {lab: (perf(n_[(n_.index >= a) & (n_.index < b)]) or {}).get("sharpe")
               for lab, a, b in (("1977–1999", "1977-01-01", "2000-01-01"), ("2000–2014", "2000-01-01", "2015-01-01"),
                                 ("2015–heute", "2015-01-01", "2100-01-01"))}
        eqc = (1 + n_.reindex(base_idx).fillna(0)).cumprod()
        combos[key] = dict(sharpe=pf.get("sharpe"), p=pf.get("p"), ret=pf.get("ret"), dd=pf.get("dd"), periods=per,
                           equity=[round(float(v), 4) for v in eqc.values])

    corr = pd.DataFrame({k: res[k]["net"] for k in ("carry", "value", "momentum")}).corr().round(2)
    out = dict(generated=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"), asof=str(last.date()),
               live_fx=str(min(px[c].dropna().index[-1] for c in CUR if c != "USD").date()),
               ranking=ranking, longs=longs, shorts=shorts,
               pairs=[dict(long=a, short=b, diff=round(float(d), 2)) for a, b, d in pairs],
               stats=stats, periods=periods, perm=perm, equity=eq, hist_rank=hist_rank,
               rolling=rolling, factor_weights=factor_weights, robust=robust, combos=combos,
               corr={k: corr[k].to_dict() for k in corr.columns}, cost_bp=COST * 1e4, n_ls=N_LS)
    (OUT / "factors.json").write_text(ds.to_json(out))
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
