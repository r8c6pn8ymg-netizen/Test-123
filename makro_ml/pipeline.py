"""Makro-ML-Pipeline: Daten -> Features -> Walk-Forward-Modelle -> JSON.

Ein *Modell* = (Basis-Indikator, Waehrung, Horizont in Kalendertagen).
Jedes Modell ist eine logistische Regression auf 3 Features der Basis
(Niveau, 3-Monats-Aenderung, 12-Monats-Aenderung) und sagt voraus, ob die
Waehrung gegen den USD (bzw. der Dollar-Index) in h Tagen hoeher steht.

Gegen Look-ahead-Bias:
* Makrodaten werden erst ab ihrem (konservativ geschaetzten)
  Veroeffentlichungsdatum genutzt (Spalte ``lag`` je Basis).
* Walk-Forward: alle 26 Wochen neu trainiert, nur mit Beispielen, deren
  Zielfenster (Datum + h) vor dem Trainingszeitpunkt bereits abgeschlossen war.
* Die Trefferquote (DA) ist rein out-of-sample (ab 2008).

Gegen "positive Varianz" (Glueckstreffer) muss ein Modell bestehen:
* DA >= 53 % gesamt und >= 51 % in BEIDEN Haelften des OOS-Zeitraums,
* mind. 2 Prozentpunkte besser als die naive Regel "immer die haeufigste
  Richtung der Trainingsdaten",
* p < 0.05 im Binomialtest mit effektiver Stichprobe (Ueberlappung der
  Zielfenster herausgerechnet).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest

import data_sources as ds

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

HORIZONS = list(range(10, 91, 5))
OOS_START = pd.Timestamp("2008-01-04")
REFIT_WEEKS = 26
MIN_TRAIN = 260  # Wochen
TOP_N = 50
STALE_DAYS = 300

ASSETS = {
    # code: (BIS-Land, OECD-Land, SIPRI-Name, FRED-Kurs, Vorzeichen)
    "EUR": ("XM", "EZ", "Germany", "DEXUSEU", +1),
    "JPY": ("JP", "JP", "Japan", "DEXJPUS", -1),
    "GBP": ("GB", "GB", "United Kingdom", "DEXUSUK", +1),
    "CHF": ("CH", "CH", "Switzerland", "DEXSZUS", -1),
    "CAD": ("CA", "CA", "Canada", "DEXCAUS", -1),
    "AUD": ("AU", "AU", "Australia", "DEXUSAL", +1),
    "NZD": ("NZ", "NZ", "New Zealand", "DEXUSNZ", +1),
}
ASSET_ORDER = ["DXY", "EUR", "JPY", "NZD", "CAD", "AUD", "CHF", "GBP"]


# ----------------------------------------------------------------- Hilfen
def monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    r = s.resample("MS")
    return (r.mean() if how == "mean" else r.last()).dropna()


def quarterly_to_monthly(s: pd.Series) -> pd.Series:
    return s.resample("MS").ffill()


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    return (100 * (s / s.shift(periods) - 1)).dropna()


@dataclass
class Basis:
    id: str          # Kurzname wie im Terminal (z.B. FEDFUNDS)
    label: str       # Beschreibung
    series: pd.Series  # monatlich, Index = Beobachtungsmonat
    lag: int         # Tage bis Veroeffentlichung (konservativ)
    group: str       # Zinsen / Inflation / Arbeitsmarkt / ...


# ------------------------------------------------------------ Datenaufbau
def load_fx(refresh: bool) -> dict[str, pd.Series]:
    px = {}
    for code, (_, _, _, sid, sign) in ASSETS.items():
        s = ds.fred(sid, refresh)
        px[code] = s if sign > 0 else 1.0 / s  # Preis der Waehrung in USD
    # Synthetischer US-Dollar-Index (ICE-Formel) aus FRED H.10
    eur, jpy, gbp = ds.fred("DEXUSEU", refresh), ds.fred("DEXJPUS", refresh), ds.fred("DEXUSUK", refresh)
    cad, sek, chf = ds.fred("DEXCAUS", refresh), ds.fred("DEXSDUS", refresh), ds.fred("DEXSZUS", refresh)
    df = pd.concat([eur, jpy, gbp, cad, sek, chf], axis=1).dropna()
    dxy = 50.14348112 * df.iloc[:, 0] ** -0.576 * df.iloc[:, 1] ** 0.136 * df.iloc[:, 2] ** -0.119 \
        * df.iloc[:, 3] ** 0.091 * df.iloc[:, 4] ** 0.042 * df.iloc[:, 5] ** 0.036
    px["DXY"] = dxy
    return px


def us_bases(refresh: bool) -> list[Basis]:
    f = lambda i: ds.fred(i, refresh)
    return [
        Basis("FEDFUNDS", "US Fed Funds Rate", monthly(f("FEDFUNDS")), 32, "Zinsen"),
        Basis("DGS2", "US 2J-Rendite (OIS-Proxy: Markterwartung Fed)", monthly(f("DGS2")), 3, "Zinsen"),
        Basis("T10Y2Y", "US Zinskurve 10J-2J", monthly(f("T10Y2Y")), 3, "Zinsen"),
        Basis("CPIAUCSL", "US Verbraucherpreise (CPI) YoY", yoy(monthly(f("CPIAUCSL"))), 45, "Inflation"),
        Basis("CPILFESL", "US Kern-CPI YoY", yoy(monthly(f("CPILFESL"))), 45, "Inflation"),
        Basis("UNRATE", "US Arbeitslosenquote", monthly(f("UNRATE")), 37, "Arbeitsmarkt"),
        Basis("PAYEMS", "US Beschäftigung (NFP) 3M-Veraenderung", monthly(f("PAYEMS")).diff(3).dropna(), 37, "Arbeitsmarkt"),
        Basis("ICSA", "US Erstanträge Arbeitslosenhilfe", monthly(f("ICSA")), 7, "Arbeitsmarkt"),
        Basis("INDPRO", "US Industrieproduktion YoY", yoy(monthly(f("INDPRO"))), 47, "Wachstum"),
        Basis("OPHNFB", "US Produktivität je Stunde YoY", yoy(quarterly_to_monthly(f("OPHNFB")), 12), 130, "Wachstum"),
        Basis("UMCSENT", "US Verbraucherstimmung (Michigan)", monthly(f("UMCSENT")), 30, "Stimmung"),
        Basis("BOPGSTB", "US Handelsbilanz (Mrd USD)", monthly(f("BOPGSTB")) / 1000, 65, "Handel"),
    ]


def country_series(cc_bis: str, cc_oecd: str, sipri_name: str, sipri: pd.DataFrame,
                   refresh: bool) -> dict[str, tuple[pd.Series, int]]:
    """Laenderspezifische Reihen (monatlich) + Veroeffentlichungs-Lag."""
    out: dict[str, tuple[pd.Series, int]] = {}

    def tryset(key, fn, lag):
        try:
            s = fn()
            if len(s) > 24:
                out[key] = (s, lag)
        except Exception as e:  # Quelle fehlt -> Basis entfaellt
            print(f"  ! {cc_oecd} {key}: {str(e)[:60]}", file=sys.stderr)

    tryset("POL", lambda: monthly(ds.bis("WS_CBPOL", f"M.{cc_bis}", refresh)), 32)
    tryset("CPI", lambda: ds.bis("WS_LONG_CPI", f"M.{cc_bis}.771", refresh), 50)
    if cc_oecd == "EZ":
        tryset("Y10", lambda: monthly(ds.ecb("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", refresh)), 3)
        tryset("R3M", lambda: ds.ecb("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", refresh), 32)
        tryset("UNR", lambda: ds.ecb("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T", refresh), 35)
    else:
        tryset("Y10", lambda: ds.fred(f"IRLTLT01{cc_oecd}M156N", refresh), 32)
        tryset("R3M", lambda: ds.fred(f"IR3TIB01{cc_oecd}M156N", refresh), 32)
        if cc_oecd in ("CH", "NZ"):
            tryset("UNR", lambda: quarterly_to_monthly(ds.fred(f"LRHUTTTT{cc_oecd}Q156S", refresh)), 140)
        elif cc_oecd == "US":
            tryset("UNR", lambda: ds.fred("UNRATE", refresh), 37)
        else:
            tryset("UNR", lambda: ds.fred(f"LRHUTTTT{cc_oecd}M156S", refresh), 45)
        x = lambda: ds.fred(f"XTEXVA01{cc_oecd}M667S", refresh)
        m = lambda: ds.fred(f"XTIMVA01{cc_oecd}M667S", refresh)
        tryset("TRD", lambda: (100 * (x() - m()) / (x() + m())).dropna(), 60)
        tryset("EXP", lambda: yoy(x()), 60)
        tryset("EQ", lambda: yoy(ds.fred(f"SPASTT01{cc_oecd}M661N", refresh)), 32)
    if sipri_name in sipri.columns:
        a = sipri[sipri_name].dropna() * 100
        a.index = pd.to_datetime([f"{y}-01-01" for y in a.index])
        out["MIL"] = (a.resample("MS").ffill(), 485)  # Jahr Y ab Mai Y+1 bekannt
    return out


DIFF_DEFS = {
    "POL": ("CBPOL-Diff", "Leitzins-Differenz zum USD (BIS)", "Zinsen"),
    "R3M": ("IR3M-Diff", "3M-Geldmarktzins-Differenz (OIS-nah)", "Zinsen"),
    "Y10": ("Y10-Diff", "10J-Rendite-Differenz", "Zinsen"),
    "CURVE": ("CURVE-Diff", "Zinskurven-Differenz (10J minus Leitzins)", "Zinsen"),
    "CPI": ("CPI-Diff", "Inflations-Differenz (CPI YoY, BIS)", "Inflation"),
    "REAL": ("REALRATE-Diff", "Realzins-Differenz (Leitzins minus CPI)", "Zinsen"),
    "UNR": ("UNR-Diff", "Arbeitslosenquoten-Differenz", "Arbeitsmarkt"),
    "TRD": ("TRADE", "Handelsbilanz (Exp-Imp)/(Exp+Imp) des Landes", "Handel"),
    "EXP": ("EXPORTS", "Exporte YoY des Landes", "Handel"),
    "EQ": ("EQUITY-Rel", "Aktienmarkt YoY relativ zu USA", "Stimmung"),
    "MIL": ("MILEX-Diff", "Militärausgaben % BIP minus USA (SIPRI)", "Geopolitik"),
}


def diff_bases(cc: dict, us: dict) -> list[Basis]:
    res = []
    def add(key, s, lag):
        bid, lab, grp = DIFF_DEFS[key]
        res.append(Basis(bid, lab, s.dropna(), lag, grp))
    for key in ["POL", "R3M", "Y10", "CPI", "UNR", "EQ", "MIL"]:
        if key in cc and key in us:
            (a, la), (b, lb) = cc[key], us[key]
            add(key, a - b, max(la, lb))
    for key in ["TRD", "EXP"]:
        if key in cc:
            add(key, *cc[key])
    if all(k in cc and k in us for k in ("Y10", "POL")):
        add("CURVE", (cc["Y10"][0] - cc["POL"][0]) - (us["Y10"][0] - us["POL"][0]), 32)
    if all(k in cc and k in us for k in ("POL", "CPI")):
        add("REAL", (cc["POL"][0] - cc["CPI"][0]) - (us["POL"][0] - us["CPI"][0]), 50)
    return res


# --------------------------------------------------------------- Modelle
def features(b: Basis, dates: pd.DatetimeIndex) -> pd.DataFrame:
    s = b.series.sort_index()
    f = pd.DataFrame({"lvl": s, "d3": s - s.shift(3), "d12": s - s.shift(12)}).dropna()
    # Monatswert gilt ab Monatsende + Veroeffentlichungs-Lag
    f.index = f.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=b.lag)
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f.reindex(dates, method="ffill")


def _fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    # L2-regularisierte logistische Regression per Newton (3 Features -> schnell)
    A1 = np.c_[np.ones(len(A)), A]
    B1 = np.c_[np.ones(len(B)), B]
    w = np.zeros(A1.shape[1])
    lam = 2.0
    for _ in range(25):
        p = 1 / (1 + np.exp(-A1 @ w))
        g = A1.T @ (p - ytr) + lam * np.r_[0, w[1:]]
        H = (A1 * (p * (1 - p))[:, None]).T @ A1 + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H + 1e-8 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return 1 / (1 + np.exp(-B1 @ w))


def run_model(X: pd.DataFrame, price: pd.Series, h: int, today: pd.Timestamp):
    dates = X.index
    fwd = price.reindex(dates + pd.Timedelta(days=h), method="ffill").values
    now = price.reindex(dates, method="ffill").values
    ret = fwd / now - 1
    done = (dates + pd.Timedelta(days=h)) <= price.index[-1]
    ret = np.where(done, ret, np.nan)
    ok = X.notna().all(axis=1).values & ~np.isnan(now)
    Xv, y = X.values, (ret > 0).astype(float)

    first = dates[ok][0] if ok.any() else None
    if first is None:
        return None
    start = max(OOS_START, first + pd.Timedelta(weeks=MIN_TRAIN))
    probs = np.full(len(dates), np.nan)
    naive = np.full(len(dates), np.nan)
    refits = pd.date_range(start, dates[-1] + pd.Timedelta(days=1), freq=f"{REFIT_WEEKS}W-FRI")
    if len(refits) < 4:
        return None
    for i, R in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else dates[-1] + pd.Timedelta(days=7)
        tr = ok & ((dates + pd.Timedelta(days=h)) <= R) & ~np.isnan(ret)
        te = ok & (dates >= R) & (dates < nxt)
        if tr.sum() < MIN_TRAIN // 2 or not te.any():
            continue
        probs[te] = _fit_predict(Xv[tr], y[tr], Xv[te])
        naive[te] = float(y[tr].mean() > 0.5)
    ev = ~np.isnan(probs) & ~np.isnan(ret)
    n = int(ev.sum())
    if n < 150:
        return None
    hit = ((probs[ev] > 0.5) == (y[ev] == 1))
    nhit = (naive[ev] == y[ev])
    half = n // 2
    da, da1, da2 = hit.mean(), hit[:half].mean(), hit[half:].mean()
    n_eff = max(int(n * 7 / max(h, 7)), 10)
    pval = binomtest(int(round(da * n_eff)), n_eff, 0.5, alternative="greater").pvalue

    # Aktuelle Prognose: auf allen abgeschlossenen Beispielen trainieren
    tr = ok & ~np.isnan(ret)
    last = np.where(X.notna().all(axis=1).values)[0]
    prob_now = float(_fit_predict(Xv[tr], y[tr], Xv[last[-1:]])[0]) if len(last) else np.nan

    # OOS-Prognosen fuer den Ensemble-Rueckblick (monatlich ausgeduennt)
    hist = pd.Series(probs, index=dates).dropna()
    return dict(da=float(da), da1=float(da1), da2=float(da2), naive=float(nhit.mean()),
                p=float(pval), n=n, n_eff=n_eff, prob=prob_now,
                oos_from=str(dates[ev][0].date()), hist=hist)


def tier(r) -> str | None:
    """A = signifikant (strenger Test), B = robust (stabil, kleiner Vorteil)."""
    edge = r["da"] - r["naive"]
    if (r["da"] >= 0.53 and r["da1"] >= 0.51 and r["da2"] >= 0.51
            and edge >= 0.02 and r["p"] < 0.05):
        return "A"
    if r["da"] >= 0.52 and r["da1"] >= 0.505 and r["da2"] >= 0.505 and edge >= 0.01:
        return "B"
    return None


N_NOISE = 12  # Zufalls-Basen je Waehrung (Benchmark gegen positive Varianz)


def noise_bases(asset: str, start="1990-01-01") -> list[Basis]:
    rng = np.random.default_rng(sum(map(ord, asset)) * 7919)
    idx = pd.date_range(start, pd.Timestamp.today(), freq="MS")
    return [Basis(f"NOISE{i}", "Zufallsreihe", pd.Series(rng.standard_normal(len(idx)).cumsum(), idx),
                  32, "Zufall") for i in range(N_NOISE)]


def label(prob: float) -> str:
    if prob >= 0.52:
        return "Bullish"
    if prob <= 0.48:
        return "Bearish"
    return "Neutral"


# ------------------------------------------------------------------ Main
def main(refresh: bool = True):
    today = pd.Timestamp(date.today())
    print("Lade Daten ...")
    px = load_fx(refresh)
    sipri = ds.sipri_share_of_gdp(refresh)
    us_cc = country_series("US", "US", "United States of America", sipri, refresh)
    ub = us_bases(refresh)

    bases_by_asset: dict[str, list[Basis]] = {"DXY": list(ub)}
    if "MIL" in us_cc:
        bases_by_asset["DXY"].append(Basis("MILEX-US", "US Militärausgaben % BIP (SIPRI)",
                                           us_cc["MIL"][0], 485, "Geopolitik"))
    for code, (cb, co, sn, _, _) in ASSETS.items():
        cc = country_series(cb, co, sn, sipri, refresh)
        bases_by_asset[code] = list(ub) + diff_bases(cc, us_cc)

    tasks, meta_bases = [], {}
    for asset, bases in bases_by_asset.items():
        price = px[asset].dropna()
        dates = pd.DatetimeIndex(price.resample("W-FRI").last().dropna().index)
        dates = dates[dates >= "1999-01-08"]
        dates = dates.append(pd.DatetimeIndex([today])) if dates[-1] < today else dates
        for b in bases:
            age = (today - b.series.index[-1]).days
            if age > STALE_DAYS + b.lag:
                print(f"  - {asset} {b.id}: Daten veraltet ({b.series.index[-1].date()}), uebersprungen")
                continue
            meta_bases[b.id] = dict(label=b.label, group=b.group)
            X = features(b, dates)
            for h in HORIZONS:
                tasks.append((asset, b, h, X, price))
        for b in noise_bases(asset):
            X = features(b, dates)
            for h in HORIZONS:
                tasks.append((asset, b, h, X, price))
    print(f"Trainiere {len(tasks)} Modelle inkl. Zufalls-Benchmark (Walk-Forward) ...")
    results = Parallel(n_jobs=-1, batch_size=16)(
        delayed(run_model)(X, price, h, today) for (_, _, h, X, price) in tasks)

    models = []
    cand = {a: dict(n=0, A=0, B=0, da=[], noise_n=0, noise_A=0, noise_B=0, noise_da=[])
            for a in ASSET_ORDER}
    for (asset, b, h, _, _), r in zip(tasks, results):
        if r is None:
            continue
        c, t = cand[asset], tier(r)
        if b.group == "Zufall":
            c["noise_n"] += 1
            c["noise_da"].append(round(r["da"] * 100, 1))
            if t:
                c[f"noise_{t}"] += 1
            continue
        c["n"] += 1
        c["da"].append(round(r["da"] * 100, 1))
        if not t:
            continue
        c[t] += 1
        models.append(dict(basis=b.id, asset=asset, h=h, tier=t, da=round(r["da"] * 100, 1),
                           da1=round(r["da1"] * 100, 1), da2=round(r["da2"] * 100, 1),
                           naive=round(r["naive"] * 100, 1), p=float(f"{r['p']:.2g}"),
                           n_eff=r["n_eff"], prob=round(r["prob"], 3), pred=label(r["prob"]),
                           date=str(b.series.index[-1].date()), oos_from=r["oos_from"],
                           _hist=r["hist"]))
    # Top-N je Waehrung: erst signifikante (A), dann robuste (B), jeweils nach DA
    sel = []
    for a in ASSET_ORDER:
        ms = sorted([m for m in models if m["asset"] == a],
                    key=lambda m: (m["tier"], -m["da"], m["p"]))[:TOP_N]
        sel += ms

    # Ensemble-Rueckblick: Anteil bullischer Modelle je Woche (OOS-Prognosen)
    history = {}
    for a in ASSET_ORDER:
        hs = [ (m["_hist"] > 0.5).astype(float) for m in sel if m["asset"] == a]
        if hs:
            df = pd.concat(hs, axis=1)
            share = df.mean(1)[df.count(1) >= max(3, len(hs) // 4)]
            share = share[share.index >= "2015-01-01"].resample("W-FRI").last().dropna()
            p = px[a].reindex(share.index, method="ffill")
            history[a] = dict(d=[str(x.date()) for x in share.index],
                              bull=[round(v * 100, 1) for v in share.values],
                              px=[round(float(v), 5) for v in p.values])
    for m in sel:
        m.pop("_hist")

    latest = {}
    for b in ub:
        s = b.series.dropna()
        latest[b.id] = dict(v=round(float(s.iloc[-1]), 2), d=str(s.index[-1].date()))
    milex = {}
    for name, code in [("United States of America", "USA"), ("Germany", "DEU (EUR)"), ("Japan", "JPN"),
                       ("United Kingdom", "GBR"), ("Switzerland", "CHE"), ("Canada", "CAN"),
                       ("Australia", "AUS"), ("New Zealand", "NZL")]:
        if name in sipri.columns:
            s = sipri[name].dropna() * 100
            milex[code] = dict(y=int(s.index[-1]), v=round(float(s.iloc[-1]), 2),
                               chg=round(float(s.iloc[-1] - s.iloc[-6]), 2))
    out = dict(generated=str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
               fx_through=str(min(p.index[-1] for p in px.values()).date()),
               assets=ASSET_ORDER, horizons=HORIZONS, models=sel, candidates=cand,
               bases=meta_bases, latest=latest, milex=milex, history=history,
               fx_last={a: round(float(px[a].iloc[-1]), 5) for a in ASSET_ORDER})
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False))
    print("Kandidaten je Waehrung (A=signifikant, B=robust):")
    for a in ASSET_ORDER:
        c = cand[a]
        k = [m for m in sel if m["asset"] == a]
        bull = sum(m["pred"] == "Bullish" for m in k)
        print(f"  {a}: echt {c['n']:4d} A={c['A']:3d} B={c['B']:3d} | Zufall {c['noise_n']:3d} "
              f"A={c['noise_A']:2d} B={c['noise_B']:3d} | angezeigt {len(k):3d} bullish {bull}")
    return out


if __name__ == "__main__":
    main(refresh="--cache" not in sys.argv)
