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

HORIZONS = list(range(10, 101, 5)) + [120, 150, 180, 240, 300]
OOS_START = pd.Timestamp("2008-01-04")
HOLDOUT_START = pd.Timestamp("2022-01-01")  # ab hier: unberuehrter Schlusstest
PERIODS = [("2008–2012", "2008-01-01", "2013-01-01"), ("2013–2017", "2013-01-01", "2018-01-01"),
           ("2018–2021", "2018-01-01", "2022-01-01"), ("2022–heute", "2022-01-01", "2100-01-01")]
COT_CODES = {"DXY": "098662", "EUR": "099741", "JPY": "097741", "GBP": "096742",
             "CHF": "092741", "CAD": "090741", "AUD": "232741", "NZD": "112741"}
COMMODITIES = {  # Rohstoffe, die fuer Rohstoffwaehrungen zaehlen (FRED/IMF)
    "CAD": [("OIL-WTI", "Ölpreis WTI YoY", "DCOILWTICO", 3)],
    "AUD": [("IRONORE", "Eisenerzpreis YoY", "PIORECRUSDM", 45), ("COPPER", "Kupferpreis YoY", "PCOPPUSDM", 45)],
    "NZD": [("FOOD-IDX", "Nahrungsmittel-Preisindex YoY (u. a. Milch)", "PFOODINDEXM", 45)],
}
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


def pricing(refresh: bool) -> dict[str, pd.Series]:
    """Eingepreiste Zinsaenderung (Prozentpunkte): Marktrendite minus aktueller Leitzins.

    Frei verfuegbarer Ersatz fuer OIS-Kurven / FedWatch / ECB-Watch mit langer Historie.
    """
    dff, dfr = ds.fred("DFF", refresh), ds.fred("ECBDFR", refresh)
    out = {}
    for k, y in [("US1", ds.fred("DGS1", refresh)), ("US2", ds.fred("DGS2", refresh))]:
        out[k] = (y - dff.reindex(y.index, method="ffill")).dropna()
    for k, key in [("EZ1", "SR_1Y"), ("EZ2", "SR_2Y")]:
        y = ds.ecb("YC", f"B.U2.EUR.4F.G_N_A.SV_C_YM.{key}", refresh)
        out[k] = (y - dfr.reindex(y.index, method="ffill")).dropna()
    return out


def pro_bases(asset: str, px: pd.Series, cot_cc: str, refresh: bool) -> list[Basis]:
    """Momentum, Bewertung (REER), CFTC-Positionierung und Rohstoffe."""
    res = []
    pm = px.resample("MS").last().dropna()
    res.append(Basis("MOM-12M", "Momentum: eigene Kursentwicklung 12 Monate", yoy(pm), 1, "Momentum"))
    res.append(Basis("MOM-3M", "Momentum: eigene Kursentwicklung 3 Monate", yoy(pm, 3), 1, "Momentum"))
    try:
        reer = ds.bis("WS_EER", f"M.R.B.{cot_cc}", refresh)
        gap = (100 * (reer / reer.rolling(120, min_periods=60).mean() - 1)).dropna()
        res.append(Basis("REER-GAP", "Bewertung: realer effektiver Wechselkurs vs. 10J-Schnitt (BIS)", gap, 45, "Bewertung"))
    except Exception as e:
        print(f"  ! {asset} REER: {str(e)[:60]}", file=sys.stderr)
    try:
        cot = ds.cftc_net(COT_CODES[asset], refresh)
        res.append(Basis("COT-NET", "CFTC: Netto-Position Spekulanten in % Open Interest",
                         monthly(cot["net_pct"]), 3, "Positionierung"))
    except Exception as e:
        print(f"  ! {asset} COT: {str(e)[:60]}", file=sys.stderr)
    for bid, lab, sid, lag in COMMODITIES.get(asset, []):
        try:
            res.append(Basis(bid, lab, yoy(monthly(ds.fred(sid, refresh))), lag, "Rohstoffe"))
        except Exception as e:
            print(f"  ! {asset} {bid}: {str(e)[:60]}", file=sys.stderr)
    return res


def ensemble_backtest(entries: list[tuple[str, pd.Series]], price: pd.Series) -> dict | None:
    """Woechentlich umgeschichtete Strategie: Long/Short nach Mehrheit der Basen.

    Jede Basis zaehlt einmal (Mittel ihrer Modelle), damit fast identische Modelle
    (z. B. FEDFUNDS 60/61/62 Tage) nicht mehrfach abstimmen.
    """
    if not entries:
        return None
    votes = pd.concat({i: np.sign(hst - 0.5) for i, (_, hst) in enumerate(entries)}, axis=1)
    groups = pd.Series([b for b, _ in entries])
    by_basis = votes.T.groupby(groups.values).mean().T
    score = by_basis.mean(axis=1)
    idx = score.index
    p = price.reindex(idx, method="ffill")
    ret = (p.shift(-1) / p - 1)
    pos = np.sign(score).where(score.abs() >= 0.1, 0.0)
    strat = (pos * ret).dropna()
    bh = ret.reindex(strat.index)

    def stats(r, pos_):
        if len(r) < 20:
            return None
        ann, vol = r.mean() * 52, r.std() * np.sqrt(52)
        eq = (1 + r).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        act = pos_.reindex(r.index) != 0
        hit = float((np.sign(ret.reindex(r.index)[act]) == pos_.reindex(r.index)[act]).mean()) if act.any() else None
        return dict(ret=round(ann * 100, 1), vol=round(vol * 100, 1),
                    sharpe=round(ann / vol, 2) if vol > 0 else None, dd=round(dd * 100, 1),
                    hit=None if hit is None else round(hit * 100, 1), active=round(float(act.mean()) * 100))

    out = {"periods": {}, "bh": {}}
    for name, a, b in PERIODS + [("Auswahl 2008–2021", "2008-01-01", "2022-01-01"),
                                 ("Schlusstest 2022–heute", "2022-01-01", "2100-01-01")]:
        m = (strat.index >= a) & (strat.index < b)
        out["periods"][name] = stats(strat[m], pos)
        out["bh"][name] = stats(bh[m], pd.Series(1.0, index=bh.index))
    eq = (1 + strat).cumprod()
    eqb = (1 + bh).cumprod()
    pick = eq.index[::2]
    out["equity"] = dict(d=[str(x.date()) for x in pick], s=[round(float(v), 4) for v in eq[pick]],
                         b=[round(float(v), 4) for v in eqb[pick]])
    out["score_now"] = round(float(score.iloc[-1]), 2) if len(score) else None
    return out


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
        Basis("VIX", "Volatilitätsindex VIX (Risikostimmung)", monthly(f("VIXCLS")), 3, "Risiko"),
        Basis("CREDIT", "Kreditspread Baa-Anleihen minus 10J-Treasury", monthly(f("BAA10Y")), 3, "Risiko"),
        Basis("FF-PRICED-1Y", "US Zinseinpreisung 12M: 1J-Rendite minus Fed Funds (OIS-Proxy)",
              (monthly(f("DGS1")) - monthly(f("DFF"))).dropna(), 3, "Zinseinpreisung"),
        Basis("FF-PRICED-2Y", "US Zinseinpreisung 24M: 2J-Rendite minus Fed Funds (OIS-Proxy)",
              (monthly(f("DGS2")) - monthly(f("DFF"))).dropna(), 3, "Zinseinpreisung"),
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
    if int(ev.sum()) < 150:
        return None
    hit = ((probs > 0.5) == (y == 1))
    nhit = (naive == y)
    full = _metrics(hit[ev], nhit[ev], h)
    # Auswahl-Zeitraum (bis Ende 2021) und unberuehrter Schlusstest (ab 2022)
    pre_m = ev & (dates < HOLDOUT_START)
    post_m = ev & (dates >= HOLDOUT_START)
    pre = _metrics(hit[pre_m], nhit[pre_m], h) if pre_m.sum() >= 100 else None
    post_da = float(hit[post_m].mean()) if post_m.sum() >= 20 else None

    # Aktuelle Prognose: auf allen abgeschlossenen Beispielen trainieren
    tr = ok & ~np.isnan(ret)
    last = np.where(X.notna().all(axis=1).values)[0]
    prob_now = float(_fit_predict(Xv[tr], y[tr], Xv[last[-1:]])[0]) if len(last) else np.nan

    hist = pd.Series(probs, index=dates).dropna()
    return dict(**full, prob=prob_now, oos_from=str(dates[ev][0].date()), hist=hist,
                pre=pre, post_da=post_da, n_post=int(post_m.sum()))


def _metrics(hit: np.ndarray, nhit: np.ndarray, h: int) -> dict:
    n = len(hit)
    half = n // 2
    da = hit.mean()
    n_eff = max(int(n * 7 / max(h, 7)), 10)
    pval = binomtest(int(round(da * n_eff)), n_eff, 0.5, alternative="greater").pvalue
    return dict(da=float(da), da1=float(hit[:half].mean()), da2=float(hit[half:].mean()),
                naive=float(nhit.mean()), p=float(pval), n=n, n_eff=n_eff)


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
    pr = pricing(refresh)
    bis_cc = {"DXY": "US", **{k: v[0] for k, v in ASSETS.items()}}
    for a in ASSET_ORDER:
        bases_by_asset[a] += pro_bases(a, px[a], bis_cc[a], refresh)
    bases_by_asset["EUR"].append(Basis(
        "ECB-PRICED-Diff", "Zinseinpreisung EZB minus Fed (1J-Rendite minus Leitzins, Differenz)",
        (monthly(pr["EZ1"]) - monthly(pr["US1"])).dropna(), 3, "Zinseinpreisung"))

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
    pre_pool = {a: {"real": [], "noise": []} for a in ASSET_ORDER}
    for (asset, b, h, _, _), r in zip(tasks, results):
        if r is None:
            continue
        if r["pre"] is not None:
            tp = tier(r["pre"])
            if tp:
                pre_pool[asset]["noise" if b.group == "Zufall" else "real"].append(
                    dict(basis=b.id, h=h, tier=tp, da_pre=r["pre"]["da"], post=r["post_da"], hist=r["hist"]))
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

    # Schlusstest: Auswahl NUR mit 2008–2021, Bewertung 2022–heute (+ Zufalls-Vergleich)
    validation = {}
    for a in ASSET_ORDER:
        v = {}
        for kind in ("real", "noise"):
            pool = sorted(pre_pool[a][kind], key=lambda m: (m["tier"], -m["da_pre"]))[:TOP_N]
            posts = [m["post"] for m in pool if m["post"] is not None]
            v[kind] = dict(n=len(pool), da_pre=round(float(np.mean([m["da_pre"] for m in pool])) * 100, 1) if pool else None,
                           da_post=round(float(np.mean(posts)) * 100, 1) if posts else None)
            bt = ensemble_backtest([(m["basis"], m["hist"]) for m in pool], px[a].dropna())
            if kind == "real":
                v["backtest"] = bt
            else:
                v["noise_holdout"] = bt["periods"].get("Schlusstest 2022–heute") if bt else None
        validation[a] = v

    # CFTC-Positionierung fuer die Anzeige
    cot_view = {}
    for a in ASSET_ORDER:
        try:
            c = ds.cftc_net(COT_CODES[a], False)["net_pct"].dropna()
            last3 = c[c.index >= c.index[-1] - pd.Timedelta(days=3 * 365)]
            pct = float((last3 < c.iloc[-1]).mean() * 100)
            w = c[c.index >= c.index[-1] - pd.Timedelta(days=365)]
            cot_view[a] = dict(net=round(float(c.iloc[-1]), 1), chg4=round(float(c.iloc[-1] - c.iloc[-5]), 1),
                               pct3y=round(pct), d=str(c.index[-1].date()), spark=[round(float(x), 1) for x in w.values])
        except Exception as e:
            print(f"  ! COT-Anzeige {a}: {e}", file=sys.stderr)

    priced = {}
    for k, lab in [("US1", "Fed · 12 Monate"), ("US2", "Fed · 24 Monate"), ("EZ1", "EZB · 12 Monate"), ("EZ2", "EZB · 24 Monate")]:
        s = pr[k].dropna()
        m1 = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
        w = s[s.index >= s.index[-1] - pd.Timedelta(days=365)].resample("W-FRI").last().dropna()
        priced[k] = dict(label=lab, bp=round(float(s.iloc[-1]) * 100), chg=round(float(s.iloc[-1] - m1.iloc[-1]) * 100),
                         d=str(s.index[-1].date()), spark=[round(float(v) * 100) for v in w.values])
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
               priced=priced, validation=validation, cot=cot_view, fx_last={a: round(float(px[a].iloc[-1]), 5) for a in ASSET_ORDER})
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False))
    print("Kandidaten je Waehrung (A=signifikant, B=robust):")
    for a in ASSET_ORDER:
        c = cand[a]
        k = [m for m in sel if m["asset"] == a]
        bull = sum(m["pred"] == "Bullish" for m in k)
        print(f"  {a}: echt {c['n']:4d} A={c['A']:3d} B={c['B']:3d} | Zufall {c['noise_n']:3d} "
              f"A={c['noise_A']:2d} B={c['noise_B']:3d} | angezeigt {len(k):3d} bullish {bull}")
    print("Schlusstest 2022–heute (Auswahl nur mit 2008–2021):")
    for a in ASSET_ORDER:
        v = validation[a]
        bt = (v.get("backtest") or {}).get("periods", {}).get("Schlusstest 2022–heute") or {}
        print(f"  {a}: echt DA vorher {v['real']['da_pre']} -> Schlusstest {v['real']['da_post']} | "
              f"Zufall {v['noise']['da_pre']} -> {v['noise']['da_post']} | Ensemble Sharpe {bt.get('sharpe')} Treffer {bt.get('hit')}")
    return out


if __name__ == "__main__":
    main(refresh="--cache" not in sys.argv)
