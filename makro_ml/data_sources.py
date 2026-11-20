"""Datenabruf fuer das Makro-ML-Cockpit (FRED, BIS, EZB, SIPRI).

Alle Quellen sind kostenlos und brauchen keinen API-Key. Rohdaten werden
in ``cache/`` abgelegt; ``refresh=True`` laedt neu, sonst wird der Cache
genutzt, wenn er juenger als ``max_age_h`` Stunden ist.
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
UA: dict = {}  # FRED blockt eigene User-Agents -> Python-Standard nutzen


def _download(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=90).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(url)


def _cached(name: str, url: str, refresh: bool, max_age_h: float = 20) -> bytes:
    path = CACHE / name
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age_h * 3600:
        return path.read_bytes()
    try:
        raw = _download(url)
        head = raw[:200].lstrip()
        if len(raw) < 20 or head[:1] in (b"<", b"{", b"["):  # HTML-/JSON-Fehlerseite statt CSV
            raise ValueError(f"Unerwartete Antwort von {url[:60]}")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return raw
    except Exception:
        if path.exists():  # offline -> letzter Stand
            return path.read_bytes()
        raise


def to_json(obj) -> str:
    """JSON ohne NaN/Inf (die waeren ungueltiges JSON und wuerden das Cockpit leer lassen) -> null."""
    import json
    import math

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) or type(o).__module__ == "numpy":
            try:
                f = float(o)
            except (TypeError, ValueError):
                return o
            if not math.isfinite(f):
                return None
            return int(o) if type(o).__name__.startswith("int") else f
        return o
    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False)


def fred(series_id: str, refresh: bool = False) -> pd.Series:
    raw = _cached(f"fred_{series_id}.csv",
                  f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", refresh)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "v"]
    s = pd.Series(pd.to_numeric(df["v"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]), name=series_id)
    return s.dropna()


def bis(flow: str, key: str, refresh: bool = False) -> pd.Series:
    raw = _cached(f"bis_{flow}_{key}.csv",
                  f"https://stats.bis.org/api/v2/data/dataflow/BIS/{flow}/1.0/{key}?format=csv",
                  refresh)
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name=f"{flow}.{key}")
    return s.dropna().sort_index()


def bis_daily_recent(flow: str, key: str, days: int = 400, refresh: bool = True) -> pd.Series:
    """Tageswerte der letzten ``days`` Tage (z. B. Leitzinsen); ``refresh=False`` nutzt den Cache (bis 20 h)."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    raw = _cached(f"bis_{flow}_{key}_recent.csv",
                  f"https://stats.bis.org/api/v2/data/dataflow/BIS/{flow}/1.0/{key}?format=csv&startPeriod={start}",
                  refresh)
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name=f"{flow}.{key}")
    return s.dropna().sort_index()


def ecb(flow: str, key: str, refresh: bool = False) -> pd.Series:
    raw = _cached(f"ecb_{flow}_{key}.csv",
                  f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata",
                  refresh)
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name=f"{flow}.{key}")
    return s.dropna().sort_index()


def ecb_fx_daily(days: int = 450) -> pd.DataFrame:
    """Taegliche EZB-Referenzkurse (16 Uhr MEZ, ohne Key), umgerechnet in US-Dollar je Waehrungseinheit.
    Spalten EUR, JPY, GBP, CHF, CAD, AUD, NZD, SEK. Nur fuer den aktuellen Stand gedacht (siehe live_extend)."""
    cur = ["USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK"]
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    raw = _cached("ecb_fx_daily.csv", "https://data-api.ecb.europa.eu/service/data/EXR/D." + "+".join(cur)
                  + f".EUR.SP00.A?format=csvdata&startPeriod={start}", True)
    df = pd.read_csv(io.BytesIO(raw))
    p = df.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    p.index = pd.to_datetime(p.index)
    p = p.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"EUR": p["USD"]})                     # USD je EUR
    for c in cur[1:]:
        out[c] = p["USD"] / p[c]                              # USD je Einheit = (USD je EUR) / (Einheit je EUR)
    return out.dropna(how="all")


def live_extend(fed: pd.Series, ecb: pd.Series) -> tuple[pd.Series, str | None]:
    """Fed-Reihe (Backtest-Quelle) um die Tage nach ihrem letzten Wert verlaengern. Uebernommen wird nur die
    VERAENDERUNG der EZB-Kurse seit dem letzten Fed-Tag (Verkettung), nicht ihr Niveau: Fed (12 Uhr New York)
    und EZB (14:15 Uhr Frankfurt) messen zu verschiedenen Zeiten. Gibt die Reihe und das letzte Datum zurueck."""
    fed = fed.dropna()
    ecb = ecb.dropna()
    base = ecb[ecb.index <= fed.index[-1]]
    if base.empty:
        return fed, None
    new = ecb[ecb.index > fed.index[-1]]
    if new.empty:
        return fed, None
    ext = fed.iloc[-1] * new / base.iloc[-1]
    return pd.concat([fed, ext]), str(new.index[-1].date())


def live_prices(px: dict) -> tuple[dict, dict]:
    """Fed-Kurse (USD je Waehrung, dazu 'DXY') um die neuesten EZB-Tage verlaengert (verkettet, siehe live_extend).
    Nur fuer den aktuellen Stand (Rangliste, 52 Wochen), nie fuer Backtests. Rueckfall bei Fehler: reine Fed-Kurse."""
    meta = {"fed_last": str(min(s.dropna().index[-1] for k, s in px.items() if k != "USD").date()), "ecb_last": None}
    try:
        e = ecb_fx_daily()
    except Exception as ex:  # EZB nicht erreichbar -> Fed-Stand bleibt
        meta["error"] = type(ex).__name__
        return dict(px), meta
    out = dict(px)
    for c in ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]:
        if c in px and c in e:
            out[c], d = live_extend(px[c], e[c])
            meta["ecb_last"] = d or meta["ecb_last"]
    if "DXY" in px:  # Dollar-Index nach ICE-Formel aus den EZB-Kursen, ebenfalls nur verkettet
        d = e[["EUR", "JPY", "GBP", "CAD", "SEK", "CHF"]].dropna()
        dxy = (50.14348112 * d["EUR"] ** -0.576 * (1 / d["JPY"]) ** 0.136 * d["GBP"] ** -0.119
               * (1 / d["CAD"]) ** 0.091 * (1 / d["SEK"]) ** 0.042 * (1 / d["CHF"]) ** 0.036)
        out["DXY"], _ = live_extend(px["DXY"], dxy)
    return out, meta


SIPRI_PAGE = "https://www.sipri.org/databases/milex"


def sipri_share_of_gdp(refresh: bool = False) -> pd.DataFrame:
    """Militaerausgaben in % des BIP (jaehrlich), Spalten = Laendername."""
    path = CACHE / "sipri_milex.xlsx"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400:
        try:
            html = _download(SIPRI_PAGE).decode("utf-8", "ignore")
            import re
            m = re.search(r'href="([^"]*SIPRI-Milex-data[^"]*\.xlsx)"', html)
            url = m.group(1)
            if url.startswith("//"):
                url = "https:" + url
            path.write_bytes(_download(url))
        except Exception:
            if not path.exists():
                raise
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if "GDP" in s and "Share" in s)
    raw = xl.parse(sheet, header=None)
    hdr = next(i for i in range(15) if str(raw.iloc[i, 0]).strip().lower() == "country")
    df = xl.parse(sheet, header=hdr)
    df = df.rename(columns={df.columns[0]: "Country"}).set_index("Country")
    years = [c for c in df.columns if str(c).strip().split(".")[0].isdigit()]
    df = df[years]
    df.columns = [int(str(c).split(".")[0]) for c in years]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.T  # index = Jahr


def cftc_net(code: str, refresh: bool = False) -> pd.DataFrame:
    """CFTC Commitments of Traders (Legacy, Futures only) fuer einen Kontrakt.

    Woechentlich (Index = Stichtag Dienstag). Netto-Positionen in % Open Interest:
    ``net_pct`` = Non-Commercials (grosse Spekulanten), ``comm_pct`` = Commercials
    (Absicherer, z. B. Banken/Unternehmen), ``small_pct`` = Non-Reportables (Kleinspekulanten).
    """
    cols = ["noncomm_positions_long_all", "noncomm_positions_short_all", "comm_positions_long_all",
            "comm_positions_short_all", "nonrept_positions_long_all", "nonrept_positions_short_all",
            "open_interest_all"]
    url = ("https://publicreporting.cftc.gov/resource/6dca-aqww.csv?"
           "$select=report_date_as_yyyy_mm_dd," + ",".join(cols) +
           f"&$where=cftc_contract_market_code=%27{code}%27"
           "&$order=report_date_as_yyyy_mm_dd&$limit=50000")
    raw = _cached(f"cftc3_{code}.csv", url, refresh)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "nc_l", "nc_s", "c_l", "c_s", "nr_l", "nr_s", "oi"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce").dropna()
    df = df[~df.index.duplicated(keep="last")]
    df["net_pct"] = 100 * (df["nc_l"] - df["nc_s"]) / df["oi"]
    df["comm_pct"] = 100 * (df["c_l"] - df["c_s"]) / df["oi"]
    df["small_pct"] = 100 * (df["nr_l"] - df["nr_s"]) / df["oi"]
    return df


def alfred_vintages(series_id: str, vintage_dates: list[pd.Timestamp], refresh: bool = False) -> pd.DataFrame:
    """ALFRED-Echtzeitdaten: je Datenstand (Spalte) die damals veroeffentlichte Reihe.

    Abruf ohne API-Key ueber den ALFRED-CSV-Download, 12 Datenstaende je Anfrage.
    Alte Datenstaende aendern sich nie und werden dauerhaft im Cache gehalten; die
    letzten zwei werden bei ``refresh`` neu geladen.
    """
    path = CACHE / f"alfred_{series_id}.csv"
    have = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    want = [pd.Timestamp(v).strftime("%Y-%m-%d") for v in vintage_dates]
    missing = [v for v in want if v not in have.columns]
    if refresh:
        missing = sorted(set(missing) | set(want[-2:]))
    for i in range(0, len(missing), 12):
        chunk = missing[i:i + 12]
        url = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=" + ",".join([series_id] * len(chunk))
               + "&vintage_date=" + ",".join(chunk))
        try:
            df = pd.read_csv(io.BytesIO(_download(url)))
        except Exception:
            # Einzeln versuchen (ein fehlender Datenstand laesst die ganze Anfrage scheitern)
            parts = []
            for v in chunk:
                try:
                    parts.append(pd.read_csv(io.BytesIO(_download(
                        f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series_id}&vintage_date={v}",
                        tries=1))))
                except Exception:
                    pass
                time.sleep(0.3)
            if not parts:
                continue
            df = parts[0]
            for p in parts[1:]:
                df = df.merge(p, on="observation_date", how="outer")
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.set_index("observation_date").apply(pd.to_numeric, errors="coerce")
        df.columns = [f"{c.split('_')[-1][:4]}-{c.split('_')[-1][4:6]}-{c.split('_')[-1][6:8]}" for c in df.columns]
        have = df if have.empty else have.drop(columns=[c for c in df.columns if c in have.columns]).join(df, how="outer")
        time.sleep(0.5)
    have = have.reindex(sorted(have.columns), axis=1)
    have.to_csv(path)
    return have[[c for c in want if c in have.columns]]
