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
        path.write_bytes(raw)
        return raw
    except Exception:
        if path.exists():  # offline -> letzter Stand
            return path.read_bytes()
        raise


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


def ecb(flow: str, key: str, refresh: bool = False) -> pd.Series:
    raw = _cached(f"ecb_{flow}_{key}.csv",
                  f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata",
                  refresh)
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name=f"{flow}.{key}")
    return s.dropna().sort_index()


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

    Rueckgabe (woechentlich, Index = Stichtag Dienstag): long, short, oi, net_pct
    (Netto-Position der Spekulanten / Open Interest in %).
    """
    url = ("https://publicreporting.cftc.gov/resource/6dca-aqww.csv?"
           "$select=report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
           "noncomm_positions_short_all,open_interest_all"
           f"&$where=cftc_contract_market_code=%27{code}%27"
           "&$order=report_date_as_yyyy_mm_dd&$limit=50000")
    raw = _cached(f"cftc_{code}.csv", url, refresh)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "long", "short", "oi"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce").dropna()
    df = df[~df.index.duplicated(keep="last")]
    df["net_pct"] = 100 * (df["long"] - df["short"]) / df["oi"]
    return df
