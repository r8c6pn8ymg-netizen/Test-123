"""Termine der Woche: US-Datentermine und Zinsentscheide der 8 Notenbanken.

Quellen (kostenlos, ohne Key, ohne Prognosewerte):
- US-Daten, wo moeglich direkt von den Statistikaemtern: Census (Einzelhandel, Bau, Auftraege),
  BEA (BIP, PCE, Handelsbilanz), Fed (Industrieproduktion). Die BLS-Seite blockt automatische
  Abrufe; deren Termine (CPI, Arbeitsmarkt, PPI, JOLTS) sowie Erstantraege, Uni Michigan und
  regionale Fed-Umfragen kommen aus dem FRED-Kalender. Dessen Uhrzeiten sind teils die
  FRED-Einstellzeiten, deshalb werden ueberall die offiziellen Uhrzeiten (New York) verwendet.
- ISM-Einkaufsmanagerindizes: ISM veroeffentlicht Industrie am 1. und Dienstleistung am
  3. US-Geschaeftstag des Monats um 10:00 New York. Die Seite von ISM ist nicht erreichbar,
  die Termine werden nach dieser Regel berechnet und so gekennzeichnet.
- Notenbank-Sitzungen: von den offiziellen Seiten abgeschrieben (Stand 20.11.2026). Die RBNZ-Seite
  blockt automatische Abrufe; ihre Termine stammen aus der RBNZ-Mitteilung "OCR decision dates ...
  to Feb 2028" (ueber eine Websuche).
- Zuletzt veroeffentlichte Werte (Ist und Vorwert) aus FRED. Konsensprognosen gibt es nicht
  kostenlos; die stehen weiter in TradingView.
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from data_sources import CACHE, _download, fred, to_json

OUT = Path(__file__).parent / "output"
BERLIN = ZoneInfo("Europe/Berlin")
LIST_AS_OF = "2026-11-20"

# (Waehrung, Notenbank, Ortszeit der Bekanntgabe, Zeitzone, Quelle, Termine)
CENTRAL_BANKS = [
    ("USD", "Fed (FOMC)", time(14, 0), "America/New_York", "federalreserve.gov",
     ["2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28",
      "2027-09-15", "2027-10-27", "2027-11-17"]),
    ("EUR", "EZB", time(14, 15), "Europe/Berlin", "ecb.europa.eu",
     ["2026-12-17", "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-10", "2027-07-22",
      "2027-09-09", "2027-10-28", "2027-12-16"]),
    ("GBP", "Bank of England", time(12, 0), "Europe/London", "bankofengland.co.uk",
     ["2026-12-17", "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-17", "2027-07-29",
      "2027-09-16", "2027-11-04", "2027-12-16"]),
    ("JPY", "Bank of Japan", None, "Asia/Tokyo", "boj.or.jp",  # keine feste Uhrzeit, meist mittags Tokio
     ["2026-12-18", "2027-01-22", "2027-03-18", "2027-04-28", "2027-06-11", "2027-07-22",
      "2027-09-22", "2027-10-29", "2027-12-17"]),
    ("CHF", "SNB", time(9, 30), "Europe/Zurich", "snb.ch",
     ["2026-12-10", "2027-03-18", "2027-06-24", "2027-09-23", "2027-12-16"]),
    ("CAD", "Bank of Canada", time(9, 45), "America/Toronto", "bankofcanada.ca",
     ["2026-12-09", "2027-01-27", "2027-03-03", "2027-04-28", "2027-06-02", "2027-07-21",
      "2027-09-08", "2027-10-27", "2027-12-08"]),
    ("AUD", "RBA", time(14, 30), "Australia/Sydney", "rba.gov.au",  # Entscheid am 2. Sitzungstag
     ["2026-12-08", "2027-02-09", "2027-03-23", "2027-05-04", "2027-06-22", "2027-08-10",
      "2027-09-28", "2027-11-02", "2027-12-14"]),
    ("NZD", "RBNZ", time(14, 0), "Pacific/Auckland", "rbnz.govt.nz (über Websuche)",
     ["2026-12-09", "2027-02-10", "2027-03-17", "2027-05-05", "2027-06-16", "2027-08-04",
      "2027-09-15", "2027-10-27", "2027-12-08"]),
]

# Hinweise zu einzelnen Terminen (so auf der Quellseite vermerkt)
CB_NOTES = {
    ("USD", "2027-10-27"): "Die Fed nennt diese Sitzung „Monetary Policy Strategy Meeting“; für Dezember 2027 ist noch keine Sitzung eingetragen",
}

NY = "America/New_York"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MON_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

# US-Veroeffentlichungen: Schluessel -> Name, Wichtigkeit (1–3), offizielle Uhrzeit New York,
# Quelle des Datums und Suchtext dort, Werte aus FRED (Bezeichnung, Reihe, Umrechnung)
US_RELEASES = {
    "nfp": dict(label="Arbeitsmarktbericht (Non-Farm Payrolls, Arbeitslosenquote)", imp=3, et=time(8, 30),
                src="fred", key="Employment Situation",
                vals=[("neue Stellen", "PAYEMS", "diff_k"), ("Arbeitslosenquote", "UNRATE", "pct")]),
    "cpi": dict(label="Verbraucherpreise (CPI)", imp=3, et=time(8, 30), src="fred", key="Consumer Price Index",
                vals=[("Preise zum Vormonat", "CPIAUCSL", "mom"), ("Kern (ohne Energie, Lebensmittel) zum Vormonat", "CPILFESL", "mom")]),
    "pce": dict(label="Einkommen & Konsum, PCE-Inflation", imp=3, et=time(8, 30), src="bea",
                key="Personal Income and Outlays",
                vals=[("Kern-PCE zum Vormonat", "PCEPILFE", "mom"), ("Kern-PCE zum Vorjahr", "PCEPILFE", "yoy")]),
    "gdp": dict(label="Bruttoinlandsprodukt (BIP)", imp=3, et=time(8, 30), src="bea", key="GDP (",
                vals=[("Wachstum, aufs Jahr hochgerechnet", "A191RL1Q225SBEA", "pct")]),
    "ism_m": dict(label="ISM-Einkaufsmanagerindex Industrie (PMI)", imp=2, et=time(10, 0), src="ism", key=1, vals=[]),
    "ism_s": dict(label="ISM-Einkaufsmanagerindex Dienstleistung (PMI)", imp=2, et=time(10, 0), src="ism", key=3, vals=[]),
    "retail": dict(label="Einzelhandelsumsätze", imp=2, et=time(8, 30), src="census",
                   key="Advance Monthly Sales for Retail and Food Services", vals=[("zum Vormonat", "RSAFS", "mom")]),
    "ppi": dict(label="Erzeugerpreise (PPI)", imp=2, et=time(8, 30), src="fred", key="Producer Price Index",
                vals=[("Erzeugerpreise zum Vormonat", "PPIFIS", "mom")]),
    "jolts": dict(label="Offene Stellen (JOLTS)", imp=2, et=time(10, 0), src="fred",
                  key="Job Openings and Labor Turnover Survey", vals=[("offene Stellen", "JTSJOL", "mio")]),
    "ip": dict(label="Industrieproduktion", imp=2, et=time(9, 15), src="g17", key=None, vals=[("zum Vormonat", "INDPRO", "mom")]),
    "claims": dict(label="Erstanträge Arbeitslosenhilfe", imp=2, et=time(8, 30), src="fred",
                   key="Unemployment Insurance Weekly Claims Report", vals=[("Anträge pro Woche", "ICSA", "k_persons")]),
    "umich": dict(label="Verbraucherstimmung Uni Michigan", imp=2, et=time(10, 0), src="fred",
                  key="Surveys of Consumers", vals=[]),  # auf FRED nur mit Verzoegerung (Lizenz)
    "eci": dict(label="Arbeitskostenindex", imp=2, et=time(8, 30), src="fred", key="Employment Cost Index",
                vals=[("Arbeitskosten zum Vorquartal", "ECIALLCIV", "mom")]),
    "trade": dict(label="Handelsbilanz", imp=2, et=time(8, 30), src="bea",
                  key="U.S. International Trade in Goods and Services", vals=[("Saldo (minus = Defizit)", "BOPGSTB", "bn")]),
    "durable": dict(label="Auftragseingänge langlebige Güter", imp=2, et=time(8, 30), src="census",
                    key="Advance Report on Durable Goods", vals=[("Aufträge zum Vormonat", "DGORDER", "mom")]),
    "adv": dict(label="Vorab: Warenhandel & Lagerbestände", imp=1, et=time(8, 30), src="census",
                key="Advance Economic Indicators Report", vals=[]),
    "starts": dict(label="Baubeginne & Baugenehmigungen", imp=1, et=time(8, 30), src="census",
                   key="New Residential Construction", vals=[("Baubeginne aufs Jahr hochgerechnet", "HOUST", "k")]),
    "nhs": dict(label="Neubauverkäufe", imp=1, et=time(10, 0), src="census", key="New Residential Sales",
                vals=[("Verkäufe aufs Jahr hochgerechnet", "HSN1F", "k")]),
    "existing": dict(label="Verkäufe bestehender Häuser", imp=1, et=time(10, 0), src="fred", key="Existing Home Sales", vals=[]),
    "prod": dict(label="Produktivität & Lohnstückkosten", imp=1, et=time(8, 30), src="fred", key="Productivity and Costs",
                 vals=[("Produktivität, aufs Jahr hochgerechnet", "PRS85006092", "pct")]),
    "empire": dict(label="Empire-State-Index (Industrie New York)", imp=1, et=time(8, 30), src="fred",
                   key="Empire State Manufacturing Survey", vals=[("Indexwert (über 0 = Wachstum)", "GACDISA066MSFRBNY", "level")]),
    "philly": dict(label="Philly-Fed-Index (Industrie)", imp=1, et=time(8, 30), src="fred",
                   key="Manufacturing Business Outlook Survey", vals=[("Indexwert (über 0 = Wachstum)", "GACDFSA066MSFRBPHI", "level")]),
    "constr": dict(label="Bauausgaben", imp=1, et=time(10, 0), src="census",
                   key="Construction Spending", vals=[("zum Vormonat", "TTLCONS", "mom")]),
}

# Welcher Zeitraum mit dem Termin neu erscheint: Monate vor dem Veroeffentlichungsmonat
# (0 = gleicher Monat, z. B. Regional-Umfragen). Quartalszahlen: Vorquartal, Wochenzahlen: Vorwoche.
PERIOD_LAG = {"nfp": 1, "cpi": 1, "pce": 1, "retail": 1, "ppi": 1, "jolts": 2, "ip": 1, "trade": 2, "durable": 1,
              "starts": 1, "nhs": 1, "empire": 0, "philly": 0, "constr": 2}
# Warum zu einem Termin kein Wert gezeigt wird
NO_VALUE = {
    "umich": "Kein Wert: Die Uni Michigan gibt ihre Zahlen an FRED erst rund zwei Monate später frei.",
    "ism_m": "Kein Wert: Die ISM-Zahlen sind lizenziert und nicht frei abrufbar.",
    "ism_s": "Kein Wert: Die ISM-Zahlen sind lizenziert und nicht frei abrufbar.",
    "adv": "Kein Einzelwert: Vorab-Bericht mit mehreren Teilzahlen (Warenhandel, Lager).",
    "existing": "Kein Wert: Die Zahlen des Maklerverbands NAR sind auf FRED nur eingeschränkt verfügbar.",
}
PREV_WORD = {"M": "Vormonat", "Q": "Vorquartal", "W": "Vorwoche"}


def expected_period(key: str, release: date, freq: str) -> pd.Timestamp:
    """Ab diesem Datum (Index bei FRED) gilt der Wert als Ergebnis dieses Termins."""
    if freq == "W":  # Wochenzahl endet am Samstag vor dem Termin (Do. 19.11. -> Woche bis Sa. 14.11.)
        return pd.Timestamp(release - timedelta(days=(release.weekday() - 5) % 7 or 7))
    if freq == "Q":
        q0 = (release.month - 1) // 3 * 3 + 1
        return pd.Timestamp(date(release.year, q0, 1)) - pd.DateOffset(months=3)
    return pd.Timestamp(date(release.year, release.month, 1)) - pd.DateOffset(months=PERIOD_LAG.get(key, 1))


_ROW = re.compile(
    r'<span style="font-weight: bold;">([A-Z][a-z]+day [A-Z][a-z]+ \d{1,2}, \d{4})</span>'
    r'|<td nowrap[^>]*>\s*([^<]*?)\s*</td>\s*<td text-align="left">\s*<a href="/release\?rid=(\d+)">([^<]+)</a>')


def _text(raw: str) -> str:
    raw = re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.S)
    # geschuetzte Leerzeichen (&nbsp;) wie normale behandeln, sonst passen Datumsmuster nicht
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)).replace("\xa0", " "))


def _fetch(name: str, url: str, must: str) -> str:
    """Seite laden; bei Fehler letzter Stand aus dem Cache (bis 40 Tage alt)."""
    path = CACHE / f"termine_{name}.html"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; makro-ml-cockpit)"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
        if must not in raw:
            raise ValueError(f"{name}: unerwartete Seite")
        path.write_text(raw, encoding="utf-8")
        return raw
    except Exception:
        if path.exists() and (datetime.now().timestamp() - path.stat().st_mtime) < 40 * 86400:
            return path.read_text(encoding="utf-8")
        raise


def parse_fred_calendar(page: str) -> list[dict]:
    """Zeilen des FRED-Kalenders: Datum, Release-ID, Name (die Uhrzeit dort wird nicht verwendet)."""
    rows, day = [], None
    for m in _ROW.finditer(page):
        if m.group(1):
            day = datetime.strptime(m.group(1), "%A %B %d, %Y").date()
        elif day is not None:
            rows.append(dict(day=day, rid=int(m.group(3)), name=html.unescape(m.group(4)).strip()))
    return rows


def fred_calendar(start: date, end: date, max_pages: int = 20) -> list[dict]:
    """Alle Seiten des FRED-Kalenders (50 Eintraege je Seite) laden und zusammen auswerten."""
    path = CACHE / f"termine_fred_{start}_{end}.html"
    base = f"https://fred.stlouisfed.org/releases/calendar?vs={start}&ve={end}"
    try:
        pages = []
        for n in range(1, max_pages + 1):
            raw = _download(base + (f"&pageID={n}" if n > 1 else "")).decode("utf-8", "ignore")
            if "All times are US Central Time" not in raw:
                raise ValueError("FRED-Kalender: unerwartete Seite")
            pages.append(raw)
            if f"pageID={n + 1}" not in raw:
                break
        raw = "\n".join(pages)
        path.write_text(raw, encoding="utf-8")
    except Exception:
        if not path.exists():
            raise
        raw = path.read_text(encoding="utf-8")
    return parse_fred_calendar(raw)


def parse_census(page: str) -> list[tuple[date, str]]:
    """Census-Kalender: jeder Eintrag endet mit einem Code A + JJJJMMTThhmm (Datum, Uhrzeit New York)."""
    out = []
    x = _text(page)
    for m in re.finditer(r"A(\d{12}) A\d{5,6}", x):
        before = x[max(0, m.start() - 260):m.start()]
        before = re.split(r"A\d{12} A\d{5,6}", before)[-1]
        out.append((datetime.strptime(m.group(1)[:8], "%Y%m%d").date(), before.strip()))
    return out


def parse_bea(page: str, today: date) -> list[tuple[date, str]]:
    """BEA-Terminliste (Monat Tag Uhrzeit, ohne Jahr): Jahr so waehlen, dass das Datum nahe an heute liegt."""
    x = _text(page)
    out = []
    pat = re.compile(rf"({MONTHS}) (\d{{1,2}}) (\d{{1,2}}:\d\d [AP]M) [A-Z] ?(?:ews|rticle|ata)\s+(.+?)"
                     rf"(?= View| (?:{MONTHS}) \d{{1,2}} \d{{1,2}}:\d\d| Quick Links|$)")
    for m in pat.finditer(x):
        mon = datetime.strptime(m.group(1), "%B").month
        cands = [date(y, mon, int(m.group(2))) for y in (today.year - 1, today.year, today.year + 1)]
        d = min(cands, key=lambda c: abs((c - today).days))
        out.append((d, m.group(4).strip()))
    return out


def parse_g17(page: str) -> list[date]:
    return [datetime.strptime(f"{d}-{mo}-{y}", "%d-%B-%Y").date()
            for d, mo, y in re.findall(rf"(\d{{1,2}})-({MONTHS})-(\d{{4}})", _text(page))]


def ism_dates(start: date, end: date, nth: int) -> list[date]:
    """n-ter US-Geschaeftstag jedes Monats (ISM-Regel), Feiertage nach US-Bundeskalender."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = set(USFederalHolidayCalendar().holidays(start - timedelta(days=40), end + timedelta(days=40)).date)
    out, m = [], date(start.year, start.month, 1)
    while m <= end:
        d, k = m, 0
        while True:
            if d.weekday() < 5 and d not in hol:
                k += 1
                if k == nth:
                    break
            d += timedelta(days=1)
        if start <= d <= end:
            out.append(d)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def _to_berlin(d: date, t: time | None, tz: str) -> tuple[str, str | None]:
    """Datum und Uhrzeit (HH:MM) in deutscher Zeit; ohne Uhrzeit bleibt das Ortsdatum."""
    if t is None:
        return d.isoformat(), None
    b = datetime.combine(d, t, ZoneInfo(tz)).astimezone(BERLIN)
    return b.date().isoformat(), b.strftime("%H:%M")


def _de(v: float, d: int = 1, sign: bool = False) -> str:
    s = f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _period(idx: pd.Timestamp, freq: str) -> str:
    if freq == "W":
        return f"Woche bis {idx:%d.%m.}"
    if freq == "Q":
        return f"Q{(idx.month - 1) // 3 + 1} {idx.year}"
    return f"{MON_DE[idx.month - 1]} {idx.year}"


_FRESH: dict = {}
_STALE: set = set()


def _fred_fresh(sid: str) -> pd.Series:
    """FRED-Reihe einmal je Lauf frisch laden (die gestrigen Veroeffentlichungen sollen drin sein).
    Klappt der Download nicht, liefert der Cache den alten Stand; das wird vermerkt und angezeigt."""
    if sid not in _FRESH:
        t0 = datetime.now().timestamp()
        _FRESH[sid] = fred(sid, refresh=True)
        path = CACHE / f"fred_{sid}.csv"
        if not path.exists() or path.stat().st_mtime < t0 - 60:
            _STALE.add(sid)
    return _FRESH[sid]


def latest_values(spec: list, get=None, upto: pd.Timestamp | None = None) -> list[dict]:
    """Zuletzt veroeffentlichter Wert und Vorwert je Kennzahl (aus FRED); mit ``upto`` der Wert genau
    fuer diesen Zeitraum (damit ein frueherer Termin seine eigene Zahl zeigt, nicht die neueste)."""
    get = get or _fred_fresh
    out = []
    for name, sid, how in spec:
        s = get(sid).dropna()
        gap = (s.index[-1] - s.index[-2]).days
        freq = "W" if gap <= 8 else "Q" if gap > 80 else "M"
        if how == "mom":
            x = 100 * s.pct_change()
            f = lambda v: _de(v, 1, True) + " %"
        elif how == "yoy":
            x = 100 * s.pct_change(4 if freq == "Q" else 12)
            f = lambda v: _de(v, 1) + " %"
        elif how == "diff_k":
            x = s.diff()
            f = lambda v: _de(v, 0, True) + " Tsd."
        elif how == "pct":
            x = s
            f = lambda v: _de(v, 1) + " %"
        elif how == "k_persons":
            x = s / 1000
            f = lambda v: _de(v, 0) + " Tsd."
        elif how == "k":
            x = s
            f = lambda v: _de(v, 0) + " Tsd."
        elif how == "mio":
            x = s / 1000
            f = lambda v: _de(v, 2) + " Mio."
        elif how == "bn":
            x = s / 1000
            f = lambda v: _de(v, 1) + " Mrd. $"
        else:
            x = s
            f = lambda v: _de(v, 1)
        x = x.dropna()
        if upto is not None:
            x = x[x.index <= upto]
        out.append(dict(name=name, value=f(x.iloc[-1]), prev=f(x.iloc[-2]), period=_period(x.index[-1], freq),
                        prev_period=_period(x.index[-2], freq),
                        idx=x.index[-1].date().isoformat(), freq=freq, prev_word=PREV_WORD[freq]))
    return out


def build(today: date | None = None, days: int = 14, fred_rows: list[dict] | None = None,
          pages: dict | None = None, get=None) -> dict:
    """``fred_rows``/``pages``/``get`` nur fuer Tests (sonst werden die Quellen live geladen)."""
    if today is None:
        now = datetime.now(BERLIN)
        today = now.date()
    else:  # Tests: Zeitpunkt des taeglichen Laufs
        now = datetime.combine(today, time(7, 47), BERLIN)
    now_b = now.strftime("%Y-%m-%d %H:%M")
    start = today - timedelta(days=today.weekday())  # Montag dieser Woche
    end = start + timedelta(days=days - 1)
    events, warnings, next_cb = [], [], []
    for ccy, bank, t, tz, src, dates in CENTRAL_BANKS:
        ds = [date.fromisoformat(x) for x in dates]
        if max(ds) < today + timedelta(days=60):
            warnings.append(f"Terminliste {bank} läuft bald aus – bitte auf {src} aktualisieren.")
        nxt = [d for d in ds if d >= today][:2]
        if nxt:
            up = [dict(zip(("date", "time"), _to_berlin(d, t, tz))) for d in nxt]
            next_cb.append(dict(ccy=ccy, bank=bank, date=up[0]["date"], time=up[0]["time"], src=src, upcoming=up))
        for d in ds:
            bd, bt = _to_berlin(d, t, tz)
            if start.isoformat() <= bd <= end.isoformat():
                events.append(dict(date=bd, time=bt, ccy=ccy, kind="cb", imp=3,
                                   title=f"{bank}: Zinsentscheid", src=src,
                                   note=CB_NOTES.get((ccy, d.isoformat())) or (
                                       None if t else "Uhrzeit nicht fest (meist mittags Tokio, ca. 4–5 Uhr deutscher Zeit)")))

    # --- US-Daten: Datum je Quelle sammeln ---
    pages = pages if pages is not None else {}
    found: dict[str, list[tuple[date, str]]] = {k: [] for k in US_RELEASES}

    def official(name, loader):
        try:
            res = loader() if name not in pages else pages[name]
            if name not in pages and not res:
                raise ValueError("keine Termine erkannt (Seitenaufbau geändert?)")
            return res
        except Exception as e:
            warnings.append(f"{name}-Kalender nicht erreichbar ({type(e).__name__}); Termine dafür aus dem FRED-Kalender.")
            return None

    years = sorted({start.year, end.year})
    census_failed: list[int] = []
    def census_all():
        out, errs = [], []
        for y in years:  # je Jahr einzeln: fehlt die Liste fuers neue Jahr noch, bleibt das alte erhalten
            try:
                out += parse_census(_fetch(f"census_{y}",
                                           f"https://www.census.gov/economic-indicators/calendar-listview-{y}.html",
                                           "Economic Indicator"))
            except Exception as e:
                errs.append(e)
                census_failed.append(y)
        if not out and errs:
            raise errs[0]
        return out
    cen = official("Census", census_all)
    bea = official("BEA", lambda: parse_bea(_fetch("bea", "https://www.bea.gov/news/schedule/full", "Release Schedule"), today))
    g17 = official("Fed-G.17", lambda: parse_g17(_fetch(
        "g17", "https://www.federalreserve.gov/releases/g17/release_dates.htm", "Industrial Production")))
    try:
        frows = fred_rows if fred_rows is not None else fred_calendar(start, end)
    except Exception as e:
        frows = []
        warnings.append(f"FRED-Kalender nicht erreichbar ({type(e).__name__}); BLS-Termine fehlen.")
    if cen is not None and census_failed:
        warnings.append(f"Census-Kalender {', '.join(map(str, census_failed))} nicht erreichbar; "
                        "Termine dieses Jahres aus dem FRED-Kalender.")
    src_ok = {"census": cen is not None, "bea": bea is not None, "g17": g17 is not None}
    for k, r in US_RELEASES.items():
        if r["src"] == "census" and cen is not None:
            found[k] = [(d, "census.gov") for d, txt in cen if r["key"] in txt]
            if census_failed and _FRED_FALLBACK.get(k):  # fehlendes Jahr aus FRED ergaenzen
                found[k] += sorted({(x["day"], "FRED-Kalender") for x in frows
                                    if x["name"] == _FRED_FALLBACK[k] and x["day"].year in census_failed})
        elif r["src"] == "bea" and bea is not None:
            found[k] = [(d, "bea.gov") for d, txt in bea
                        if r["key"] in txt and not (k == "gdp" and txt.startswith("GDP by"))]
        elif r["src"] == "g17" and g17 is not None:
            found[k] = [(d, "federalreserve.gov") for d in g17]
        elif r["src"] == "ism":
            found[k] = [(d, "ISM-Regel") for d in ism_dates(start, end, r["key"])]
        else:  # FRED-Kalender (BLS, DOL, Uni Michigan, regionale Fed) oder Rueckfall
            key = r["key"] if r["src"] == "fred" else _FRED_FALLBACK.get(k)
            if key and (r["src"] == "fred" or not src_ok.get(r["src"], True)):
                found[k] = sorted({(x["day"], "FRED-Kalender") for x in frows if x["name"] == key})

    vals_cache: dict = {}
    for k, lst in found.items():
        r = US_RELEASES[k]
        for d, src in lst:
            bd, bt = _to_berlin(d, r["et"], NY)
            if not (start.isoformat() <= bd <= end.isoformat()):
                continue
            ev = dict(date=bd, time=bt, ccy="USD", kind="data", imp=r["imp"], title=r["label"], src=src,
                      note="Datum nach ISM-Regel berechnet (1. bzw. 3. US-Geschäftstag), nicht von ISM abgerufen"
                      if src == "ISM-Regel" else None)
            if r["vals"]:
                try:
                    if k not in vals_cache:
                        vals_cache[k] = latest_values(r["vals"], get)
                    v = vals_cache[k]
                    exp = expected_period(k, d, v[0]["freq"])
                    have_new = pd.Timestamp(v[0]["idx"]) >= exp
                    past = f"{bd} {bt or '23:59'}" <= now_b  # erst nach der Uhrzeit vorbei
                    if past and have_new and pd.Timestamp(v[0]["idx"]) > exp:
                        v = latest_values(r["vals"], get, upto=exp)  # Zahl genau dieses Termins
                    ev["vals"] = v
                    ev["vstate"] = ("result" if have_new else "missing") if past else (
                        "revision" if have_new else "before")  # BIP: zweite/dritte Schaetzung desselben Quartals
                    ev["vnext"] = _period(exp, v[0]["freq"])
                except Exception:
                    ev["vnote"] = "Wert konnte heute nicht von FRED geladen werden."
            elif k in NO_VALUE:
                ev["vnote"] = NO_VALUE[k]
            events.append(ev)
    if _STALE:
        warnings.append("FRED heute nicht erreichbar für " + ", ".join(sorted(_STALE))
                        + " – die Werte dazu sind vom letzten erfolgreichen Abruf.")
    events.sort(key=lambda e: (e["date"], e["time"] or "99:99", -e["imp"]))
    for c in next_cb:
        c["days"] = (date.fromisoformat(c["date"]) - today).days
    return dict(generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                week_start=start.isoformat(), week_end=end.isoformat(), today=today.isoformat(),
                list_as_of=LIST_AS_OF, events=events, next_cb=sorted(next_cb, key=lambda x: (x["date"], x["time"] or "")),
                warnings=warnings)


# Namen im FRED-Kalender, falls die offizielle Quelle (Census/BEA/Fed) ausfaellt
_FRED_FALLBACK = {
    "pce": "Personal Income and Outlays", "gdp": "Gross Domestic Product",
    "retail": "Advance Monthly Sales for Retail and Food Services",
    "ip": "G.17 Industrial Production and Capacity Utilization",
    "trade": "U.S. International Trade in Goods and Services",
    "durable": "Manufacturer's Shipments, Inventories, and Orders (M3) Survey",
    "adv": "Advance Economic Indicators", "starts": "New Residential Construction",
    "nhs": "New Residential Sales", "constr": "Construction Spending",
}


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    t = build()
    (OUT / "termine.json").write_text(to_json(t))
    print(f"Termine: {len(t['events'])} Einträge {t['week_start']} bis {t['week_end']}")
    for w in t["warnings"]:
        print("  !", w)
    return t


if __name__ == "__main__":
    main()
