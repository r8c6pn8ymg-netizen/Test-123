"""Termine der Woche: US-Datentermine (FRED-Kalender) und Zinsentscheide der 8 Notenbanken.

Quellen (kostenlos, ohne Key, ohne Prognosewerte):
- US-Daten: offizieller Veroeffentlichungskalender von FRED (Fed St. Louis), taeglich neu geladen.
  Zeiten dort in US Central Time, hier nach deutscher Zeit umgerechnet.
- Notenbank-Sitzungen: von den offiziellen Seiten der Notenbanken abgeschrieben (Stand 20.11.2026).
  Die RBNZ-Seite blockt automatische Abrufe; ihre Termine stammen aus der RBNZ-Mitteilung
  "OCR decision dates ... to Feb 2028" (ueber eine Websuche, bitte im Zweifel auf rbnz.govt.nz pruefen).

Konsensprognosen gibt es nicht kostenlos; die stehen weiter in TradingView.
"""
from __future__ import annotations

import html
import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from data_sources import CACHE, _download

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
      "2027-09-28", "2027-11-02", "2027-12-07"]),
    ("NZD", "RBNZ", time(14, 0), "Pacific/Auckland", "rbnz.govt.nz (über Websuche)",
     ["2026-12-09", "2027-02-10", "2027-03-17", "2027-05-05", "2027-06-16", "2027-08-04",
      "2027-09-15", "2027-10-27", "2027-12-08"]),
]

# FRED-Veroeffentlichungen, die fuer Waehrungen zaehlen: Name im FRED-Kalender -> (deutscher Name, Wichtigkeit 1–3)
US_RELEASES = {
    "Employment Situation": ("Arbeitsmarktbericht (Non-Farm Payrolls, Arbeitslosenquote)", 3),
    "Consumer Price Index": ("Verbraucherpreise (CPI)", 3),
    "Personal Income and Outlays": ("Einkommen & Konsum, PCE-Inflation", 3),
    "Gross Domestic Product": ("Bruttoinlandsprodukt (BIP)", 3),
    "Advance Monthly Sales for Retail and Food Services": ("Einzelhandelsumsätze", 2),
    "Producer Price Index": ("Erzeugerpreise (PPI)", 2),
    "Job Openings and Labor Turnover Survey": ("Offene Stellen (JOLTS)", 2),
    "G.17 Industrial Production and Capacity Utilization": ("Industrieproduktion", 2),
    "Unemployment Insurance Weekly Claims Report": ("Erstanträge Arbeitslosenhilfe", 2),
    "Surveys of Consumers": ("Verbraucherstimmung Uni Michigan", 2),
    "ADP National Employment Report": ("ADP-Arbeitsmarkt", 2),
    "Employment Cost Index": ("Arbeitskostenindex", 2),
    "U.S. International Trade in Goods and Services": ("Handelsbilanz", 2),
    "Manufacturer's Shipments, Inventories, and Orders (M3) Survey": ("Auftragseingänge Industrie", 2),
    "Advance Economic Indicators": ("Vorab: Warenhandel & Lagerbestände", 1),
    "New Residential Construction": ("Baubeginne & Baugenehmigungen", 1),
    "New Residential Sales": ("Neubauverkäufe", 1),
    "Existing Home Sales": ("Verkäufe bestehender Häuser", 1),
    "Productivity and Costs": ("Produktivität & Lohnstückkosten", 1),
    "Empire State Manufacturing Survey": ("Empire State Index (Industrie New York)", 1),
    "Manufacturing Business Outlook Survey": ("Philly-Fed-Index (Industrie)", 1),
    "Construction Spending": ("Bauausgaben", 1),
}
CENTRAL = ZoneInfo("America/Chicago")  # "All times are US Central Time" (FRED)

_ROW = re.compile(
    r'<span style="font-weight: bold;">([A-Z][a-z]+day [A-Z][a-z]+ \d{1,2}, \d{4})</span>'
    r'|<td nowrap[^>]*>\s*([^<]*?)\s*</td>\s*<td text-align="left">\s*<a href="/release\?rid=(\d+)">([^<]+)</a>')


def parse_fred_calendar(page: str) -> list[dict]:
    """Zeilen des FRED-Wochenkalenders: Datum, Uhrzeit (Central Time), Release-ID, Name."""
    rows, day, last_t = [], None, None
    for m in _ROW.finditer(page):
        if m.group(1):
            day, last_t = datetime.strptime(m.group(1), "%A %B %d, %Y").date(), None
            continue
        if day is None:
            continue
        t_raw = m.group(2).strip()
        if t_raw and t_raw != "N/A":
            last_t = datetime.strptime(t_raw.upper().replace(" ", ""), "%I:%M%p").time()
        elif t_raw == "N/A":
            last_t = None
        rows.append(dict(day=day, t=last_t, rid=int(m.group(3)), name=html.unescape(m.group(4)).strip()))
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
        path.write_text(raw)
    except Exception:
        if not path.exists():
            raise
        raw = path.read_text()
    return parse_fred_calendar(raw)


def _to_berlin(d: date, t: time | None, tz: str) -> tuple[str, str | None]:
    """Datum und Uhrzeit (HH:MM) in deutscher Zeit; ohne Uhrzeit bleibt das Ortsdatum."""
    if t is None:
        return d.isoformat(), None
    b = datetime.combine(d, t, ZoneInfo(tz)).astimezone(BERLIN)
    return b.date().isoformat(), b.strftime("%H:%M")


def build(today: date | None = None, days: int = 14, fred_rows: list[dict] | None = None) -> dict:
    today = today or datetime.now(BERLIN).date()
    start = today - timedelta(days=today.weekday())  # Montag dieser Woche
    end = start + timedelta(days=days - 1)
    events, warnings, next_cb = [], [], []
    for ccy, bank, t, tz, src, dates in CENTRAL_BANKS:
        ds = [date.fromisoformat(x) for x in dates]
        nxt = [d for d in ds if d >= today]
        if nxt:
            bd, bt = _to_berlin(nxt[0], t, tz)
            next_cb.append(dict(ccy=ccy, bank=bank, date=bd, time=bt, src=src,
                                days=(date.fromisoformat(bd) - today).days))
        if max(ds) < today + timedelta(days=60):
            warnings.append(f"Terminliste {bank} läuft bald aus – bitte auf {src} aktualisieren.")
        for d in ds:
            bd, bt = _to_berlin(d, t, tz)
            if start.isoformat() <= bd <= end.isoformat():
                events.append(dict(date=bd, time=bt, ccy=ccy, kind="cb", imp=3,
                                   title=f"{bank}: Zinsentscheid", src=src,
                                   note=None if t else "Uhrzeit nicht fest (meist mittags Tokio, ca. 04–05 Uhr deutscher Zeit)"))
    try:
        rows = fred_rows if fred_rows is not None else fred_calendar(start, end)
        seen = set()
        for r in rows:
            lab = US_RELEASES.get(r["name"])
            if not lab or (r["day"], r["name"]) in seen:
                continue
            seen.add((r["day"], r["name"]))
            bd, bt = _to_berlin(r["day"], r["t"], "America/Chicago")
            events.append(dict(date=bd, time=bt, ccy="USD", kind="data", imp=lab[1], title=lab[0],
                               src="FRED-Kalender", note=None))
    except Exception as e:
        warnings.append(f"FRED-Kalender nicht erreichbar ({type(e).__name__}); nur Notenbank-Termine.")
    events.sort(key=lambda e: (e["date"], e["time"] or "99:99", -e["imp"]))
    return dict(generated=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                week_start=start.isoformat(), week_end=end.isoformat(), today=today.isoformat(),
                list_as_of=LIST_AS_OF, events=events, next_cb=sorted(next_cb, key=lambda x: x["date"]),
                warnings=warnings)


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    t = build()
    (OUT / "termine.json").write_text(json.dumps(t, ensure_ascii=False))
    print(f"Termine: {len(t['events'])} Einträge {t['week_start']} bis {t['week_end']}")
    for w in t["warnings"]:
        print("  !", w)
    return t


if __name__ == "__main__":
    main()
