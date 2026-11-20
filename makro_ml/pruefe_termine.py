"""Unabhaengige Gegenpruefung der Termine (Aufruf nach pipeline.py: ``python3 pruefe_termine.py``).

Laedt die offiziellen Seiten selbst, wertet sie mit eigenen Mustern aus (nicht mit den Parsern
aus termine.py) und vergleicht:
- jeden Notenbank-Termin in termine.CENTRAL_BANKS mit der Seite der Notenbank,
- jeden US-Termin in output/termine.json mit Census, BEA bzw. Fed (G.17), inklusive Uhrzeit,
- ob ein offizieller Termin im Fenster im Cockpit fehlt.
Exit-Code 0 = alles stimmt, 1 = Abweichung (Details in der Ausgabe).
"""
import html
import json
import re
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import termine  # nur fuer die Listen, die geprueft werden

TODAY = date.today()
Y0, Y1 = TODAY.year, TODAY.year + 1
URLS = {"ecb": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
        "boe": "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates",
        "boj": "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
        "snb": "https://www.snb.ch/en/services-events/digital-services/event-schedule",
        "boc": "https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/",
        "rba": "https://www.rba.gov.au/schedules-events/board-meeting-schedules.html",
        "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "bea": "https://www.bea.gov/news/schedule/full",
        "g17": "https://www.federalreserve.gov/releases/g17/release_dates.htm"}
for y in (Y0, Y1):
    URLS[f"census{y}"] = f"https://www.census.gov/economic-indicators/calendar-listview-{y}.html"
M = {m: i + 1 for i, m in enumerate("January February March April May June July August September "
                                    "October November December".split())}
MN = "|".join(M)
BOJ_M = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "July": 7, "Aug": 8,
         "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def page(name: str) -> str:
    req = urllib.request.Request(URLS[name], headers={"User-Agent": "Mozilla/5.0 (compatible; makro-ml-cockpit)"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    raw = re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)).replace("\xa0", " "))


def between(x: str, a: str, b: str) -> str:
    i = x.find(a)
    if i < 0:
        return ""
    j = x.find(b, i + len(a)) if b else -1
    return x[i:j] if j > 0 else x[i:i + 4000]


def official_cb() -> dict:
    off = {}
    x = page("ecb")
    off["EUR"] = {datetime.strptime(d, "%d/%m/%Y").date() for d in
                  re.findall(r"(\d\d/\d\d/\d{4}) Governing Council of the ECB: monetary policy meeting[^()]*\(Day 2\)", x)}
    x = page("boe")
    off["GBP"] = set()
    for y, end in ((Y0, f"{Y1} confirmed dates"), (Y1, "Monetary Policy Committee Reports")):
        off["GBP"] |= {date(y, M[m], int(d)) for d, m in re.findall(r"Thursday (\d{1,2}) (\w+)",
                                                                   between(x, f"{y} confirmed dates", end)) if m in M}
    x = page("boj")
    off["JPY"] = set()
    for y in (Y0, Y1):  # Sitzung "Mon. d1 (Tag), d2 (Tag)" -> Entscheid am 2. Tag
        seg = between(x, f"{y} Table", "Note:")
        off["JPY"] |= {date(y, BOJ_M[m], int(d2)) for m, d1, d2 in
                       re.findall(r"(Jan|Feb|Mar|Apr|May|June|July|Aug|Sept|Oct|Nov|Dec)\.? (\d{1,2}) \(\w+\.\), (\d{1,2}) \(", seg)}
    x = page("snb")
    off["CHF"] = {datetime.strptime(d, "%d.%m.%Y").date() for d in
                  re.findall(r"(\d\d\.\d\d\.\d{4}) \d\d:\d\d Monetary policy assessment of [^(]*\(press release\)", x)}
    x = page("boc")
    off["CAD"] = set()
    for y, end in ((Y0, f"Schedule for {Y1}"), (Y1, "See Blackout")):
        off["CAD"] |= {date(y, M[m], int(d)) for m, d in
                       re.findall(r"(\w+) (\d{1,2}) Interest rate announcement", between(x, f"Schedule for {y}", end)) if m in M}
    x = page("rba")
    off["AUD"] = set()
    for y, end in ((Y0, f"Board meeting schedules {Y1}"), (Y1, "Related Information")):
        off["AUD"] |= {date(y, M[m], int(d2)) for d1, d2, m in
                       re.findall(r"(\d{1,2})–(\d{1,2}) (\w+)", between(x, f"Board meeting schedules {y}", end)) if m in M}
    x = page("fomc")
    off["USD"] = set()
    for y, end in ((Y0, f"{Y0 - 1} FOMC Meetings"), (Y1, "Note:")):
        off["USD"] |= {date(y, M[m], int(d2)) for m, d1, d2 in
                       re.findall(rf"({MN}) (\d{{1,2}})-(\d{{1,2}})", between(x, f"{y} FOMC Meetings", end))}
    return off


def main() -> bool:
    ok = True
    off = official_cb()
    print(f"=== Notenbanken: Cockpit-Liste gegen offizielle Seite (ab {TODAY}) ===")
    for ccy, bank, t, tz, src, dates in termine.CENTRAL_BANKS:
        mine = [date.fromisoformat(d) for d in dates if d >= TODAY.isoformat()]
        if ccy == "NZD":
            print(f"{bank:16s} nicht prüfbar (Seite sperrt Abruf) – Liste: {dates}")
            continue
        o = sorted(d for d in off[ccy] if TODAY <= d <= max(mine or [TODAY]))
        good = bool(mine) and o == mine
        ok &= good
        print(f"{bank:16s} {'OK' if good else 'ABWEICHUNG'}  offiziell {len(o)} / Cockpit {len(mine)}"
              + ("" if good else f"\n   offiziell: {[str(d) for d in o]}\n   Cockpit:   {[str(d) for d in mine]}"))

    ev = json.load(open(HERE / "output" / "termine.json"))
    ws, we = ev["week_start"], ev["week_end"]
    cen = []
    for y in (Y0, Y1):
        try:
            x = page(f"census{y}")
        except Exception:
            continue  # Liste fuers naechste Jahr evtl. noch nicht veroeffentlicht
        for m in re.finditer(r"A(\d{8})(\d{4}) A\d{5,6}", x):
            name = re.split(r"A\d{12} A\d{5,6}", x[max(0, m.start() - 250):m.start()])[-1]
            cen.append((datetime.strptime(m.group(1), "%Y%m%d").date(), f"{m.group(2)[:2]}:{m.group(2)[2:]}", name))

    def near(mo, d):
        c = [date(y, M[mo], int(d)) for y in (Y0 - 1, Y0, Y1)]
        return min(c, key=lambda z: abs((z - TODAY).days))
    bea = [(near(mo, d), datetime.strptime(tm, "%I:%M %p").strftime("%H:%M"), nm) for mo, d, tm, nm in
           re.findall(rf"({MN}) (\d{{1,2}}) (\d{{1,2}}:\d\d [AP]M) [A-Z] ?(?:ews|rticle|ata)\s+(.+?)(?= View| (?:{MN}) \d| Quick)",
                      page("bea"))]
    g17 = [datetime.strptime(s, "%d-%B-%Y").date() for s in re.findall(r"\d{1,2}-[A-Z][a-z]+-\d{4}", page("g17"))]
    NY, BER = ZoneInfo("America/New_York"), ZoneInfo("Europe/Berlin")

    def ber(d, hhmm):
        h, mi = map(int, hhmm.split(":"))
        return datetime(d.year, d.month, d.day, h, mi, tzinfo=NY).astimezone(BER).strftime("%Y-%m-%d %H:%M")

    print(f"\n=== US-Termine im Cockpit ({ws} bis {we}) gegen offizielle Kalender ===")
    for e in [e for e in ev["events"] if e["kind"] == "data"]:
        r = next(r for r in termine.US_RELEASES.values() if r["label"] == e["title"])
        got = f"{e['date']} {e['time']}"
        if e["src"] == "census.gov":
            hits = [ber(d, tm) for d, tm, nm in cen if r["key"] in nm]
        elif e["src"] == "bea.gov":
            hits = [ber(d, tm) for d, tm, nm in bea if r["key"] in nm]
        elif e["src"] == "federalreserve.gov":
            hits = [ber(d, "09:15") for d in g17]
        else:
            print(f"{got}  {e['title'][:42]:42s} Quelle {e['src']} – nicht direkt prüfbar")
            continue
        good = got in hits
        ok &= good
        print(f"{got}  {e['title'][:42]:42s} {'OK, steht so bei ' + e['src'] if good else 'ABWEICHUNG'}")

    print("\n=== Fehlt im Cockpit etwas, das offiziell im Fenster steht? ===")
    have = {(e["date"], e["title"]) for e in ev["events"]}
    missing = 0
    for lst, src in ((cen, "census"), (bea, "bea")):
        for d, tm, nm in lst:
            for r in termine.US_RELEASES.values():
                if r["src"] == src and r["key"] in nm and not (src == "bea" and nm.startswith("GDP by")):
                    b = ber(d, tm)[:10]
                    if ws <= b <= we and (b, r["label"]) not in have:
                        missing += 1
                        print("FEHLT:", b, nm[:70])
    for d in g17:
        b = ber(d, "09:15")[:10]
        if ws <= b <= we and (b, "Industrieproduktion") not in have:
            missing += 1
            print("FEHLT:", b, "Industrieproduktion")
    ok &= missing == 0
    print("nichts" if missing == 0 else "")
    print("\nGESAMT:", "ALLES OK" if ok else "ES GIBT ABWEICHUNGEN")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
