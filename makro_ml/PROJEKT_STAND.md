# Makro ML Cockpit – Projektstand und Übergabe

Stand: 20.11.2026. Diese Datei ist für neue Sitzungen gedacht: zuerst lesen, dann weiterarbeiten.

## Wo liegt was

- **Cockpit (privat, angepinnt):** https://claude.ai/artifact/PKmB9UQBGRjhWc9U3tDCH2
- **Repo / Branch:** `r8c6pn8ymg-netizen/test-123`, Branch `claude/charming-dijkstra-useq92`, Ordner `makro_ml/`
- **Tägliche Aktualisierung:** geplante Aufgabe „Makro ML Cockpit aktualisieren“, Mo–Fr 07:47 (Europe/Berlin).
  Sie läuft in der **ursprünglichen Sitzung** (`session_01Wa5ThyMmVEhU7BLeXRc8r6`, Aufgabe `trig_01WSt8ry3x8iwnqhqJ5fcffR`)
  und veröffentlicht unter derselben URL. Diese Sitzung nicht löschen oder archivieren, sonst stoppt das Update.

| Datei | Zweck |
|---|---|
| `pipeline.py` | Makro-ML-Modelle (FRED- und ALFRED-Variante), Schlusstest, Statistik-Tests, CFTC, Zinseinpreisung |
| `factors.py` | Faktor-Portfolio Carry/Value/Momentum seit 1977, alle 7 Faktor-Auswahlen, Robustheit |
| `consensus.py` | Konsens-Test (Makro + Faktor + CFTC, gegen USD und relativ) – nur Forschung, nicht im Cockpit |
| `research/turning_points.py` | Wendepunkt-Analyse mit TradingView-CSV-Exporten – nur Forschung |
| `termine.py` | Termine der Woche: US-Termine von Census, BEA, Fed (G.17) und FRED-Kalender (BLS), offizielle Uhrzeiten, ISM-PMI nach Regel, letzte Werte aus FRED; Zinsentscheide der 8 Notenbanken (Liste Stand 20.11.2026, bis Ende 2027) |
| `pruefe_termine.py` | Unabhängige Gegenprüfung: lädt die offiziellen Seiten selbst und vergleicht jeden Notenbank- und US-Termin (Exit-Code 1 bei Abweichung); läuft täglich nach der Pipeline |
| `data_sources.py` | Abruf FRED, ALFRED (ohne Key), BIS, EZB, SIPRI, CFTC; Cache in `cache/` |
| `cockpit.html` | Oberfläche; lädt `data_fred.json`, `data_alfred.json`, `factors.json`, `termine.json` (4. Reiter „Termine“) |
| `../tests/test_makro_ml.py` | Selbsttests (Lecks, Fristen, Euro ab 1999, Kosten, Phantom-Monat) |

Neu rechnen: `cd makro_ml && python3 -W ignore pipeline.py` (ca. 10 Min., Exit-Code 1 = ein Teil ist fehlgeschlagen und zeigt den alten Stand).
Danach `cockpit.html` veröffentlichen mit `files` = {`data_fred.json`, `data_alfred.json`, `factors.json`, `termine.json`} jeweils aus `output/`.
Tests: `python3 -m pytest tests/test_makro_ml.py -q` – erst **nach** einem Pipeline-Lauf, sonst werden sie mangels Cache übersprungen („skipped“ heißt nicht bestanden).

## Ergebnisse (nach allen Korrekturen)

- **Makro ML (Stand 20.11., ALFRED):** Ø Trefferquote aller Kandidaten ≈ 50 %. **Auswahl-Effekt:** Nach der Auswahl der besten
  erreichen echte Modelle Ø 57,1 %, reine Zufallsmodelle aber auch 54,9 %; im Schlusstest 2022+ bleiben 53,5 % (Zufall 49,5 %).
  Die DA in der Tabelle ist also geschönt, das steht jetzt im Cockpit. Signifikant im Schlusstest nur **AUD** (t-Test p ≈ 0,004,
  nach Bonferroni für 8 Währungen ≈ 0,03). Ensemble JPY nicht mehr signifikant (p ≈ 0,11, vorher 0,04). Nach FDR kein Einzelmodell.
- **ALFRED vs. FRED:** PAYEMS und Handelsbilanz sahen mit korrigierten Daten geschönt aus; Kernaussagen gleich. Standardansicht = ALFRED.
- **Faktor-Portfolio 1977–2026 (nach 5 Bp Kosten):** Kombi Sharpe 0,51 (p < 0,001), Carry 0,52, Value 0,18, Momentum 0,10.
  Seit 2015: Kombi −0,17, **Carry +0,23** (einziger positiver Faktor). Realistische Erwartung: Median aller Einstellungen, seit 2000 ≈ 0,19.
  Ab ~20 Bp Kosten ist der Vorteil seit 2000 weg. Walk-Forward-Umgewichtung („Adaptiv“) ist schlechter als fest.
- **CFTC:** als Gegensignal nicht belegt (1990–2014 Verlust), nur als Warnhinweis bei Extremen.
- **Konsens/Ampel:** bringt keinen belastbaren Zusatznutzen (wenige Trades, nach Mehrfachtest-Korrektur nicht signifikant) → bewusst **nicht** gebaut.
- **Ergänzt 20.11. (Abgleich Zinscapital-Bericht):** Leitzinsen im Faktor-Ranking tagesaktuell (BIS). Neue Makro-Kandidaten:
  Kern-PCE (Ø DA 51,5 %, ähnlich gut wie Kern-CPI und weitgehend dieselbe Information), realer Einzelhandel und Erzeugerpreise
  (≈ 49 %, kein Vorteil, werden vom Filter fast nie ausgewählt). Gesamtbild unverändert, nach FDR weiterhin kein Einzelmodell.
  PMIs und Konsensprognosen sind kostenpflichtig; TradingView-/Investing-Kalender haben keine offene Schnittstelle.
- **Gesamtprüfung 20.11.:** Unabhängige Nachrechnung Carry (eigener Code): Sharpe 0,51 vs. Cockpit 0,52; Newey-West mit statsmodels
  t = 3,51 vs. 3,52. Behoben: Absturz am Monatsletzten (ALFRED-Datenstand doppelt), laufender Monat zählte im Faktor-Backtest als voller
  Monat (jetzt 598 statt 599 Monate, Ergebnis unverändert), NaN in JSON, heutige US-Termine morgens als „fehlt“ markiert.
- **Prüfung 20.11. (Termine + Zinscapital-Abgleich):** FRED-Kalender hat teils Einstellzeiten und Doppel-Einträge
  (Industrieproduktion/Baubeginne 24./25.11.), deshalb offizielle Kalender. Marktwerte gegen PDF: US-Renditen, VIX, SOFR, HY-OAS
  identisch; Öl (er: Futures, FRED: Spot) und EUR/USD (andere Fixing-Uhrzeit) weichen leicht ab. Leitzinsen identisch.
- **Wendepunkte:** mit keinem Signal vorhersagbar (Vorhersage 49–52 %). Gute Treffer an Wenden (CFTC, umgedrehtes Momentum) sind Rückblick-Effekte.
  Optimieren auf einen Wendepunkt: 100 % dort, 51 % anderswo → Überoptimierung.

## Grundsätze, an die wir uns halten

1. Keine Daten vor ihrer Veröffentlichung (Fristen gegen ALFRED gemessen, Quartalswerte erst ab Quartalsende).
2. Regeln vorab festlegen, nicht nach Ergebnis auswählen; Zweier-Faktor-Kombinationen sind als „eigene Filterung“ markiert.
3. Jede Neuerung: Schlusstest, Zufalls-/Placebo-Vergleich, Mehrfachtest-Korrektur, Kosten, Selbstprüfung.
4. Ergebnisse ehrlich berichten, auch wenn sie schlechter werden.

## Offen / mögliche nächste Schritte

- **Notenbank-Termine** in `termine.py` sind abgeschrieben und reichen bis Ende 2027; die Pipeline warnt 60 Tage vorher. RBNZ-Seite blockt Abrufe (403).
- **Kalender mit Prognosen:** kostenlos und erlaubt gibt es keinen. Günstigste saubere Option: financecalendar.com Starter (9,99 $/Jahr, Prognosen 10 Länder, US-Konsens 12 Reihen). High Yield bewusst weggelassen (FRED nur 3 Jahre).

- Optionsdaten (Risk Reversals) und Konsensprognosen (Überraschungs-Indizes) wären die nächsten Datenquellen, sind aber kostenpflichtig.
- OECD-Revisionsdatenbank für Länderdaten vor 2018 (optional, geringer Nutzen erwartet).
