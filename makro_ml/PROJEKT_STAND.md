# Makro ML Cockpit – Projektstand und Übergabe

Stand: 20.11.2026. Diese Datei ist für neue Sitzungen gedacht: zuerst lesen, dann weiterarbeiten.

## Wo liegt was

- **Cockpit (privat, angepinnt):** https://claude.ai/artifact/PKmB9UQBGRjhWc9U3tDCH2
- **Repo / Branch:** `r8c6pn8ymg-netizen/test-123`, Branch `claude/charming-dijkstra-useq92`, Ordner `makro_ml/`
- **Tägliche Aktualisierung:** geplante Aufgabe „Makro ML Cockpit aktualisieren“, Mo–Fr 07:47 (Europe/Berlin).
  Sie läuft in der **ursprünglichen Sitzung** und veröffentlicht unter derselben URL.
  Diese Sitzung nicht löschen, sonst stoppt das Update.

| Datei | Zweck |
|---|---|
| `pipeline.py` | Makro-ML-Modelle (FRED- und ALFRED-Variante), Schlusstest, Statistik-Tests, CFTC, Zinseinpreisung |
| `factors.py` | Faktor-Portfolio Carry/Value/Momentum seit 1977, alle 7 Faktor-Auswahlen, Robustheit |
| `consensus.py` | Konsens-Test (Makro + Faktor + CFTC, gegen USD und relativ) – nur Forschung, nicht im Cockpit |
| `research/turning_points.py` | Wendepunkt-Analyse mit TradingView-CSV-Exporten – nur Forschung |
| `data_sources.py` | Abruf FRED, ALFRED (ohne Key), BIS, EZB, SIPRI, CFTC; Cache in `cache/` |
| `cockpit.html` | Oberfläche; lädt `data_fred.json`, `data_alfred.json`, `factors.json` |
| `../tests/test_makro_ml.py` | Selbsttests (Lecks, Fristen, Euro ab 1999, Kosten, Phantom-Monat) |

Neu rechnen: `cd makro_ml && python3 -W ignore pipeline.py` (ca. 10 Min.), danach Cockpit mit den drei JSON-Dateien veröffentlichen.
Tests: `python3 -m pytest tests/test_makro_ml.py -q`

## Ergebnisse (nach allen Korrekturen)

- **Makro ML:** Ø Trefferquote aller Kandidaten ≈ 50 %. Im unberührten Schlusstest 2022+ nur **AUD** signifikant besser als Zufall
  (t-Test p ≈ 0,007), Ensemble-Strategie bei **JPY** signifikant (Newey-West p ≈ 0,04). Nach FDR-Korrektur bleibt kein Einzelmodell.
- **ALFRED vs. FRED:** PAYEMS und Handelsbilanz sahen mit korrigierten Daten geschönt aus; Kernaussagen gleich. Standardansicht = ALFRED.
- **Faktor-Portfolio 1977–2026 (nach 5 Bp Kosten):** Kombi Sharpe 0,51 (p < 0,001), Carry 0,52, Value 0,18, Momentum 0,10.
  Seit 2015: Kombi −0,17, **Carry +0,23** (einziger positiver Faktor). Realistische Erwartung: Median aller Einstellungen, seit 2000 ≈ 0,19.
  Ab ~20 Bp Kosten ist der Vorteil seit 2000 weg. Walk-Forward-Umgewichtung („Adaptiv“) ist schlechter als fest.
- **CFTC:** als Gegensignal nicht belegt (1990–2014 Verlust), nur als Warnhinweis bei Extremen.
- **Konsens/Ampel:** bringt keinen belastbaren Zusatznutzen (wenige Trades, nach Mehrfachtest-Korrektur nicht signifikant) → bewusst **nicht** gebaut.
- **Wendepunkte:** mit keinem Signal vorhersagbar (Vorhersage 49–52 %). Gute Treffer an Wenden (CFTC, umgedrehtes Momentum) sind Rückblick-Effekte.
  Optimieren auf einen Wendepunkt: 100 % dort, 51 % anderswo → Überoptimierung.

## Grundsätze, an die wir uns halten

1. Keine Daten vor ihrer Veröffentlichung (Fristen gegen ALFRED gemessen, Quartalswerte erst ab Quartalsende).
2. Regeln vorab festlegen, nicht nach Ergebnis auswählen; Zweier-Faktor-Kombinationen sind als „eigene Filterung“ markiert.
3. Jede Neuerung: Schlusstest, Zufalls-/Placebo-Vergleich, Mehrfachtest-Korrektur, Kosten, Selbstprüfung.
4. Ergebnisse ehrlich berichten, auch wenn sie schlechter werden.

## Offen / mögliche nächste Schritte

- Optionsdaten (Risk Reversals) und Konsensprognosen (Überraschungs-Indizes) wären die nächsten Datenquellen, sind aber kostenpflichtig.
- OECD-Revisionsdatenbank für Länderdaten vor 2018 (optional, geringer Nutzen erwartet).
