# Makro ML Cockpit

FX-Bias (DXY, EUR, JPY, GBP, CHF, CAD, AUD, NZD) aus Makrodaten über 10–300 Tage, dazu
Faktor-Portfolio (Carry, Value, Momentum), Termine der Woche und der Reiter **Statistik** (Rendite-Statistik,
Walk-Forward-VaR-Test und Strategie-Tester für 46 Instrumente). Ausführlicher Stand: `PROJEKT_STAND.md`,
Definitionen des Statistik-Reiters: `SPEC_STATISTIK.md`.

```bash
python3 pipeline.py          # alles neu laden und rechnen (ca. 10 Min.), Exit-Code 1, wenn ein Teil fehlschlägt
python3 pipeline.py --cache  # dasselbe ohne Neudownload
python3 termine.py           # nur die Termine neu
python3 returns_stats.py     # Reiter Statistik: Kurse laden (yfinance/FRED), alle Kennzahlen rechnen (ca. 2 Min.)
python3 returns_stats.py --cache  # dasselbe ohne Neudownload
```

Ergebnis in `output/`: `data_fred.json`, `data_alfred.json`, `factors.json`, `termine.json`, `research_52w.json`.
`cockpit.html` ist die Oberfläche und wird als Claude-Artifact veröffentlicht. Die fünf JSON-Dateien
liegen dort unter genau diesen Namen neben der Seite (`files`: `data_fred.json` → `output/data_fred.json` usw.).

Reiter **Statistik** zusätzlich: `strategie.js` (aus `makro_ml/`), `stats_index.json` und `stats/<id>.json`
(aus `output/`, 46 Dateien, zusammen ca. 50 MB). Tests: `python3 -m pytest tests/test_returns_stats.py tests/test_strategie.py -q`
(offline; der Strategie-Test vergleicht `strategie.js` per Node mit der Python-Gegenrechnung `strategie_ref.py`).
