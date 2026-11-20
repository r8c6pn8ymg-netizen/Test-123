# Makro ML Cockpit

FX-Bias (DXY, EUR, JPY, GBP, CHF, CAD, AUD, NZD) aus Makrodaten über 10–300 Tage, dazu
Faktor-Portfolio (Carry, Value, Momentum) und Termine der Woche. Ausführlicher Stand: `PROJEKT_STAND.md`.

```bash
python3 pipeline.py          # alles neu laden und rechnen (ca. 10 Min.), Exit-Code 1, wenn ein Teil fehlschlägt
python3 pipeline.py --cache  # dasselbe ohne Neudownload
python3 termine.py           # nur die Termine neu
```

Ergebnis in `output/`: `data_fred.json`, `data_alfred.json`, `factors.json`, `termine.json`.
`cockpit.html` ist die Oberfläche und wird als Claude-Artifact veröffentlicht. Die vier JSON-Dateien
liegen dort unter genau diesen Namen neben der Seite (`files`: `data_fred.json` → `output/data_fred.json` usw.).
