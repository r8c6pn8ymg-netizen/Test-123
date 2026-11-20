# Makro ML Cockpit

FX-Bias (DXY, EUR, JPY, GBP, CHF, CAD, AUD, NZD) aus Makrodaten über 10–90 Tage.

```bash
python3 pipeline.py          # Daten laden (FRED, BIS, EZB, SIPRI), Modelle trainieren -> output/data.json
python3 pipeline.py --cache  # dasselbe ohne Neudownload
```

`cockpit.html` ist die Oberfläche; sie lädt `data.json` (aus `output/`) und wird als
Claude-Artifact veröffentlicht. Methodik: siehe Docstring in `pipeline.py` und Abschnitt
„Methodik“ im Cockpit.
