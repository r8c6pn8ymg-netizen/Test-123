# Tests

Alles prüfen: `scripts/check.sh` (Ruff + pytest). Läuft auch automatisch auf GitHub bei jedem Push (`.github/workflows/tests.yml`).

| Datei | Marker | Was wird geprüft |
|---|---|---|
| `test_units.py` | – | Rechen-Bausteine mit kleinen Handbeispielen: Rundung (Menge, Tick wie Pine), Drawdown, SL/TP-Simulation, Kennzahlen, Trade-Vergleich |
| `test_data_integrity.py` | – | Kursdaten vollständig, sortiert, OHLC plausibel; TradingView-Exporte lesbar; gespeichertes Raster vollständig |
| `test_regression.py` | `regression` | Python-Backtests = TradingView, Trade für Trade (Friday Rush 197, Weekly 202, Weekly SL6/TP5 202); gespeicherte Ergebnisse werden reproduziert |
| `test_analysis_scripts.py` | `analysis` | Schnelltests der Analyse-Skripte im leeren Testordner: `quant_analysis.py` (alle 6 Grafiken + Zusammenfassung), `daily_stop_backtest.py` und `sma_filter_test.py` reproduzieren ihre gespeicherten Ergebnisse exakt; Monte-Carlo-, Robustheits- und SL/TP-Raster laufen verkleinert und müssen mit den gespeicherten Rastern übereinstimmen. Prüft zum Schluss, dass `data/` und `reports/` unverändert sind |
| `test_pine.py` | `pine` | Pine-Indikatoren: `//@version=6`, Klammern, und die am Original kalibrierten Standardwerte (Regime-Schwellen, COT-Regeln, Saison-Optionen) |

Einzelne Gruppe: `pytest -m regression`, `pytest -m pine`, `pytest -m analysis`. Ohne die ~20 s lange Grafik-Analyse: `pytest -m "not slow"`.

Schlägt ein Regressions- oder Pine-Test fehl, hat eine Änderung die Logik bzw. einen kalibrierten Wert verändert.
Ist das gewollt, Test und Referenzwert bewusst mit anpassen.
