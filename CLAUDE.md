# CLAUDE.md – Projektanweisungen

## VERBINDLICHE ARBEITS- UND QUALITÄTSREGELN FÜR DAS GESAMTE TRADING-PROJEKT

Diese Regeln gelten dauerhaft für alle zukünftigen Aufgaben, Änderungen, Analysen, Indikatoren, Backtests und Python-Skripte in diesem Repository. Du sollst eigenständig und kritisch arbeiten. Ich möchte nicht jeden einzelnen Prüfschritt vorgeben müssen.

### REGEL 1 – VOR JEDER ÄNDERUNG BESTAND ANALYSIEREN

* Untersuche zuerst den bestehenden Code, die Projektstruktur und die vorhandenen Funktionen.
* Verstehe, wie die einzelnen Module miteinander zusammenhängen.
* Prüfe, ob eine gewünschte Funktion bereits existiert.
* Verändere keine funktionierenden Bestandteile unnötig.
* Installiere keine bereits vorhandenen Bibliotheken erneut.
* Erstelle keine doppelten Funktionen, Tests oder Dateien.

### REGEL 2 – EIGENE ARBEIT KRITISCH KONTROLLIEREN

* Überprüfe jeden selbst geschriebenen oder geänderten Code eigenständig.
* Suche aktiv nach Syntaxfehlern, Logikfehlern, falschen Berechnungen, fehlerhaften Datenverarbeitungen und möglichen Randfällen.
* Verlasse dich niemals darauf, dass dein eigener Code automatisch richtig ist.
* Kontrolliere auch, ob Änderungen unbeabsichtigte Auswirkungen auf andere Projektbereiche haben.
* Führe eine unabhängige Gegenprüfung der wichtigsten Ergebnisse durch.

### REGEL 3 – QUANTITATIVE UND FINANZMATHEMATISCHE VALIDIERUNG

Bei allen Trading-Strategien, Backtests und Finanzmodellen müssen insbesondere folgende Punkte geprüft werden:

* Korrektheit sämtlicher mathematischer und statistischer Berechnungen.
* Look-ahead Bias und Survivorship Bias.
* Data Snooping und Overfitting.
* In-Sample- und Out-of-Sample-Performance.
* Sharpe Ratio, Drawdown, Volatilität und Renditeberechnung.
* Transaktionskosten, Spread, Slippage und Finanzierungskosten.
* Realistische Positionsgrößen und Risikobegrenzungen.
* Reproduzierbarkeit der Ergebnisse.

Unterscheide immer zwischen technisch korrektem Code und einer tatsächlich statistisch belastbaren oder profitablen Strategie.

### REGEL 4 – FEHLER SELBSTSTÄNDIG BEHEBEN

* Wenn du bei deinen Prüfungen einen Fehler findest, behebe ihn eigenständig, sofern die Korrektur eindeutig und sicher ist.
* Ändere dabei nicht unnötig die ursprüngliche Berechnungslogik oder Strategie.
* Wenn eine Änderung das fachliche Ergebnis, die Strategieparameter oder historische Daten verändern könnte, erkläre zuerst die Auswirkungen und hole meine Zustimmung ein.
* Behebe nicht nur den ersten gefundenen Fehler, sondern untersuche auch mögliche Folgefehler.

### REGEL 5 – NACH JEDER ÄNDERUNG ERNEUT VOLLSTÄNDIG TESTEN

* Führe die relevanten automatisierten Tests aus.
* Führe anschließend die vollständige verfügbare Testsuite aus.
* Verwende Ruff zur Codeprüfung.
* Prüfe, ob bestehende Funktionen weiterhin korrekt arbeiten.
* Vergleiche wichtige Ergebnisse vor und nach der Änderung.
* Wiederhole die Prüfung nach jeder Fehlerbehebung.
* Beende die Aufgabe erst, wenn alle Prüfungen erfolgreich sind oder verbleibende Probleme ausdrücklich dokumentiert wurden.

Erwartete Fehlschläge (xfail) sind von tatsächlichen Testfehlern zu unterscheiden.

### REGEL 6 – TRANSPARENTE ERGEBNISSE UND KEINE UNGEFRAGTEN ÄNDERUNGEN

* Berichte am Ende genau, was du untersucht, geändert und getestet hast.
* Nenne die Anzahl bestandener, fehlgeschlagener und erwarteter Tests.
* Liste alle noch offenen Fehler, Risiken und Einschränkungen auf.
* Behaupte niemals, dass etwas vollständig geprüft wurde, wenn tatsächlich nur Teilbereiche getestet wurden.
* Verändere keine historischen Daten, Reports, Strategieergebnisse oder produktiven Dateien ohne ausdrückliche Notwendigkeit.
* Erstelle, merge oder veröffentliche keine neuen Pull Requests ohne meine ausdrückliche Zustimmung.
* Committe und pushe Änderungen nur, wenn ich es ausdrücklich beauftrage.

## Praktische Prüfbefehle in diesem Repository

* Testsuite: `python3 -m pytest -q` (Ergebnis nach passed / failed / xfailed getrennt angeben)
* Codeprüfung: `ruff check .` (muss ohne Befund durchlaufen, Konfiguration in `ruff.toml`)
* Cockpit (`makro_ml/`): `node makro_ml/pruefe_cockpit.js` (Seitenprüfung aller Tabs, Desktop und Handy) und `cd makro_ml && python3 pruefe_termine.py` (Termine gegen offizielle Seiten)
