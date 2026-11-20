# Vorab-Registrierung: Trendfolge und Carry mit Volatilitäts-Bremse

Festgelegt am 20.11.2026, **bevor** Code geschrieben oder ein Ergebnis gesehen wurde. Änderungen nach Sichtung
der Ergebnisse sind nur als Fehlerkorrektur erlaubt und werden unten mit Begründung protokolliert.

## Daten (wie im Faktor-Portfolio)
- 7 Währungen gegen USD (EUR erst ab 1999), Monatsende-Kurse, nur abgeschlossene Monate.
- Monatsertrag = Kursänderung + Zinsdifferenz zum USD / 12 (BIS-Leitzinsen), Kosten 5 Bp je umgeschichteter Einheit.
- Tageskurse (Fed H.10) nur für die Schwankungsschätzung. Zeitraum ab 1977.

## Strategie T: Trendfolge (Time-Series-Momentum, Moskowitz/Ooi/Pedersen 2012)
- Signal je Währung am Monatsende t: Vorzeichen des Überschussertrags der letzten 12 Monate (bis t).
- Positionsgröße: 10 % Ziel-Schwankung je Währung / geschätzte Schwankung. Schätzung: exponentiell gewichtete
  Standardabweichung der täglichen Kursänderungen, Schwerpunkt 60 Tage (wie im Paper), annualisiert mit √261.
- Portfolio = Durchschnitt über alle handelbaren Währungen. Keine weiteren Parameter.

## Strategie V: Carry mit Volatilitäts-Bremse (Moreira/Muir 2017, „volatility-managed portfolios“)
- Basis: das bestehende Carry-Portfolio (3 long / 3 short nach Zinsdifferenz, Gleichstände geteilt).
- Gewicht im Monat t+1 = c / (realisierte Varianz der täglichen Carry-Portfolio-Erträge im Monat t).
- c ist eine Konstante, so dass die Gesamt-Schwankung der des ungebremsten Carry entspricht (ändert die Sharpe-Ratio nicht).
- Keine Obergrenze für den Hebel (wie im Paper); der höchste Hebel wird berichtet.

## Tests (für T und V gleich)
1. Sharpe 1977–heute, vor 2000, seit 2000, Forward-Test **2022–heute** (bei keiner Entscheidung verwendet).
2. Newey-West-t-Test (3 Lags), Ertrag > 0.
3. T: Vergleich mit 2.000 Zufallsportfolios (zufälliges Vorzeichen je Währung und Monat, gleiche Positionsgrößen).
4. V: Alpha gegenüber ungebremstem Carry (Regression V = a + b·Carry, Newey-West), wie im Paper.
5. Kosten-Empfindlichkeit 5 / 10 / 20 Bp.

## Selbstprüfung
- **Timing-Test:** Signal mit Daten aus der Zukunft (t+1) muss auffällig besser sein; Signal einen Monat später darf
  nicht besser sein als das richtige. Sonst stimmt die Zeitzuordnung nicht.
- **Placebo:** Signale zufällig vertauscht → Sharpe nahe 0.
- **Unabhängige Nachrechnung** von T mit getrennt geschriebenem Code.

## Entscheidungsregel (fest)
Zwei Strategien werden getestet, deshalb Signifikanzschwelle **p < 0,025** (Bonferroni).
- **T kommt ins Cockpit**, wenn: Newey-West p < 0,025 über 1977–heute **und** Sharpe seit 2000 > 0 **und**
  Sharpe im Forward-Test 2022+ > 0.
- **V ersetzt Carry**, wenn: Alpha gegenüber Carry p < 0,025 **und** Sharpe seit 2000 höher als Carry **und**
  Sharpe im Forward-Test 2022+ nicht schlechter als Carry.
- **Kombination:** Nur Strategien, die bestehen, werden geprüft. Mischung zu gleichem Risiko mit der bestehenden
  Kombi (Risiko je Teil aus den letzten 60 Monaten, nur Vergangenheit). Aufgenommen wird sie, wenn sie
  1977–2021 eine höhere Sharpe hat als die Kombi allein **und** im Forward-Test 2022+ nicht schlechter ist.
- Besteht nichts: Ergebnis als „getestet und verworfen“ dokumentieren, Cockpit bleibt unverändert.

## Protokoll nach Sichtung der Ergebnisse
(wird nach dem Lauf ergänzt)
