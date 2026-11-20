# Vorab-Registrierung: 52-Wochen-Hoch und -Tief

Festgelegt am 20.11.2026, **bevor** Code geschrieben oder ein Ergebnis gesehen wurde.

## Frage
Laufen Währungen nach einem 52-Wochen-Hoch weiter hoch (Ausbruch, Trend) oder drehen sie (Rücklauf)?
Gleiches für das 52-Wochen-Tief. Und: Gilt das auch, wenn der Kurs nur **nahe** am Hoch/Tief ist (z. B. EUR 2 % darunter)?

## Daten
- Tageskurse Fed H.10 (FRED), 1971–heute, 7 Währungen gegen USD (EUR ab 1999) und der Dollar-Index (DXY).
- 52 Wochen = die 260 vorherigen Handelstage (nur Vergangenheit, der heutige Tag zählt nicht mit).

## Ereignisse (zwei, fest)
1. **Neues Hoch / neues Tief:** Schlusskurs über dem Höchst- bzw. unter dem Tiefstkurs der 260 Tage davor.
2. **Nahe am Hoch / Tief:** Schlusskurs höchstens 2 % unter dem Hoch bzw. über dem Tief (ohne neues Hoch/Tief).
- Ereignisse häufen sich (viele Tage hintereinander am Hoch). Gezählt wird ein Ereignis nur, wenn das letzte
  gezählte Ereignis derselben Art mindestens 20 Handelstage zurückliegt.

## Ergebnis-Maße (fest)
- Kursänderung nach **5, 20 und 65 Handelstagen** (1 Woche, 1 Monat, 3 Monate), in Richtung des Ereignisses
  (nach Hoch: steigt = Weiterlauf; nach Tief: fällt = Weiterlauf).
- Vergleich mit der **normalen** Kursänderung derselben Währung über denselben Abstand (alle Tage), damit
  ein allgemeiner Trend der Währung nicht als Effekt zählt.
- Zinsdifferenz wird nicht berücksichtigt (Kursbewegung ist die Frage).

## Tests
- Hauptergebnis: **alle 7 Währungen zusammen** (DXY getrennt), je Ereignis und Horizont: mittlere Über-Bewegung,
  Trefferquote, p-Wert (Block-Bootstrap über Kalenderjahre, 2.000 Ziehungen, zweiseitig).
- Mehrfachtests: 2 Ereignisse × 2 Richtungen × 3 Horizonte = 12 Haupttests → Holm-Korrektur.
- Stabilität: getrennt **1971–1999, 2000–2014, 2015–heute** und je Währung (nur zur Beschreibung).
- Placebo: gleiche Anzahl zufälliger Tage je Währung → Effekt muss nahe 0 sein.
- Timing-Kontrolle: Rechnet man das Hoch versehentlich inklusive der Zukunft, muss das sofort auffallen.

## Entscheidungsregel (fest)
Ein Ergebnis kommt als **Hinweis ins Cockpit**, wenn:
1. Holm-korrigiert p < 0,05 (alle Währungen zusammen), **und**
2. dieselbe Richtung (Weiterlauf oder Drehung) in **allen drei** Zeiträumen, **und**
3. in mindestens 5 von 7 Währungen dieselbe Richtung.
Sonst: als „getestet“ dokumentiert, nicht im Cockpit.

## Protokoll nach Sichtung der Ergebnisse
Lauf am 20.11.2026, Regeln unverändert. Einzige Änderung am Code vor dem Lauf: eine überflüssige, sofort
überschriebene Zeile in der p-Wert-Funktion entfernt (keine Auswirkung). Über-Bewegung in % in Ereignis-Richtung
(positiv = Weiterlauf):

| Test | n | Ø % | Treffer / normal | p (Holm) | 1971–99 / 2000–14 / 2015– | Währungen gleich | Regel |
|---|---|---|---|---|---|---|---|
| new_high_5 | 579 | +0.116 | 52.2 % / 49.8 % | 0.808 | +0.201 / +0.158 / -0.233 | 4/7 | nein |
| new_high_20 | 579 | +0.220 | 52.2 % / 50.0 % | 0.808 | +0.437 / +0.301 / -0.617 | 5/7 | nein |
| new_high_65 | 579 | +0.324 | 53.5 % / 50.6 % | 0.839 | +0.835 / +0.438 / -1.470 | 5/7 | nein |
| new_low_5 | 571 | +0.222 | 56.4 % / 50.3 % | 0.024 | +0.230 / +0.390 / +0.059 | 6/7 | **ja** |
| new_low_20 | 571 | +0.428 | 56.4 % / 50.2 % | 0.075 | +0.657 / +0.532 / -0.099 | 7/7 | nein |
| new_low_65 | 567 | +0.666 | 56.3 % / 50.0 % | 0.808 | +0.843 / +1.351 / -0.296 | 6/7 | nein |
| near_high_5 | 1363 | +0.028 | 50.8 % / 49.8 % | 0.839 | +0.057 / +0.069 / -0.109 | 4/7 | nein |
| near_high_20 | 1363 | +0.222 | 52.3 % / 49.9 % | 0.4275 | +0.434 / +0.229 / -0.314 | 7/7 | nein |
| near_high_65 | 1358 | +0.412 | 52.5 % / 50.3 % | 0.808 | +0.842 / +0.457 / -0.744 | 6/7 | nein |
| near_low_5 | 1281 | +0.059 | 51.8 % / 50.3 % | 0.808 | +0.103 / +0.042 / -0.007 | 5/7 | nein |
| near_low_20 | 1279 | +0.278 | 55.1 % / 50.2 % | 0.0385 | +0.517 / +0.077 / +0.003 | 7/7 | **ja** |
| near_low_65 | 1271 | +0.580 | 55.5 % / 50.2 % | 0.808 | +0.956 / +0.541 / -0.093 | 6/7 | nein |

**Entscheidung nach Regel:** `new_low_5` und `near_low_20` bestehen → als **Hinweis** ins Cockpit (Faktor-Reiter).
Ehrlich: `near_low_20` besteht nur formal (2015– Ø +0,003 %, praktisch null); `new_low_5` 2015– nur +0,06 %.
Für 52-Wochen-Hochs nichts belegt.

**Selbstprüfung:** Placebo (zufällige Tage) -0.069 %. Timing-Kontrolle mit absichtlichem Zukunftsblick:
Scheineffekt -3.79 % / -4.01 % (würde also sofort auffallen).
Unabhängige Nachrechnung (eigene Schleife): new_low_5 +0,222 % (n 571), near_low_20 +0,278 % (n 1.279) – identisch.

**Korrekturen nach Code-Prüfung (20.11., abends):** (1) Vergleichswert „an einem beliebigen Tag“ zählte Tage ohne
Kursänderung als „gefallen“; jetzt wie die Treffer gezählt (z. B. new_low_5: 48,7 statt 50,3 von 100). p-Werte und
Entscheidung unverändert. (2) Das Cockpit zeigte Zahlen je Währung auch für Hoch-Tests, die die Regel **nicht** bestanden
haben – entgegen dieser Registrierung. Jetzt stehen Zahlen nur bei bestandenen Tests, sonst „kein verlässliches Muster“.
(3) Status (neues Tief / nahe Tief) wird aus ungerundeten Werten bestimmt.
