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
(wird nach dem Lauf ergänzt)
