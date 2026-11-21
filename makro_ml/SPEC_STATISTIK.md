# SPEC – Reiter „Statistik & Strategie-Tester“ im Makro ML Cockpit

Stand: 21.11.2026. Grundlage: Prompt „Return Statistics“ (QuantCockpit_Prompt.md), angepasst an die Architektur
des Cockpits. Diese Datei ist verbindlich; jede Abweichung wird hier eingetragen (Abschnitt 9).

## 1. Architektur (angepasst)

Das Cockpit ist **eine** statische Seite (`cockpit.html`), veröffentlicht als Claude-Artifact. Es gibt keinen laufenden
Server. Deshalb statt FastAPI + React:

| Teil | Wo gerechnet | Datei |
|---|---|---|
| Daten laden, Wochen/Monate bilden, alle Kennzahlen, Tests, Histogramm, Q-Q, Walk-Forward-VaR | **Python**, vorab, täglich | `returns_stats.py` → `output/stats_index.json`, `output/stats/<id>.json` |
| Strategie-Tester (Regeln wählst du im Browser) | **Browser** (JavaScript), weil die Regeln frei wählbar sind | `strategie.js` |
| Unabhängige Gegenrechnung des Strategie-Testers | **Python** (eigener Code) | `strategie_ref.py` |
| Darstellung | Browser, nur Anzeige/Formatierung | neuer Reiter `#statistik` in `cockpit.html` |

- Der Statistik-Teil rechnet im Browser **nichts** (nur Formatierung). Alle Varianten (Intervall × Rendite-Typ × Zeitraum ×
  Bins) werden vorab berechnet; die Seite lädt nur die Datei des gewählten Instruments.
- Der Strategie-Tester muss im Browser rechnen. Absicherung: `strategie.js` wird in den Tests mit Node auf denselben
  Daten gegen `strategie_ref.py` verglichen (Toleranz 1e-9).
- Bestehende Reiter, `pipeline.py`, `factors.py`, `termine.py` bleiben unverändert. In `cockpit.html` kommen nur ein
  Reiter-Knopf, ein neuer Bereich und ein eigenes Skript dazu; `pruefe_cockpit.js` prüft den neuen Reiter mit.
- Farben, Schriften, Panels: die vorhandenen Cockpit-Tokens (`--bg`, `--panel`, `--bull`, `--bear`, `--warn`, `--accent`,
  JetBrains Mono). Rot = `--bear`, Grün = `--bull`, Kurven = `--warn` (Bernstein).
- Diagramme: eigenes SVG wie im restlichen Cockpit (keine neue Bibliothek).

## 2. Datenbasis

- **Instrumente** (`returns_stats.INSTRUMENTS`), auswählbar nach Gruppe:
  - Aktienindizes/ETFs: S&P 500 (^GSPC), SPY, QQQ, Nasdaq 100, Dow Jones, Russell 2000, DAX, Euro Stoxx 50, FTSE 100,
    CAC 40, SMI, Nikkei 225, Hang Seng, Emerging Markets (EEM), MSCI World (URTH)
  - Anleihen-ETFs: TLT (20J+), IEF (7–10J), SHY (1–3J), TIP, LQD, HYG, AGG
  - Rohstoffe: Gold, Silber, Kupfer, WTI, Brent, Erdgas, Weizen, Mais, Sojabohnen (Futures, Front-Monat) sowie GLD, SLV,
    USO, DBC (ETFs)
  - Währungen: EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, USD/CHF, USD/CAD (Fed H.10 über FRED wie im Cockpit,
    verlängert mit EZB-Referenzkursen, `data_sources.live_extend`), Dollar-Index (DX-Y.NYB)
  - Krypto: BTC-USD, ETH-USD
  - Volatilität: VIX (nicht direkt handelbar, nur Statistik; im Strategie-Tester mit Warnung)
- **Quelle:** yfinance mit `auto_adjust=True`, Spalte `Close` (Adjusted Close). Währungen: FRED wie oben.
  Cache `cache/yf_<id>.csv`, höchstens 20 h alt (wie `data_sources._cached`), offline letzter Stand.
- **Rohdaten immer täglich**; Wochen/Monate selbst:
  - Weekly = letzter Handelstag je Woche Sa–Fr (`W-FRI`), Datum = tatsächlicher letzter Handelstag (z. B. Donnerstag bei
    Feiertag am Freitag)
  - Monthly = letzter Handelstag je Kalendermonat
  - Daily = Handelstage wie geliefert
- **Zeitraum:** Max (Standard), 20Y, 10Y, 5Y, 1Y: Renditen mit Datum > (letztes Datum − Jahre). Kombinationen mit N < 30
  werden nicht berechnet („zu wenig Daten“).
- **Intervall:** 1d / 1wk / 1mo, Standard 1wk.
- **Datenqualität:** Duplikate entfernen (letzter Wert), sortieren, Zeitzone entfernen, Kurse ≤ 0 und NaN verwerfen.
  Erste Periode entfällt. Auffällige Tagesrenditen (|r| > 25 %, Krypto > 40 %) werden **nicht** gelöscht, aber als
  Datenqualitäts-Hinweis mit Datum angezeigt.
- **Laufende Periode („Live“):** Eine Periode ist live, wenn ihr Ende (Tag selbst / Freitag der Woche / Monatsende)
  ≥ heutigem Datum (Europe/Berlin) ist. Sie geht **nicht** in Historie und Kennzahlen ein. Ihre Rendite =
  letzter verfügbarer Kurs / Schluss der Vorperiode − 1 (bzw. ln bei Log).
- **Futures-Hinweis:** Yahoo-Futures (`=F`) sind Front-Monate ohne Roll-Bereinigung; Roll-Sprünge verfälschen
  Renditen (v. a. Öl, Gas). Das steht am Instrument. Für Backtests sind die ETFs ehrlicher (Roll-Kosten enthalten).
- **Indizes** (^GSPC usw.) sind Kursindizes ohne Dividenden; ETFs (SPY …) sind dividendenbereinigt.

## 3. Mathematische Definitionen (aus dem Prompt übernommen)

Notation: r = Vektor der abgeschlossenen Periodenrenditen, N = Anzahl. Intern Dezimalzahlen, Anzeige in Prozent.

1. **Rendite:** Simple r_t = P_t / P_{t−1} − 1 (Standard). Log r_t = ln(P_t / P_{t−1}) (umschaltbar). Alle Kennzahlen
   werden auf dem gewählten Rendite-Typ berechnet.
2. **Mean:** `np.mean(r)`.
3. **Median:** `np.median(r)`.
4. **Sigma:** `np.std(r, ddof=1)`. **Annualized Vol** = Sigma · √k; k: Daily 252 (Krypto 365), Weekly 52, Monthly 12.
5. **Skewness:** `scipy.stats.skew(r, bias=False)`.
6. **Excess Kurtosis:** `scipy.stats.kurtosis(r, fisher=True, bias=False)`.
7. **Best / Worst Period:** max(r) / min(r) mit Datum (Periodenende).
8. **Up / Down Periods:** Anzahl r > 0 bzw. r < 0 mit Anteil; r = 0 separat.
9. **Quantile** 1 %, 5 %, 95 %, 99 %: beobachtet `np.quantile(r, q, method="linear")`, normal μ + σ·Φ⁻¹(q).
10. **Value at Risk** (c ∈ {95 %, 99 %}, α = 1 − c), als **positive Verlustzahl**:
    historisch VaR = −Quantil_α(r); parametrisch VaR = −(μ + σ·Φ⁻¹(α)).
11. **Expected Shortfall** (positive Verlustzahl): historisch ES = −mean(r_i | r_i ≤ Quantil_α(r)) (inklusive Gleichheit,
    mind. 1 Beobachtung); parametrisch ES = −(μ − σ·φ(Φ⁻¹(α))/α).
12. **Tail-Events:** Anzahl/Anteil |r − μ| > kσ für k = 1, 2, 3 vs. Normal 2·(1 − Φ(k)).
13. **Tests** (Statistik, p, N):
    - Jarque-Bera: `scipy.stats.jarque_bera(r)`.
    - Shapiro-Wilk: `scipy.stats.shapiro(r)`, nur N ≤ 5000, sonst „n/a (N > 5000)“.
    - Anderson-Darling: A² = `scipy.stats.anderson(r, dist="norm", method="interpolate").statistic`;
      A*² = A²·(1 + 0.75/N + 2.25/N²); p nach D'Agostino/Stephens (1986):
      A*² ≥ 0.6: exp(1.2937 − 5.709A*² + 0.0186A*⁴); ≥ 0.34: exp(0.9177 − 4.279A*² − 1.38A*⁴);
      ≥ 0.2: 1 − exp(−8.318 + 42.796A*² − 59.938A*⁴); sonst 1 − exp(−13.436 + 101.14A*² − 223.73A*⁴).
      **Normal ja/nein:** A*² ≤ kritischer Wert nach Stephens (Fall: μ und σ geschätzt): 5 % → 0.752, 1 % → 1.035
      (siehe Abschnitt 9, Abweichung 2). Beide Niveaus werden geliefert, das UI zeigt das gewählte CL.
    - Chi²-Anpassung gegen N(μ, σ): Bins wie das gewählte Histogramm, äußere Bins nach ±∞ offen; benachbarte Bins von
      links und von rechts zusammenlegen, bis jede erwartete Häufigkeit ≥ 5; ndf = Bins − 1 − 2. Chi², ndf, Chi²/ndf, p.
      Bei ndf < 1: „n/a“.
    - t-Test Mean ≠ 0: `scipy.stats.ttest_1samp(r, 0)`, zweiseitig, df = N − 1.
14. **Histogramm-Bins:** Auto = Freedman-Diaconis (`np.histogram_bin_edges(r, bins="fd")`, bei IQR = 0 Sturges), fest 50. Binbreite h bleibt, Kanten werden auf Vielfache von h gelegt (0 ist immer eine Kante). Geliefert: Kanten,
    Häufigkeiten, Farbe (Bin rechts ≤ 0 → rot, links ≥ 0 → grün), erwartete Häufigkeit je Bin N·(Φ(b)−Φ(a)) und
    Normalkurve auf feinem Raster (120 Punkte) N·h·φ((x − μ)/σ)/σ.
15. **Aktuelle / Live-Periode** (r* = Live-Rendite; ohne Live-Periode die letzte abgeschlossene, die dann für den Vergleich
    aus der Historie genommen wird):
    - `worse_count` = Anzahl historischer r < r* (streng) → „Nur X Wochen schlechter … in den letzten Y Jahren“
    - `at_least_as_bad` = Anzahl r ≤ r* plus die aktuelle Periode selbst → „X von N+1 mindestens so tief“
    - `better_count` = r > r*, `at_least_as_good` = r ≥ r* plus selbst (für positives r*)
    - Perzentil = Anteil r < r* in %; z = (r* − μ)/σ; Normal-Erwartung = N·Φ(z) (bei r* > μ: N·(1 − Φ(z)))
    - Y = Jahre zwischen erster und letzter Periode, gerundet.
16. **Q-Q-Plot:** sortierte r gegen Φ⁻¹((i − 0.5)/N); höchstens 400 Punkte (alle 20 äußersten je Seite, dazwischen
    gleichmäßig); Referenzgerade μ + σ·z.

## 4. Walk-Forward-Analyse (VaR-Backtest, ohne Look-Ahead)

- Nur über die volle Historie („Max“). IS/OOS Standard: Daily 756/252, Weekly 156/52, Monthly 36/12. Modus rolling (Standard) und expanding (beide geliefert).
  Konfidenz 95 % und 99 %.
- Schritt k: IS = [s_k, e_k) mit e_k = IS + k·OOS, s_k = k·OOS (rolling) bzw. 0 (expanding); OOS = [e_k, min(e_k + OOS, N)).
  Ende, wenn e_k ≥ N. Unvollständiger letzter Block = `partial`.
- Aus IS geschätzt: μ, σ (ddof 1), hist./param. VaR und ES (Def. 3.10/3.11). Auf OOS: realisiertes μ, σ,
  Verletzungen r < −VaR (je Methode und Niveau), Trefferquote = Verletzungen/OOS-Länge, realisierter ES = −Mittel der
  Verletzungen (falls vorhanden).
- Gesamt je Methode/Niveau: Verletzungen x vs. erwartet α·T; **Kupiec POF**
  LR = −2·[(T−x)·ln(1−α) + x·ln α] + 2·[(T−x)·ln(1−x/T) + x·ln(x/T)] mit 0·ln 0 = 0, p = χ²₁-Survival.
  Stabilität von IS-μ und IS-σ: Mittel und Spannweite.
- Diagramme: OOS-Buy-and-Hold-Kurve (verkettet), rollierende Vola (Fenster = OOS-Länge, annualisiert) über die
  OOS-Renditen, VaR-Verletzungen (hist. 95/99) als Marker. Kurven auf höchstens 400 Punkte ausgedünnt, Marker vollständig.
- Look-Ahead-Garantie: kein Schätzer nutzt eine Beobachtung mit Index ≥ e_k (eigener Test).
- Hinweis im UI: Das ist ein Test der **Risikoschätzung**, kein Strategie-Test.

## 5. Strategie-Tester (Regeln wählst du selbst)

Grundsatz: Die Seite liefert nur das Werkzeug. Welche Regeln getestet werden, legst du fest. Die Seite zeigt zu jedem
Ergebnis die Prüfungen nach CLAUDE.md Regel 3 (Kosten, IS/OOS, Zufallsvergleich, Anzahl getesteter Varianten).

### 5.1 Daten und Zeit
- Gerechnet wird **immer auf Tagesschlüssen** des Instruments (abgeschlossene Tage). Eigene Daten: CSV „Datum;Kurs“
  einfügen (Datum JJJJ-MM-TT oder TT.MM.JJJJ, Dezimalkomma oder -punkt).
- **Signal-Intervall** P ∈ {Tag, Woche, Monat}: Bedingungen werden nur an abgeschlossenen Periodenenden geprüft
  (Liste aus Python bzw. bei eigenen Daten gleiche Regel in JS). Indikator-Längen n zählen in Perioden von P.
- **Ausführung:** Signal am Schlusskurs der Signalperiode → Ausführung `lag` Handelstage später zum Tagesschluss.
  Standard lag = 1 (nächster Tagesschluss, kein Look-Ahead). lag = 0 (gleicher Schluss, Market-on-Close) ist wählbar und
  wird als „optimistisch“ markiert.
- Zeitraum Start/Ende frei; IS/OOS-Trennung per Datum (Standard: 70 % des Zeitraums).

### 5.2 Einstieg (bis zu 3 Bedingungen, alle müssen gelten = UND)
x_j = Schlusskurs am Periodenende j, r_j = x_j/x_{j−1} − 1. Fehlen Daten für eine Bedingung → falsch.
| Typ | Definition |
|---|---|
| Kurs vs. SMA | x_j > SMA_n,j (oder <), SMA = einfacher Durchschnitt der letzten n x |
| SMA vs. SMA | SMA_a,j > SMA_b,j (oder <) |
| Momentum | x_j / x_{j−n} − 1 > θ (oder <) |
| Periodenrendite | r_j < θ (oder >), z. B. „nach Woche < −5 %“ |
| z-Score | (r_j − Mittel(r_{j−n}…r_{j−1})) / Std(r_{j−n}…r_{j−1}, ddof 1) < k (oder >); Std = 0 → falsch |
| RSI | RSI nach Wilder über n Perioden < θ (oder >): erster Durchschnitt = einfacher Mittelwert der ersten n Gewinne/Verluste, danach (alt·(n−1) + neu)/n; Ø-Verlust 0 → 100 |
| Nähe Hoch/Tief | x_j / max(x_{j−n+1}…x_j) − 1 ≥ −θ (nahe Hoch) bzw. x_j / min(…) − 1 ≤ θ (nahe Tief) |
| Volatilität | Std(r_{j−n+1}…r_j, ddof 1)·√k_P > θ (oder <) |
| Kalender | Monat ∈ Auswahl; bei Signal-Intervall Tag zusätzlich Wochentag ∈ Auswahl |
| Immer | immer wahr (Buy-and-Hold zum Vergleich) |

### 5.3 Richtung, Ausstieg, Größe, Kosten
- Richtung: Long, Short oder „Long, sonst Short“ (immer investiert).
- Ablauf je Handelstag t: (1) Tagesertrag mit der Position seit gestern, (2) fällige Aufträge ausführen (Kosten),
  (3) Stops prüfen (nicht am Einstiegstag), (4) an Periodenenden Signal prüfen: Ausstieg bei „Bedingung nicht mehr
  erfüllt“ oder Haltedauer = Anzahl Periodenenden seit dem Einstiegssignal ≥ n; Einstieg, wenn flach, kein Auftrag offen
  und am selben Tag kein Ausstieg. Bei lag = 0 wird sofort zum Tagesschluss ausgeführt. Am ersten Tag des Zeitraums
  (Bewertungsbeginn) wird nichts ausgeführt.
- Offene Position am Ende: zum letzten Kurs bewertet, ohne Ausstiegskosten (Grund „Ende“).
- Trade-Rendite = (1 − Einstiegskosten)·Π(1 + Größe·Pos·r_t − Finanzierung_t)·(1 − Ausstiegskosten) − 1.
- Ausstieg (was zuerst eintritt): Bedingung nicht mehr erfüllt (an Periodenenden), max. Haltedauer n Perioden,
  Stop-Loss x %, Take-Profit x %, Trailing-Stop x %. Stops werden an **Tagesschlüssen** geprüft (keine Intraday-Daten:
  echte Stops können schlechter gefüllt werden, Kurslücken sind enthalten). Neuer Einstieg frühestens am nächsten
  Periodenende nach dem Ausstieg.
- Positionsgröße: Anteil × Hebel des aktuellen Kapitals (tägliche Anpassung). Kapital ≤ 0 → Ruin: Tagesverlust auf −100 % begrenzt, danach keine Trades mehr.
- Kosten je Seite (Ein- oder Ausstieg): Gebühr (Bp) + Spread/2 (Bp) + Slippage (Bp), mal Positionsgröße, am
  Ausführungstag abgezogen. Finanzierung/Leihe % p. a. × Kalendertage/365 × Positionsgröße je gehaltenem Tag.
  Kein Zins auf freies Kapital (rf = 0).

### 5.4 Ergebnisse
- Tagesrendite Strategie g_t = Größe·Pos_{t−1}·(c_t/c_{t−1} − 1) − Kosten_t − Finanzierung_t; Equity = Π(1 + g_t).
- Kennzahlen (je Gesamt, IS, OOS, Buy-and-Hold): Gesamtrendite, CAGR (365,25 Kalendertage), Vola (√252, Krypto √365),
  Sharpe (rf = 0), Sortino (Downside-Abweichung √mean(min(g,0)²)), max. Drawdown und längste Drawdown-Dauer (Tage),
  Calmar, Anzahl Trades, Trefferquote, Ø Gewinn/Verlust, Profit-Faktor, Ø Haltedauer, Zeit im Markt, gezahlte Kosten,
  t-Test der mittleren Tagesrendite (zweiseitig).
- Jahrestabelle (Strategie vs. Buy-and-Hold), Trade-Liste, Equity-Kurve und Drawdown.
- **Zufallsvergleich (Placebo):** 200 Zufallsstrategien mit gleicher Anzahl Trades, gleichen Haltedauern, gleicher
  Richtung und gleichen Kosten, zufällige nicht überlappende Startzeitpunkte (fester Seed 42). p = Anteil Placebo-Sharpe
  ≥ Strategie-Sharpe.
- **Varianten-Zähler:** zählt die verschiedenen getesteten Einstellungen (nur im Browser). Zeigt den erwarteten besten
  Sharpe reiner Zufallsstrategien bei so vielen Versuchen (Bailey/López de Prado):
  E[max SR] ≈ √(1/Jahre)·((1 − γ)·Φ⁻¹(1 − 1/M) + γ·Φ⁻¹(1 − 1/(M·e))), γ = 0,5772.
- Klarer Hinweis im UI: technisch korrekter Backtest ≠ belastbare oder profitable Strategie.

## 6. Ausgabe-Dateien
- `output/stats_index.json`: Instrumentliste (id, name, Gruppe, Ticker, Quelle, Start/Ende, Krypto, handelbar, Hinweise,
  Datenqualität), Makro-Overlay (DGS10, VIXCLS, CPI YoY, monatlich), Stand.
- `output/stats/<id>.json`: `meta`, `series` (Tagesdaten, nur abgeschlossene Tage), `period_ends` (Indizes der
  abgeschlossenen Wochen-/Monatsenden), `live` je Intervall, `combos["<intervall>|<typ>|<zeitraum>"]` mit
  distribution, quantiles, var_es, tails, tests, live_rank, hist[auto|50], qq, wf[rolling|expanding] (nur bei „max“).
  `series`: `d0` = erstes Datum, `d` = Tage seit d0, `c` = Kurse. Zahlen auf 6 signifikante Stellen gerundet.
- JSON ohne NaN/Inf (`data_sources.to_json` → `null`).

## 7. Tests (pytest, offline)
- Fixture A (12 Renditen) mit allen Referenzwerten aus dem Prompt (Toleranzen wie dort).
- Live-Rang Screenshot-Plausibilität (Normal-Erwartung 1.924, Ann. Vol 16.98 %).
- Rang-Logik mit Gleichständen; parametrischer VaR/ES gegen geschlossene Formel; Resampling mit Feiertag und
  Live-Erkennung; Log = ln(1 + Simple); Walk-Forward Look-Ahead und Fensterindizes (rolling/expanding);
  Kupiec (250 Beob., α = 1 %, 5 Verletzungen; 0 Verletzungen); Histogramm (0 ist Kante, Summe = N);
  Chi² (erwartete Häufigkeit ≥ 5, ndf); A-D-Entscheidung; JSON-Schema vollständig ohne NaN/Inf.
- Strategie: Handrechnungen (Kosten, Finanzierung, Stop, lag 0/1), Python-Referenz gegen `strategie.js` (Node) auf
  mehreren Regel-Kombinationen, Look-Ahead (Änderung künftiger Kurse ändert frühere Signale nicht).
- Netzwerk nie in Tests (Fixtures statt yfinance).

## 8. Betrieb
- `python3 returns_stats.py` (aus `makro_ml/`) lädt/aktualisiert und schreibt die JSON-Dateien; `--cache` ohne Download.
- Veröffentlichen: `cockpit.html` mit `files` zusätzlich `strategie.js`, `stats_index.json` und `stats/<id>.json`.
- Tägliche Aktualisierung: erst nach deiner Freigabe in die geplante Aufgabe aufnehmen.

## 9. Abweichungen vom ursprünglichen Prompt (mit Begründung)
1. Kein FastAPI/React/Plotly: Cockpit ist eine statische Seite (Abschnitt 1). API-Tests → JSON-Schema-Tests.
2. Anderson-Darling: scipy ≥ 1.17 liefert keine kritischen Werte mehr und kappt den p-Wert auf 0,01–0,15. Entscheidung
   deshalb mit A*² gegen die Stephens-Tabelle (Fall 4: 5 % → 0.752, 1 % → 1.035), p-Wert nach D'Agostino/Stephens.
3. Live-Text „X von N+1“ statt „X of N“: die laufende Periode zählt mit, also ist der Nenner N + 1.
4. Normal-Erwartung bei positivem r*: N·(1 − Φ(z)) (Gegenstück für „mindestens so hoch“).
5. Strategie-Tester rechnet im Browser (freie Regeln ohne Server), abgesichert durch Python-Gegenrechnung.
6. CL im UI: 0.95/0.99; alle Werte sind für beide Niveaus vorberechnet.
7. Makro-Overlay ohne Normierung: Kurs und Makroreihe in zwei übereinanderliegenden Feldern mit gleicher Zeitachse.
