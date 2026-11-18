"""Erzeugt sentiment_fear_greed.pine mit fest eingebetteten AAII-, CNN- und BofA-Daten.

Aufruf:  python3 research/make_pine.py
Liest:   research/data/aaii.csv, cnn.csv, bofa_sent.csv, bofa_cash.csv
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "sentiment_fear_greed.pine")

AAII_START = pd.Timestamp("2000-01-03")   # Montag
CNN_START = pd.Timestamp("2020-07-13")    # Montag
BOFA_START = pd.Timestamp("2001-01-01")   # Monat
CHUNK = 3500                              # Zeichen pro Pine-String (sicher unter dem Limit)


def weekly(series, start):
    s = series.copy()
    s.index = s.index.to_period("W-SUN").start_time
    s = s.groupby(level=0).last()
    idx = pd.date_range(start, s.index.max(), freq="W-MON")
    return s.reindex(idx).ffill().bfill()


def monthly(series, start):
    s = series.copy()
    s.index = s.index.to_period("M").start_time
    s = s.groupby(level=0).last()
    idx = pd.date_range(start, s.index.max(), freq="MS")
    return s.reindex(idx).ffill().bfill()


def chunks(values):
    txt = ",".join(str(int(round(v))) for v in values)
    parts, cur = [], ""
    for tok in txt.split(","):
        if len(cur) + len(tok) + 1 > CHUNK:
            parts.append(cur)
            cur = tok
        else:
            cur = tok if not cur else cur + "," + tok
    parts.append(cur)
    return parts


def pine_arr(name, parts):
    lines = [f'    {name}.concat(f_parse("{p}"))' for p in parts]
    return "\n".join(lines)


def main():
    a = pd.read_csv(os.path.join(DATA, "aaii.csv"), parse_dates=["date"]).set_index("date")["spread"] * 100
    c = pd.read_csv(os.path.join(DATA, "cnn.csv"), index_col=0, parse_dates=True)["fear_and_greed_historical"].dropna()
    bs = pd.read_csv(os.path.join(DATA, "bofa_sent.csv"), index_col=0, parse_dates=True)["bofa_sent"] * 10
    bc = pd.read_csv(os.path.join(DATA, "bofa_cash.csv"), index_col=0, parse_dates=True)["bofa_cash"] * 10

    aw = weekly(a[a.index >= AAII_START - pd.Timedelta(days=7)], AAII_START)
    cw = weekly(c, CNN_START)
    bsm = monthly(bs[bs.index >= BOFA_START], BOFA_START)
    bcm = monthly(bc[bc.index >= BOFA_START], BOFA_START)

    stand = max(a.index.max(), c.index.max()).strftime("%d.%m.%Y")
    bofa_stand = bsm.index.max().strftime("%m/%Y")

    data_block = "\n".join([
        "if barstate.isfirst",
        pine_arr("aaiiArr", chunks(aw.values)),
        pine_arr("cnnArr", chunks(cw.values)),
        pine_arr("bofaArr", chunks(bsm.values)),
        pine_arr("cashArr", chunks(bcm.values)),
    ])

    src = TEMPLATE.replace("%%DATA%%", data_block).replace("%%STAND%%", stand).replace("%%BOFASTAND%%", bofa_stand)
    with open(OUT, "w") as f:
        f.write(src)
    print(f"geschrieben: {OUT}  ({len(src)} Zeichen)  AAII {len(aw)} Wochen, CNN {len(cw)} Wochen, BofA {len(bsm)} Monate, Stand {stand}")


TEMPLATE = r'''//@version=6
// ════════════════════════════════════════════════════════════════════════════
// Sentiment Fear & Greed v3
// Kombiniert: Kurs-Oszillator (RSI/Stochastik/Z-Score) + AAII-Umfrage + CNN Fear & Greed
// (+ optional BofA Fund Manager Survey). Datenstand: %%STAND%% (BofA: %%BOFASTAND%%)
// Backtest-Sieger (S&P 500, Nasdaq, Dow 2007–2026): Score = (2×Oszillator + AAII + CNN) / 4
// Kontra-Logik: GRÜN = extreme Angst (Kaufchance) · ROT = extreme Gier (Vorsicht)
// Optimiert für den WOCHENCHART. Daten werden wöchentlich per neuem Code aktualisiert.
// ════════════════════════════════════════════════════════════════════════════
indicator("Sentiment Fear & Greed v3", "Sentiment v3", overlay = true, max_labels_count = 50)

// ───────── Quellen an/aus ─────────
grpQ = "Quellen (an/aus)"
bool useOsc  = input.bool(true,  "Kurs-Oszillator (Gewicht 2)", group = grpQ, tooltip = "Im Backtest der stärkste Baustein bei Aktienindizes.")
bool useAAII = input.bool(true,  "AAII Bullen-minus-Bären (4-Wochen-Schnitt)", group = grpQ, tooltip = "Daten seit 2000. Funktionierte 1987–2006 sehr gut, seitdem schwächer, aber positiv.")
bool useCNN  = input.bool(true,  "CNN Fear & Greed", group = grpQ, tooltip = "Daten seit Juli 2020.")
bool useBofA = input.bool(false, "BofA Fund Manager Sentiment", group = grpQ, tooltip = "Im Backtest KEIN Kontra-Vorteil (eher Trendfolge) – standardmäßig aus, Werte stehen trotzdem in der Tabelle.")
bool autoMetal = input.bool(true, "Bei Gold/Silber/Dollar: AAII & CNN automatisch aus", group = grpQ, tooltip = "AAII und CNN messen US-Aktienstimmung. Für Metalle/Dollar ergab der Backtest keinen Vorteil.")
bool weeklyOsc = input.bool(true, "Oszillator immer auf Wochenbasis rechnen", group = grpQ, tooltip = "So entspricht der Indikator auch auf dem Tageschart dem Backtest.")

// ───────── Zonen ─────────
grpZ = "Zonen & Signale"
float extGreed = input.float(80, "Extreme Gier ab", group = grpZ)
float greedLvl = input.float(75, "Gier ab", group = grpZ)
float fearLvl  = input.float(25, "Angst bis", group = grpZ)
float extFear  = input.float(20, "Extreme Angst bis", group = grpZ)
string mode    = input.string("Kontra-Signal", "Farblogik", options = ["Kontra-Signal", "Stimmung direkt"], group = grpZ, tooltip = "Kontra: Angst = grün, Gier = rot.\nDirekt: Angst = rot, Gier = grün.")
bool showSig   = input.bool(true, "Marker, wenn Extremstimmung dreht", group = grpZ)

// ───────── Optik ─────────
grpA = "Anzeige"
color colRed   = input.color(color.red, "Rot", group = grpA)
color colGreen = input.color(color.green, "Grün", group = grpA)
int transp     = input.int(85, "Transparenz Zone", 0, 100, group = grpA)
int transpExt  = input.int(68, "Transparenz extrem", 0, 100, group = grpA)
bool showTable = input.bool(true, "Info-Tabelle", group = grpA)
string tPos    = input.string("Mitte rechts", "Tabellen-Position", options = ["Oben rechts", "Mitte rechts", "Unten rechts", "Oben links", "Unten links"], group = grpA)

// ───────── Eingebettete Daten ─────────
f_parse(string s) =>
    array<string> parts = str.split(s, ",")
    array<float> out = array.new_float()
    for p in parts
        out.push(str.tonumber(p))
    out

var array<float> aaiiArr = array.new_float()
var array<float> cnnArr  = array.new_float()
var array<float> bofaArr = array.new_float()
var array<float> cashArr = array.new_float()
%%DATA%%

const int WEEK = 7 * 86400000
int AAII_T0 = timestamp("UTC", 2000, 1, 3)
int CNN_T0  = timestamp("UTC", 2020, 7, 13)

// Wert ist ab Donnerstag der Woche bekannt → Index über (Bar-Ende − 3,5 Tage)
int tRef = na(time_close) ? time : time_close
f_wIdx(int t0) =>
    math.floor((tRef - 302400000 - t0) / WEEK)
// eine Woche Puffer: neue Umfrage erscheint erst donnerstags → letzten Wert weiterverwenden
f_get(array<float> arr, int i) =>
    int n = arr.size()
    i >= 0 and n > 0 and i <= n ? arr.get(math.min(i, n - 1)) : na

int ai = f_wIdx(AAII_T0)
float aaii  = f_get(aaiiArr, ai)
float aaii4 = na
int aj = math.min(ai, aaiiArr.size() - 1)
if aj >= 3 and ai <= aaiiArr.size()
    aaii4 := (aaiiArr.get(aj) + aaiiArr.get(aj - 1) + aaiiArr.get(aj - 2) + aaiiArr.get(aj - 3)) / 4
float cnn = f_get(cnnArr, f_wIdx(CNN_T0))

// BofA: Survey erscheint ca. Mitte des Monats → Monat von (Bar-Ende − 19 Tage)
int tB = tRef - 19 * 86400000
int bi = (year(tB, "UTC") - 2001) * 12 + month(tB, "UTC") - 1
float bofa = f_get(bofaArr, bi)
float cash = f_get(cashArr, bi)
bool dataEnded = ai > aaiiArr.size()

// ───────── Kurs-Oszillator ─────────
f_osc() =>
    float r  = ta.rsi(close, 9)
    float lo = ta.lowest(low, 13)
    float hi = ta.highest(high, 13)
    float st = hi > lo ? (close - lo) / (hi - lo) * 100 : 50
    float sd = ta.stdev(close, 20)
    float z  = sd > 0 ? math.max(0, math.min(100, 50 + 20 * (close - ta.sma(close, 20)) / sd)) : 50
    (r + st + z) / 3
float oscW = request.security(syminfo.tickerid, "1W", f_osc())
float oscC = f_osc()
float osc = weeklyOsc and timeframe.in_seconds() < timeframe.in_seconds("1W") ? oscW : oscC

// ───────── Asset-Erkennung (nur für Auto-Aus bei Metallen/Dollar) ─────────
string symRoot = str.upper(syminfo.root)
string symTick = str.upper(syminfo.ticker)
bool isMetalFx = symRoot == "GC" or symRoot == "SI" or symRoot == "MGC" or symRoot == "SIL" or symRoot == "DX" or str.contains(symTick, "XAU") or str.contains(symTick, "XAG") or str.contains(symTick, "DXY") or symTick == "GLD" or symTick == "SLV"
bool aOn = useAAII and not (autoMetal and isMetalFx)
bool cOn = useCNN and not (autoMetal and isMetalFx)

// ───────── Scores (0 = Angst, 100 = Gier) ─────────
float sOsc  = osc
float sAAII = na(aaii4) ? na : (math.max(-30, math.min(30, aaii4)) + 30) / 60 * 100
float sCNN  = cnn
float sBofA = na(bofa) ? na : bofa   // bereits ×10 gespeichert → 0–100

float num = 0.0
float den = 0.0
if useOsc and not na(sOsc)
    num += 2 * sOsc
    den += 2
if aOn and not na(sAAII) and not dataEnded
    num += sAAII
    den += 1
if cOn and not na(sCNN) and not dataEnded
    num += sCNN
    den += 1
if useBofA and not na(sBofA)
    num += sBofA
    den += 1
float score = den > 0 ? num / den : na

// ───────── Färbung ─────────
bool kontra = mode == "Kontra-Signal"
color cGreed = kontra ? colRed : colGreen
color cFear  = kontra ? colGreen : colRed
color bg = na
if not na(score)
    bg := score >= extGreed ? color.new(cGreed, transpExt) : score >= greedLvl ? color.new(cGreed, transp) : score <= extFear ? color.new(cFear, transpExt) : score <= fearLvl ? color.new(cFear, transp) : na
bgcolor(bg, title = "Sentiment-Hintergrund")

bool lowSig = ta.lowest(score, 4) <= extFear and ta.crossover(score, 30)
bool topSig = ta.highest(score, 4) >= extGreed and ta.crossunder(score, 70)
plotshape(showSig and lowSig, "Angst dreht (Kaufsignal)", shape.triangleup, location.belowbar, color.new(cFear, 0), size = size.small)
plotshape(showSig and topSig, "Gier kippt (Vorsicht)", shape.triangledown, location.abovebar, color.new(cGreed, 0), size = size.small)
plot(score, "Score", display = display.data_window)
plot(sOsc, "Oszillator", display = display.data_window)
plot(aaii, "AAII Spread", display = display.data_window)
plot(cnn, "CNN F&G", display = display.data_window)

alertcondition(ta.crossunder(score, extFear), "Extreme Angst", "Sentiment: EXTREME ANGST auf {{ticker}}")
alertcondition(ta.crossover(score, extGreed), "Extreme Gier", "Sentiment: EXTREME GIER auf {{ticker}}")
alertcondition(lowSig, "Angst dreht", "Sentiment: Angst dreht – Kaufsignal auf {{ticker}}")
alertcondition(topSig, "Gier kippt", "Sentiment: Gier kippt – Vorsicht auf {{ticker}}")

// ───────── Tabelle ─────────
f_lbl(float s) =>
    na(s) ? "n/a" : s >= extGreed ? "Extreme Gier" : s >= greedLvl ? "Gier" : s <= extFear ? "Extreme Angst" : s <= fearLvl ? "Angst" : "Neutral"
f_col(float s) =>
    na(s) ? color.new(color.gray, 85) : s >= greedLvl ? color.new(cGreed, 70) : s <= fearLvl ? color.new(cFear, 70) : color.new(color.gray, 85)
string pos = tPos == "Oben rechts" ? position.top_right : tPos == "Unten rechts" ? position.bottom_right : tPos == "Oben links" ? position.top_left : tPos == "Unten links" ? position.bottom_left : position.middle_right
var table tb = table.new(pos, 3, 9, bgcolor = color.new(color.gray, 85), border_width = 1)
f_row(int row, string name, string raw, float sc, bool on) =>
    tb.cell(0, row, name, text_size = size.small, text_color = on ? chart.fg_color : color.gray, text_halign = text.align_left)
    tb.cell(1, row, raw, text_size = size.small, text_color = on ? chart.fg_color : color.gray)
    tb.cell(2, row, na(sc) ? "–" : str.tostring(sc, "#"), text_size = size.small, text_color = chart.fg_color, bgcolor = on ? f_col(sc) : color.new(color.gray, 85))
if showTable and barstate.islast
    tb.cell(0, 0, "Sentiment v3", text_size = size.small, text_color = chart.fg_color, text_halign = text.align_left)
    tb.cell(1, 0, "Wert", text_size = size.small, text_color = chart.fg_color)
    tb.cell(2, 0, "Score", text_size = size.small, text_color = chart.fg_color)
    f_row(1, "GESAMT: " + f_lbl(score), "", score, true)
    f_row(2, "Oszillator", "", sOsc, useOsc)
    f_row(3, "AAII (4W)", na(aaii4) ? "–" : str.tostring(aaii4, "#.#"), sAAII, aOn)
    f_row(4, "CNN F&G", na(cnn) ? "–" : str.tostring(cnn, "#"), sCNN, cOn)
    f_row(5, "BofA Sentiment", na(bofa) ? "–" : str.tostring(bofa / 10, "#.#"), sBofA, useBofA)
    f_row(6, "BofA Cash %", na(cash) ? "–" : str.tostring(cash / 10, "#.0"), na, false)
    tb.cell(0, 7, "Datenstand %%STAND%%", text_size = size.tiny, text_color = color.gray, text_halign = text.align_left)
    tb.cell(0, 8, dataEnded ? "⚠ Daten veraltet – neuen Code einfügen" : (isMetalFx and autoMetal ? "Metall/Dollar: nur Oszillator" : ""), text_size = size.tiny, text_color = color.orange, text_halign = text.align_left)
'''

if __name__ == "__main__":
    main()
