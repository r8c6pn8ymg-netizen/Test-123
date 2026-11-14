"""
Quant-Analyse der Weekly-Monatsmitte-Strategie (11. rein, 27. raus, optional SL 6 / TP 5).

Erzeugt Grafiken in reports/ und eine Textzusammenfassung reports/quant_summary.txt:
  1. Equity-Kurven vs. Kaufen & Halten + Drawdown-Verlauf
  2. Walk-Forward-Analyse (rollierend: 5 Jahre optimieren, 1 Jahr handeln)
  3. Monte-Carlo-Fächer + Verteilung des maximalen Drawdowns
  4. Lineare Regression: Equity-Trend (CAGR, R²) und Alpha/Beta gegenüber dem S&P 500
  5. Zufalls-Timing-Test (Permutationstest): ist der 11./27.-Kalender besser als Zufall?
  6. Parameter-Landkarte: Einstiegstag x Ausstiegstag

  python quant_analysis.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import monthly_mid_weekly_backtest as weekly

OUT = "reports"
CAP = 10_000.0
RNG = np.random.default_rng(2026)

# Palette (validierte Standardpalette, Hellmodus)
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # Slot 1-3: blau, orange, aqua
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})

summary: list[str] = []


def log(s=""):
    print(s)
    summary.append(s)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)


def set_money_axis(ax):
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10000, 15000, 20000, 30000, 40000, 60000]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))


def maxdd(eq):
    pk = np.maximum.accumulate(eq, axis=-1)
    return ((pk - eq) / pk).max(axis=-1)


# ───────────────────────── Daten ─────────────────────────
df = weekly.load_weekly("data/CAPITALCOM_SPX500_1W.csv")
idx_of = {t: i for i, t in enumerate(df.time)}


def run(sl=0.0, tp=0.0, start="2010-01-01", end="2026-10-30", e=11, x=27):
    tr, curve = weekly.run_backtest(df, start=start, end=end, capital=CAP, sl_pct=sl, tp_pct=tp,
                                    entry_day=e, exit_day=x)
    return tr, curve


# ───────────────────── 1. Equity + Drawdown ─────────────────────
def chart_equity():
    tr0, c0 = run()
    tr1, c1 = run(6, 5)
    s = c0.index >= pd.Timestamp("2010-01-04", tz="UTC")
    c0, c1 = c0[s], c1[s]
    bh = CAP * df.set_index("week_start").close.reindex(c0.index) / df.set_index("week_start").close.reindex(c0.index).iloc[0]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    a.plot(bh.index, bh, color=C3, lw=1.6, label="S&P 500 kaufen & halten")
    a.plot(c0.index, c0, color=C2, label="Strategie ohne SL/TP")
    a.plot(c1.index, c1, color=C1, label="Strategie SL 6 % / TP 5 %")
    a.set_yscale("log"); a.set_ylabel("Kapital (USD, log)")
    set_money_axis(a)
    a.set_title("Kapitalverlauf 2010–2026 (Start 10.000 USD)")
    a.legend(loc="upper left")
    for c, col, lab in [(c0, C2, "ohne SL/TP"), (c1, C1, "SL 6 / TP 5")]:
        dd = (c / c.cummax() - 1) * 100
        b.plot(dd.index, dd, color=col, lw=1.4, label=lab)
    b.set_ylabel("Drawdown %"); b.set_title("Rückgang vom letzten Hoch (wöchentlich, inkl. offener Trades)")
    b.legend(loc="lower left")
    save(fig, "1_equity_drawdown.png")
    yrs = (c0.index[-1] - c0.index[0]).days / 365.25
    log("1) Equity 2010–2026 (wöchentlich bewertet):")
    for c, lab in [(c0, "ohne SL/TP"), (c1, "SL 6 / TP 5"), (bh, "Kaufen & Halten")]:
        cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1
        log(f"   {lab:16s} Endkapital {c.iloc[-1]:9.0f}  CAGR {cagr*100:5.2f} %  Max DD {maxdd(c.values)*100:5.1f} %")


# ───────────────────── 2. Walk-Forward ─────────────────────
def chart_walkforward():
    sls = [0.0] + list(np.arange(4.0, 8.01, 0.5))
    tps = [0.0] + list(np.arange(3.0, 7.01, 0.5))
    grid = {}
    for sl in sls:
        for tp in tps:
            tr, _ = run(sl, tp, start="2007-07-01")
            grid[(sl, tp)] = (np.array([t.entry_day.year for t in tr]), np.array([t.ret_pct / 100 for t in tr]))
    years = grid[(0.0, 0.0)][0]
    rows, oos = [], {"WF": [], "Basis": [], "SL6/TP5": []}
    for test_year in range(2012, 2027):
        train = (years >= test_year - 5) & (years <= test_year - 1)
        test = years == test_year
        best, best_score = None, -np.inf
        for k, (_, r) in grid.items():
            eq = np.concatenate([[1], np.cumprod(1 + r[train])])
            score = (eq[-1] - 1) / max(maxdd(eq), 0.02)          # Rendite / Drawdown im Training
            if score > best_score:
                best, best_score = k, score
        r_wf = grid[best][1][test]
        oos["WF"].append(r_wf); oos["Basis"].append(grid[(0.0, 0.0)][1][test]); oos["SL6/TP5"].append(grid[(6.0, 5.0)][1][test])
        rows.append((test_year, best, np.prod(1 + r_wf) - 1, np.prod(1 + grid[(0.0, 0.0)][1][test]) - 1,
                     np.prod(1 + grid[(6.0, 5.0)][1][test]) - 1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for key, col, lab in [("Basis", C2, "Basis ohne SL/TP"), ("SL6/TP5", C1, "fest SL 6 / TP 5"),
                          ("WF", C3, "Walk-Forward (jährlich neu optimiert)")]:
        eq = CAP * np.cumprod(1 + np.concatenate(oos[key]))
        a.plot(np.arange(1, len(eq) + 1), eq, color=col, label=lab)
    a.set_title("Walk-Forward: nur Out-of-Sample-Jahre 2012–2026")
    a.set_xlabel("Trade-Nr. (Out-of-Sample)"); a.set_ylabel("Kapital (USD)"); a.legend(loc="upper left")
    yrs = [r[0] for r in rows]
    x = np.arange(len(yrs)); w = 0.27
    b.bar(x - w, [r[3] * 100 for r in rows], w, color=C2, label="Basis")
    b.bar(x, [r[4] * 100 for r in rows], w, color=C1, label="SL 6 / TP 5")
    b.bar(x + w, [r[2] * 100 for r in rows], w, color=C3, label="Walk-Forward")
    b.axhline(0, color=AXIS, lw=1)
    b.set_xticks(x, [str(y)[2:] for y in yrs]); b.set_title("Rendite je Testjahr (%)"); b.legend(fontsize=8)
    save(fig, "2_walk_forward.png")
    log("\n2) Walk-Forward (5 Jahre Training -> 1 Jahr Test, Ziel: Rendite/Drawdown):")
    log("   Jahr  gewählt (SL/TP)   WF-Rendite  Basis   SL6/TP5")
    for y, k, rw, rb, rf in rows:
        log(f"   {y}  SL {k[0]:3.1f} / TP {k[1]:3.1f}   {rw*100:7.1f} %  {rb*100:6.1f} %  {rf*100:6.1f} %")
    for key in oos:
        r = np.concatenate(oos[key]); eq = np.concatenate([[1], np.cumprod(1 + r)])
        log(f"   Gesamt {key:8s}: {(eq[-1]-1)*100:6.1f} %  Max DD {maxdd(eq)*100:4.1f} %")


# ───────────────────── 3. Monte-Carlo ─────────────────────
def chart_montecarlo():
    r0 = np.array([t.ret_pct / 100 for t in run()[0]])
    r1 = np.array([t.ret_pct / 100 for t in run(6, 5)[0]])
    n, sims = len(r1), 10_000
    idx = RNG.integers(0, n, (sims, n))
    eq0 = CAP * np.concatenate([np.ones((sims, 1)), np.cumprod(1 + r0[idx], axis=1)], axis=1)
    eq1 = CAP * np.concatenate([np.ones((sims, 1)), np.cumprod(1 + r1[idx], axis=1)], axis=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(n + 1)
    p = np.percentile(eq1, [5, 25, 50, 75, 95], axis=0)
    a.fill_between(x, p[0], p[4], color=C1, alpha=0.15, lw=0, label="5–95 % der Simulationen")
    a.fill_between(x, p[1], p[3], color=C1, alpha=0.30, lw=0, label="25–75 %")
    a.plot(x, p[2], color=C1, label="Median")
    a.plot(x, CAP * np.concatenate([[1], np.cumprod(1 + r1)]), color=INK, lw=1.4, ls="--", label="tatsächlicher Verlauf")
    a.set_title("Monte-Carlo: 10.000 zufällige Trade-Reihenfolgen (SL 6 / TP 5)")
    a.set_xlabel("Trade-Nr."); a.set_ylabel("Kapital (USD)"); a.legend(loc="upper left", fontsize=8)
    d0, d1 = maxdd(eq0) * 100, maxdd(eq1) * 100
    bins = np.arange(5, 70, 2.5)
    b.hist(d0, bins=bins, color=C2, alpha=0.55, label="ohne SL/TP", edgecolor=SURFACE, linewidth=1)
    b.hist(d1, bins=bins, color=C1, alpha=0.55, label="SL 6 / TP 5", edgecolor=SURFACE, linewidth=1)
    b.set_title("Verteilung des maximalen Drawdowns"); b.set_xlabel("Max. Drawdown %"); b.set_ylabel("Anzahl Simulationen")
    b.legend()
    save(fig, "3_monte_carlo.png")
    log("\n3) Monte-Carlo (10.000 Bootstrap-Reihenfolgen, 202 Trades):")
    for lab, eq, d in [("ohne SL/TP", eq0, d0), ("SL 6 / TP 5", eq1, d1)]:
        f = eq[:, -1]
        log(f"   {lab:12s} Endkapital P5 {np.percentile(f,5):7.0f}  Median {np.median(f):7.0f}  P95 {np.percentile(f,95):7.0f} | "
            f"Max DD Median {np.median(d):4.1f} %  P95 {np.percentile(d,95):4.1f} % | P(Verlust) {(f<CAP).mean()*100:4.1f} %")


# ───────────────────── 4. Regression ─────────────────────
def chart_regression():
    tr1, c1 = run(6, 5)
    eq = pd.Series(CAP + np.cumsum([t.pnl for t in tr1]), index=pd.to_datetime([t.exit_day for t in tr1]))
    t_years = np.asarray((eq.index - eq.index[0]).days / 365.25)
    y = np.log(eq.values)
    slope, icpt = np.polyfit(t_years, y, 1)
    fit = icpt + slope * t_years
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # Alpha/Beta: wöchentliche Strategie-Rendite vs. S&P 500 (2010–2026)
    s = c1.index >= pd.Timestamp("2010-01-04", tz="UTC")
    strat = c1[s].pct_change().dropna()
    mkt = df.set_index("week_start").close.reindex(c1[s].index).pct_change().dropna()
    beta, alpha = np.polyfit(mkt.values, strat.values, 1)
    rb = np.corrcoef(mkt.values, strat.values)[0, 1] ** 2
    inmkt = strat != 0

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    a.plot(eq.index, eq.values, color=C1, label="Kapital nach jedem Trade")
    a.plot(eq.index, np.exp(fit), color=INK2, lw=1.4, ls="--", label=f"Regressionslinie: {np.expm1(slope)*100:.1f} %/Jahr, R² = {r2:.2f}")
    a.set_yscale("log"); a.set_title("Lineare Regression des (log.) Kapitalverlaufs")
    set_money_axis(a)
    a.legend(loc="upper left", fontsize=9)
    b.scatter(mkt[inmkt] * 100, strat[inmkt] * 100, s=14, color=C1, alpha=0.6, edgecolor=SURFACE, linewidth=0.5, label="Wochen im Markt")
    b.scatter(mkt[~inmkt] * 100, strat[~inmkt] * 100, s=10, color=AXIS, alpha=0.6, label="Wochen ohne Position")
    xs = np.linspace(mkt.min(), mkt.max(), 10)
    b.plot(xs * 100, (alpha + beta * xs) * 100, color=INK2, lw=1.4, ls="--", label=f"Beta {beta:.2f}, Alpha {alpha*52*100:.2f} %/Jahr")
    b.set_title("Strategie vs. S&P 500 (Wochenrenditen)"); b.set_xlabel("S&P 500 Woche %"); b.set_ylabel("Strategie Woche %")
    b.legend(loc="upper left", fontsize=8)
    save(fig, "4_regression.png")
    log("\n4) Regression (SL 6 / TP 5):")
    log(f"   Trend des Kapitals: {np.expm1(slope)*100:.2f} %/Jahr, R² = {r2:.3f} (1 = perfekt gleichmäßig)")
    log(f"   Gegen S&P 500: Beta {beta:.2f}, Alpha {alpha*52*100:.2f} %/Jahr, R² {rb:.2f}, Zeit im Markt {inmkt.mean()*100:.0f} %")


# ───────────────────── 5. Zufalls-Timing-Test ─────────────────────
def chart_permutation():
    tr0, _ = run()
    real = np.prod([1 + t.ret_pct / 100 for t in tr0]) - 1
    # Für jeden echten Trade: gleiche Haltedauer (Wochen), aber zufällige Startwoche irgendwo im Datensatz 2010–2026
    closes = df.close.values
    valid = np.where((df.week_start >= "2010-01-01") & (df.week_start <= "2026-10-30"))[0]
    holds = np.array([t.bars for t in tr0])
    sims = 10_000
    starts = RNG.choice(valid[:-4], size=(sims, len(holds)))
    rets = closes[starts + holds] / closes[starts] - 1
    dist = np.prod(1 + rets, axis=1) - 1
    p = (dist >= real).mean()
    fig, a = plt.subplots(figsize=(9, 4.6))
    a.hist(dist * 100, bins=60, color=C1, alpha=0.75, edgecolor=SURFACE, linewidth=1, label="Zufälliger Einstieg, gleiche Haltedauer")
    a.axvline(real * 100, color=RED, lw=2, label=f"Strategie 11./27.: {real*100:.0f} %  (besser als {100-p*100:.0f} % der Zufallsläufe)")
    a.set_title("Zufalls-Timing-Test: ist der Kalender besser als Zufall?")
    a.set_xlabel("Gesamtrendite 2010–2026 (%)"); a.set_ylabel("Anzahl Simulationen"); a.legend(loc="upper right", fontsize=9)
    save(fig, "5_zufalls_timing.png")
    log("\n5) Zufalls-Timing-Test (ohne SL/TP, 10.000 Läufe mit gleichen Haltedauern, zufälligen Startwochen):")
    log(f"   Strategie {real*100:.1f} %, Zufall Median {np.median(dist)*100:.1f} %, p-Wert {p:.3f}")


# ───────────────────── 6. Parameter-Landkarte ─────────────────────
def chart_heatmap():
    entries, exits = list(range(6, 17)), list(range(20, 29))
    M = np.full((len(entries), len(exits)), np.nan)
    for i, e in enumerate(entries):
        for j, x in enumerate(exits):
            if x - e < 7:
                continue
            tr, _ = run(e=e, x=x)
            M[i, j] = (np.prod([1 + t.ret_pct / 100 for t in tr]) - 1) * 100
    lim = np.nanmax(np.abs(M))
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", C1])
    fig, a = plt.subplots(figsize=(9, 6))
    im = a.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    a.set_xticks(range(len(exits)), exits); a.set_yticks(range(len(entries)), entries)
    a.set_xlabel("Ziel-Ausstiegstag"); a.set_ylabel("Ziel-Einstiegstag"); a.grid(False)
    for i in range(len(entries)):
        for j in range(len(exits)):
            if not np.isnan(M[i, j]):
                a.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                       color=INK if abs(M[i, j]) < lim * 0.6 else SURFACE)
    a.add_patch(plt.Rectangle((exits.index(27) - 0.5, entries.index(11) - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    a.set_title("Gesamtrendite 2010–2026 in % je Einstiegs-/Ausstiegstag (ohne SL/TP)")
    fig.colorbar(im, ax=a, label="Rendite %")
    save(fig, "6_parameter_landkarte.png")
    vals = M[~np.isnan(M)]
    log("\n6) Parameter-Landkarte (Einstiegstag 6–16 x Ausstiegstag 20–28, ohne SL/TP):")
    log(f"   11/27: {M[entries.index(11), exits.index(27)]:.1f} % | alle Varianten: Median {np.median(vals):.1f} %, "
        f"positiv {(vals > 0).mean()*100:.0f} %, Min {vals.min():.1f} %, Max {vals.max():.1f} %")
    nb = M[entries.index(9):entries.index(13) + 1, exits.index(25):exits.index(28) + 1]
    log(f"   Nachbarn (Einstieg 9–13, Ausstieg 25–28): Median {np.nanmedian(nb):.1f} %, Min {np.nanmin(nb):.1f} %")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chart_equity()
    chart_walkforward()
    chart_montecarlo()
    chart_regression()
    chart_permutation()
    chart_heatmap()
    with open(os.path.join(OUT, "quant_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
