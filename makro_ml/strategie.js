// Strategie-Tester fuer den Cockpit-Reiter "Statistik" (Regeln: SPEC_STATISTIK.md Abschnitt 5).
// Laeuft im Browser (window.Strategie) und in Node (module.exports) fuer die Gegenpruefung mit strategie_ref.py.
// Reine Funktionen, keine Seiteneffekte ausser dem Varianten-Zaehler im Browser (siehe cockpit.html).
(function (root) {
  'use strict';
  const DAY = 86400000;

  // ---------------------------------------------------------------- Hilfen: Datum
  const toDate = s => new Date(s + 'T00:00:00Z');
  const iso = t => new Date(t).toISOString().slice(0, 10);
  const wd = t => (new Date(t).getUTCDay() + 6) % 7;  // Mo = 0 … So = 6

  function decodeSeries(series) {
    // {d0, d, c} aus returns_stats.py -> {t: [ms], d: [iso], c: [kurs]}
    const t0 = toDate(series.d0).getTime();
    const t = series.d.map(x => t0 + x * DAY);
    return { t, d: t.map(iso), c: series.c.slice() };
  }

  function periodEnd(t, interval) {
    if (interval === '1d') return t;
    const d = new Date(t);
    if (interval === '1wk') return t + ((4 - wd(t) + 7) % 7) * DAY;   // Freitag der Sa–Fr-Woche
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0);     // Monatsende
  }

  // Indizes der abgeschlossenen Periodenenden (gleiche Regel wie returns_stats.py); fuer eigene CSV-Daten
  function periodEnds(t, interval, todayMs) {
    if (interval === '1d') return t.map((_, i) => i);
    const out = [];
    for (let i = 0; i < t.length; i++) {
      const last = i === t.length - 1 || periodEnd(t[i + 1], interval) !== periodEnd(t[i], interval);
      if (last) out.push(i);
    }
    if (out.length && periodEnd(t[out[out.length - 1]], interval) >= todayMs) out.pop();
    return out;
  }

  // ---------------------------------------------------------------- Hilfen: Verteilungen
  function normCdf(x) {  // Hart (1968) in der Form von West (2005), absolute Genauigkeit ~1e-14
    const z = Math.abs(x); let c;
    if (z > 37) c = 0;
    else {
      const e = Math.exp(-z * z / 2);
      if (z < 7.07106781186547) {
        let n = 3.52624965998911e-02 * z + 0.700383064443688; n = n * z + 6.37396220353165; n = n * z + 33.912866078383;
        n = n * z + 112.079291497871; n = n * z + 221.213596169931; n = n * z + 220.206867912376;
        let d = 8.83883476483184e-02 * z + 1.75566716318264; d = d * z + 16.064177579207; d = d * z + 86.7807322029461;
        d = d * z + 296.564248779674; d = d * z + 637.333633378831; d = d * z + 793.826512519948; d = d * z + 440.413735824752;
        c = e * n / d;
      } else { let b = z + 0.65; b = z + 4 / b; b = z + 3 / b; b = z + 2 / b; b = z + 1 / b; c = e / b / 2.506628274631; }
    }
    return x > 0 ? 1 - c : c;
  }
  function normInv(p) {  // Acklam, plus ein Newton-Schritt
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425; let q, r, x;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else if (p <= 1 - pl) { q = p - 0.5; r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    else { q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  }
  function lgamma(x) {
    const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += g[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function ibeta(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  // zweiseitiger p-Wert der t-Verteilung
  function tTwoSided(t, df) { if (!isFinite(t)) return t === t ? 0 : null; return ibeta(df / 2, 0.5, df / (df + t * t)); }

  function expectedMaxSharpe(M, years) {
    // Erwarteter bester (annualisierter) Sharpe von M Zufallsstrategien ohne Vorteil (Bailey/Lopez de Prado)
    if (!(M >= 2) || !(years > 0)) return 0;
    const g = 0.5772156649;
    return Math.sqrt(1 / years) * ((1 - g) * normInv(1 - 1 / M) + g * normInv(1 - 1 / (M * Math.E)));
  }

  // ---------------------------------------------------------------- Indikatoren auf Periodenschluessen x
  function sma(x, n, j) { if (j < n - 1) return null; let s = 0; for (let i = j - n + 1; i <= j; i++) s += x[i]; return s / n; }
  function stdev(a) { const n = a.length; if (n < 2) return null; let m = 0; for (const v of a) m += v; m /= n; let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / (n - 1)); }
  function rsiSeries(x, n) {
    const out = new Array(x.length).fill(null);
    if (x.length <= n) return out;
    let g = 0, l = 0;
    for (let i = 1; i <= n; i++) { const ch = x[i] - x[i - 1]; if (ch > 0) g += ch; else l -= ch; }
    g /= n; l /= n;
    const val = () => (l === 0 ? (g === 0 ? 50 : 100) : 100 - 100 / (1 + g / l));
    out[n] = val();
    for (let i = n + 1; i < x.length; i++) {
      const ch = x[i] - x[i - 1];
      g = (g * (n - 1) + (ch > 0 ? ch : 0)) / n; l = (l * (n - 1) + (ch < 0 ? -ch : 0)) / n;
      out[i] = val();
    }
    return out;
  }
  function cmp(v, op, th) { if (v === null || v === undefined || !isFinite(v)) return false; return op === 'gt' ? v > th : v < th; }

  // Wert einer Bedingung an jedem Periodenende j -> Array<boolean>
  function condSeries(cond, x, tE, interval, crypto) {
    const m = x.length, out = new Array(m).fill(false), n = Math.max(1, Math.round(cond.n || 1));
    const kP = { '1d': crypto ? 365 : 252, '1wk': 52, '1mo': 12 }[interval];
    const r = j => (j >= 1 ? x[j] / x[j - 1] - 1 : null);
    switch (cond.type) {
      case 'always': out.fill(true); break;
      case 'price_sma': for (let j = 0; j < m; j++) { const s = sma(x, n, j); out[j] = s !== null && cmp(x[j] - s, cond.op, 0); } break;
      case 'sma_sma': {
        const a = Math.max(1, Math.round(cond.a)), b = Math.max(1, Math.round(cond.b));
        for (let j = 0; j < m; j++) { const sa = sma(x, a, j), sb = sma(x, b, j); out[j] = sa !== null && sb !== null && cmp(sa - sb, cond.op, 0); }
        break;
      }
      case 'mom': for (let j = n; j < m; j++) out[j] = cmp(x[j] / x[j - n] - 1, cond.op, cond.th); break;
      case 'ret': for (let j = 1; j < m; j++) out[j] = cmp(r(j), cond.op, cond.th); break;
      case 'z': {
        if (n < 2) break;
        for (let j = n + 1; j < m; j++) {
          const w = []; for (let i = j - n; i <= j - 1; i++) w.push(r(i));
          const sd = stdev(w); if (!sd) continue;
          let mu = 0; for (const v of w) mu += v; mu /= w.length;
          out[j] = cmp((r(j) - mu) / sd, cond.op, cond.th);
        }
        break;
      }
      case 'rsi': { const rs = rsiSeries(x, n); for (let j = 0; j < m; j++) out[j] = cmp(rs[j], cond.op, cond.th); break; }
      case 'hilo':
        for (let j = n - 1; j < m; j++) {
          let hi = -Infinity, lo = Infinity; for (let i = j - n + 1; i <= j; i++) { if (x[i] > hi) hi = x[i]; if (x[i] < lo) lo = x[i]; }
          out[j] = cond.side === 'low' ? x[j] / lo - 1 <= cond.th : x[j] / hi - 1 >= -cond.th;
        }
        break;
      case 'vol': {
        if (n < 2) break;
        for (let j = n; j < m; j++) { const w = []; for (let i = j - n + 1; i <= j; i++) w.push(r(i)); const sd = stdev(w); out[j] = sd !== null && cmp(sd * Math.sqrt(kP), cond.op, cond.th); }
        break;
      }
      case 'calendar': {
        const months = new Set(cond.months || []), days = new Set(cond.weekdays || []);
        for (let j = 0; j < m; j++) {
          const dt = new Date(tE[j]);
          const okM = !months.size || months.has(dt.getUTCMonth() + 1);
          const okD = interval !== '1d' || !days.size || days.has(wd(tE[j]) + 1);
          out[j] = okM && okD;
        }
        break;
      }
      default: throw new Error('Unbekannte Bedingung: ' + cond.type);
    }
    return out;
  }

  function signals(data, cfg) {
    const E = cfg.interval === '1d' ? data.c.map((_, i) => i) : data.ends[cfg.interval];
    const x = E.map(i => data.c[i]), tE = E.map(i => data.t[i]);
    const conds = cfg.conds && cfg.conds.length ? cfg.conds : [{ type: 'always' }];
    const parts = conds.map(c => condSeries(c, x, tE, cfg.interval, !!data.crypto));
    const sig = x.map((_, j) => parts.every(p => p[j]));
    return { E, sig };
  }

  // ---------------------------------------------------------------- Backtest
  function backtest(data, cfg) {
    const c = data.c, t = data.t, T = c.length;
    const lo = cfg.start ? lowerBound(t, toDate(cfg.start).getTime()) : 0;
    const hi = cfg.end ? upperBound(t, toDate(cfg.end).getTime()) - 1 : T - 1;
    if (hi - lo < 2) throw new Error('Zeitraum zu kurz');
    const { E, sig } = signals(data, cfg);
    const evalAt = new Map(); E.forEach((i, j) => evalAt.set(i, j));
    const size = (cfg.size ?? 1) * (cfg.lev ?? 1);
    const k = cfg.cost || {};
    const side = ((k.fee_bp || 0) + (k.spread_bp || 0) / 2 + (k.slip_bp || 0)) / 1e4 * size;
    const fin = (k.fin_pa || 0) / 100 * size;
    const ex = cfg.exit || {};
    const lag = cfg.lag ?? 1;
    const dirSign = cfg.dir === 'short' ? -1 : 1;
    const desired = j => (cfg.dir === 'ls' ? (sig[j] ? 1 : -1) : (sig[j] ? dirSign : 0));

    const g = new Array(T).fill(0), posHeld = new Array(T).fill(0), costs = new Array(T).fill(0);
    let pos = 0, pend = null, entryPx = 0, peak = 0, entryJ = -1, ruined = false;
    let eq = 1; const equity = new Array(T).fill(null); equity[lo] = 1;
    const trades = []; let cur = null;

    const execute = (target, ti, reason) => {
      const dpos = Math.abs(target - pos);
      if (!dpos) return;
      const cost = dpos * side;
      g[ti] -= cost; costs[ti] += cost;
      if (pos !== 0 && cur) { cur.exitIdx = ti; cur.exitPx = c[ti]; cur.reason = reason; cur.growth *= 1 - side; trades.push(cur); cur = null; }
      pos = target;
      if (pos !== 0) { entryPx = c[ti]; peak = c[ti]; cur = { dir: pos, entryIdx: ti, entryPx: c[ti], growth: 1 - side }; }
    };

    for (let ti = lo; ti <= hi; ti++) {
      let exitedToday = false;
      if (ti > lo && !ruined) {
        // (1) Tagesertrag der Position seit gestern
        const ri = c[ti] / c[ti - 1] - 1;
        const a = pos !== 0 ? size * pos * ri - fin * (t[ti] - t[ti - 1]) / DAY / 365 : 0;
        g[ti] += a; posHeld[ti] = pos;
        if (cur) cur.growth *= 1 + a;
      }
      if (!ruined) {
        // (2) faellige Auftraege
        if (pend && pend.at === ti) { const before = pos; execute(pend.target, ti, pend.reason); if (before !== 0 && pend.target !== before) exitedToday = true; pend = null; }
        // (3) Stops an Tagesschluessen (nicht am Einstiegstag)
        if (pos !== 0 && !pend && cur && cur.entryIdx < ti) {
          const tr = pos * (c[ti] / entryPx - 1);
          if (pos > 0) peak = Math.max(peak, c[ti]); else peak = Math.min(peak, c[ti]);
          let why = null;
          if (ex.sl > 0 && tr <= -ex.sl / 100) why = 'Stop';
          else if (ex.tp > 0 && tr >= ex.tp / 100) why = 'Ziel';
          else if (ex.trail > 0 && pos * (c[ti] / peak - 1) <= -ex.trail / 100) why = 'Trailing';
          if (why) { exitedToday = true; order(0, ti, why); }
        }
        // (4) Signal an Periodenenden
        const j = evalAt.get(ti);
        if (j !== undefined && !pend) {
          const want = desired(j);
          if (pos !== 0) {
            const held = j - entryJ;
            if (cfg.dir === 'ls' && want !== pos && want !== 0) { if (ti + lag <= hi) { entryJ = j; order(want, ti, 'Wechsel'); } }
            else if (ex.signal && want !== pos) { exitedToday = true; order(0, ti, 'Signal'); }
            else if (ex.maxHold > 0 && held >= ex.maxHold) { exitedToday = true; order(0, ti, 'Haltedauer'); }
          } else if (want !== 0 && !exitedToday && ti + lag > lo && ti + lag <= hi) {  // am ersten Tag keine Ausfuehrung
            entryJ = j; order(want, ti, 'Einstieg');
          }
        }
      }
      eq *= 1 + g[ti];
      if (!ruined && eq <= 0) {  // Ruin: Verlust am Tag auf genau -100 % begrenzen
        g[ti] = -1; eq = 0; ruined = true; pos = 0; pend = null;
        if (cur) { cur.exitIdx = ti; cur.exitPx = c[ti]; cur.reason = 'Ruin'; cur.growth = 0; trades.push(cur); cur = null; }
      }
      equity[ti] = eq;
    }
    if (cur) { cur.exitIdx = hi; cur.exitPx = c[hi]; cur.reason = 'Ende'; trades.push(cur); cur = null; }

    function order(target, ti, reason) {
      if (lag === 0) execute(target, ti, reason);
      else if (ti + lag <= hi) pend = { target, at: ti + lag, reason };
    }

    const days = [];
    for (let i = lo + 1; i <= hi; i++) days.push(i);
    const bh = days.map(i => c[i] / c[i - 1] - 1);
    const gs = days.map(i => g[i]);
    const kd = data.crypto ? 365 : 252;
    const splitMs = cfg.split ? toDate(cfg.split).getTime() : t[lo] + 0.7 * (t[hi] - t[lo]);
    const isMask = days.map(i => t[i] < splitMs);
    const pick = (arr, want) => arr.filter((_, q) => isMask[q] === want);
    const dIS = pick(days, true), dOOS = pick(days, false);
    const tradeList = trades.map(tr => ({
      dir: tr.dir, entry: iso(t[tr.entryIdx]), exit: iso(t[tr.exitIdx]), entryIdx: tr.entryIdx, exitIdx: tr.exitIdx,
      entryPx: tr.entryPx, exitPx: tr.exitPx, ret: tr.growth - 1,   // 'Ende'/'Ruin': ohne Ausstiegskosten
      days: Math.round((t[tr.exitIdx] - t[tr.entryIdx]) / DAY), reason: tr.reason,
    }));
    const res = {
      range: { start: iso(t[lo]), end: iso(t[hi]), split: iso(splitMs), lo, hi },
      metrics: {
        all: metrics(gs, days.map(i => t[i]), t[lo], kd),
        is: dIS.length > 1 ? metrics(pick(gs, true), dIS.map(i => t[i]), t[lo], kd) : null,
        oos: dOOS.length > 1 ? metrics(pick(gs, false), dOOS.map(i => t[i]), t[dOOS[0] - 1], kd) : null,
        bh: metrics(bh, days.map(i => t[i]), t[lo], kd),
        bh_is: dIS.length > 1 ? metrics(pick(bh, true), dIS.map(i => t[i]), t[lo], kd) : null,
        bh_oos: dOOS.length > 1 ? metrics(pick(bh, false), dOOS.map(i => t[i]), t[dOOS[0] - 1], kd) : null,
      },
      trades: tradeList,
      tradeStats: tradeStats(tradeList),
      exposure: days.filter(i => posHeld[i] !== 0).length / days.length,
      costs: days.reduce((s, i) => s + costs[i], 0),
      sides: trades.reduce((s, tr) => s + (tr.reason === 'Ende' ? 1 : 2), 0),
      ruined,
      series: { t: [t[lo], ...days.map(i => t[i])], eq: [1, ...days.map(i => equity[i])], bh: cumprod([0, ...bh]) },
      g: gs, bhRet: bh, days,
      years: (t[hi] - t[lo]) / DAY / 365.25,
    };
    res.yearly = yearly(days.map(i => t[i]), gs, bh);
    return res;
  }

  function cumprod(r) { let e = 1; return r.map(v => (e *= 1 + v)); }
  function lowerBound(a, v) { let l = 0, h = a.length; while (l < h) { const m = (l + h) >> 1; if (a[m] < v) l = m + 1; else h = m; } return l; }
  function upperBound(a, v) { let l = 0, h = a.length; while (l < h) { const m = (l + h) >> 1; if (a[m] <= v) l = m + 1; else h = m; } return l; }

  function metrics(g, tt, t0, k) {
    const n = g.length;
    let e = 1, peak = 1, peakT = t0, mdd = 0, longest = 0, sum = 0, dn = 0;
    for (let i = 0; i < n; i++) {
      e *= 1 + g[i]; sum += g[i]; dn += Math.min(g[i], 0) ** 2;
      if (e >= peak) { peak = e; peakT = tt[i]; }
      else { mdd = Math.min(mdd, e / peak - 1); longest = Math.max(longest, (tt[i] - peakT) / DAY); }
    }
    const mean = sum / n;
    let ss = 0; for (const v of g) ss += (v - mean) ** 2;
    const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : null;
    const years = (tt[n - 1] - t0) / DAY / 365.25;
    const total = e - 1;
    const cagr = e > 0 && years > 0 ? Math.pow(e, 1 / years) - 1 : -1;
    const tstat = sd ? mean / (sd / Math.sqrt(n)) : null;
    const dd = Math.sqrt(dn / n);
    return {
      n, total, cagr, vol: sd === null ? null : sd * Math.sqrt(k), sharpe: sd ? mean / sd * Math.sqrt(k) : null,
      sortino: dd > 0 ? mean / dd * Math.sqrt(k) : null, maxdd: mdd, dd_days: Math.round(longest),
      calmar: mdd < 0 ? cagr / -mdd : null, mean, t: tstat, p: tstat === null ? null : tTwoSided(tstat, n - 1), years,
    };
  }

  function tradeStats(tr) {
    const n = tr.length;
    if (!n) return { n: 0 };
    const w = tr.filter(x => x.ret > 0), l = tr.filter(x => x.ret <= 0);
    const sw = w.reduce((s, x) => s + x.ret, 0), sl = l.reduce((s, x) => s + x.ret, 0);
    return {
      n, win: w.length / n, avg_win: w.length ? sw / w.length : null, avg_loss: l.length ? sl / l.length : null,
      pf: sl < 0 ? sw / -sl : null, avg_days: tr.reduce((s, x) => s + x.days, 0) / n,
      avg_ret: tr.reduce((s, x) => s + x.ret, 0) / n,
    };
  }

  function yearly(tt, g, bh) {
    const m = new Map();
    tt.forEach((ti, i) => {
      const y = new Date(ti).getUTCFullYear();
      if (!m.has(y)) m.set(y, [1, 1]);
      const v = m.get(y); v[0] *= 1 + g[i]; v[1] *= 1 + bh[i];
    });
    return [...m.entries()].map(([y, v]) => ({ year: y, strat: v[0] - 1, bh: v[1] - 1 }));
  }

  // ---------------------------------------------------------------- Zufallsvergleich (Placebo)
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  function placebo(data, cfg, res, runs = 200, seed = 42) {
    if (cfg.dir === 'ls') return { na: 'immer investiert – kein Zufallsvergleich möglich' };
    const tr = res.trades;
    if (tr.length < 3) return { na: 'zu wenige Trades (mind. 3)' };
    const c = data.c, t = data.t, lo = res.range.lo, hi = res.range.hi;
    const durs = tr.map(x => Math.max(1, x.exitIdx - x.entryIdx));
    const span = hi - lo, D = durs.reduce((s, x) => s + x, 0), free = span - D;
    if (free < 0) return { na: 'Trades füllen den Zeitraum aus' };
    const size = (cfg.size ?? 1) * (cfg.lev ?? 1), k = cfg.cost || {};
    const side = ((k.fee_bp || 0) + (k.spread_bp || 0) / 2 + (k.slip_bp || 0)) / 1e4 * size;
    const fin = (k.fin_pa || 0) / 100 * size;
    const dirs = tr.map(x => x.dir);
    const rnd = mulberry32(seed), kd = data.crypto ? 365 : 252;
    const sharpes = [], totals = [];
    for (let run = 0; run < runs; run++) {
      const cuts = Array.from({ length: durs.length }, () => Math.floor(rnd() * (free + 1))).sort((a, b) => a - b);
      const order = durs.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      const g = new Array(span).fill(0);
      let pos = lo, prevCut = 0;
      order.forEach((oi, q) => {
        pos += cuts[q] - prevCut; prevCut = cuts[q];
        const s = pos, e = pos + durs[oi], dir = dirs[oi];
        for (let i = s + 1; i <= e; i++) g[i - lo - 1] += size * dir * (c[i] / c[i - 1] - 1) - fin * (t[i] - t[i - 1]) / DAY / 365;
        g[Math.max(0, s - lo - 1)] -= side;   // Einstieg zum Schluss von Tag s
        g[e - lo - 1] -= side;                // Ausstieg zum Schluss von Tag e
        pos = e;
      });
      let m = 0; for (const v of g) m += v; m /= g.length;
      let ss = 0; for (const v of g) ss += (v - m) ** 2;
      const sd = Math.sqrt(ss / (g.length - 1));
      sharpes.push(sd ? m / sd * Math.sqrt(kd) : 0);
      totals.push(g.reduce((e, v) => e * (1 + v), 1) - 1);
    }
    const s0 = res.metrics.all.sharpe ?? 0;
    const ge = sharpes.filter(s => s >= s0).length;
    const sorted = sharpes.slice().sort((a, b) => a - b);
    return { runs, p: (ge + 1) / (runs + 1), median: sorted[Math.floor(runs / 2)], p95: sorted[Math.floor(runs * 0.95)], sharpes };
  }

  const API = { decodeSeries, periodEnds, periodEnd, signals, condSeries, backtest, metrics, placebo, expectedMaxSharpe,
    normInv, normCdf, tTwoSided, rsiSeries, iso };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.Strategie = API;
})(typeof window !== 'undefined' ? window : globalThis);
