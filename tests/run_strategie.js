// Hilfsprogramm fuer tests/test_strategie.py: liest {series, ends, crypto, configs, today?} von stdin,
// rechnet jede Konfiguration mit makro_ml/strategie.js und gibt die Ergebnisse als JSON aus.
const path = require('path');
const S = require(path.join(__dirname, '..', 'makro_ml', 'strategie.js'));
let buf = '';
process.stdin.on('data', d => (buf += d));
process.stdin.on('end', () => {
  const inp = JSON.parse(buf);
  const s = S.decodeSeries(inp.series);
  const data = { ...s, ends: inp.ends, crypto: !!inp.crypto };
  const out = { results: [], ends: {} };
  for (const cfg of inp.configs || []) {
    try {
      const r = S.backtest(data, cfg);
      out.results.push({ g: r.g, days: r.days, metrics: r.metrics, trades: r.trades, exposure: r.exposure,
        placebo: inp.placebo ? S.placebo(data, cfg, r, 50, 42) : null });
    } catch (e) { out.results.push({ error: String(e.message || e) }); }
  }
  if (inp.today) for (const iv of ['1wk', '1mo']) out.ends[iv] = S.periodEnds(s.t, iv, Date.parse(inp.today + 'T00:00:00Z'));
  if (inp.math) out.math = { normInv: inp.math.p.map(S.normInv), t: inp.math.t.map(([t, df]) => S.tTwoSided(t, df)),
    ems: inp.math.ems.map(([m, y]) => S.expectedMaxSharpe(m, y)) };
  process.stdout.write(JSON.stringify(out));
});
