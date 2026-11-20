// Oeffnet cockpit.html mit den Dateien aus output/ in allen vier Reitern (Desktop und Handy) und meldet
// jeden JavaScript-Fehler und jede seitliche Verschiebung auf dem Handy. Exit-Code 1 bei einem Fehler.
// Aufruf aus makro_ml/:  node pruefe_cockpit.js
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const ROOT = __dirname + path.sep;
(async () => {
  const b = await chromium.launch();
  let bad = 0;
  for (const tab of ['alfred', 'fred', 'faktoren', 'termine']) {
    for (const [name, vp] of [['Desktop', { width: 1300, height: 900 }], ['Handy', { width: 390, height: 900 }]]) {
      const p = await b.newPage({ viewport: vp });
      const errs = [];
      p.on('pageerror', e => errs.push(e.message));
      await p.route('http://cockpit.test/**', r => {
        const f = new URL(r.request().url()).pathname.slice(1);
        const file = f === 'index.html' ? ROOT + 'cockpit.html' : ROOT + 'output/' + f;
        if (!fs.existsSync(file)) return r.fulfill({ status: 404, body: '' });
        r.fulfill({ body: fs.readFileSync(file), contentType: f.endsWith('.json') ? 'application/json' : 'text/html; charset=utf-8' });
      });
      await p.goto('http://cockpit.test/index.html#' + tab);
      await p.waitForTimeout(1500);
      const w = await p.evaluate(() => document.documentElement.scrollWidth);
      const ok = !errs.length && w <= vp.width;
      if (!ok) bad++;
      console.log(`${ok ? 'OK    ' : 'FEHLER'} ${tab.padEnd(9)} ${name.padEnd(8)}${errs.length ? ' JS: ' + errs.join(' | ') : ''}${w > vp.width ? ` seitlich verschiebbar (${w}px)` : ''}`);
      await p.close();
    }
  }
  await b.close();
  console.log(bad ? `GESAMT: ${bad} FEHLER` : 'GESAMT: ALLES OK');
  process.exit(bad ? 1 : 0);
})();
