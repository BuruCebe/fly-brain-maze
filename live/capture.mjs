// Records the 3D simulator frame by frame: 0.04 s of simulated time per frame, played back at 25 fps = real time.
// usage: node capture.mjs <outdir> <maxSeconds> [nFlies] [noSmell] [mazeSize] [focus]
import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
import { mkdirSync } from 'fs';
const [,, out, maxT = '45', nFlies = '4', noSmell = '1', size = '4', focusId = '1', mSeed, fSeed] = process.argv;
mkdirSync(out, { recursive: true });
const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
p.on('pageerror', e => console.log('pageerror:', e.message));
await p.route('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js', r => r.fulfill({ path: 'maze/web/vendor/three.module.js', contentType: 'text/javascript' }));
await p.route('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js', r => r.fulfill({ path: 'maze/web/vendor/OrbitControls.js', contentType: 'text/javascript' }));
await p.route('https://fonts.googleapis.com/**', r => r.fulfill({ body: '', contentType: 'text/css' }));
await p.goto('http://localhost:8766/index.html#capture');
await p.waitForSelector('#loading[hidden]', { state: 'attached', timeout: 120000 });
await p.evaluate(([n, ns, sz]) => {
  const set = (id, v) => { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event('input')); };
  set('nFlies', n); set('noSmell', ns); set('mazeSize', sz);
}, [nFlies, noSmell, size]);
if (mSeed) await p.evaluate(([m, f]) => window.__cap.seeded(m, f), [+mSeed, +fSeed]); else await p.evaluate(() => window.__cap.setup());
await p.evaluate(id => window.__cap.focus(id), +focusId);
let k = 0, st;
for (; ;) {
  st = await p.evaluate(() => window.__cap.step(4));
  await p.evaluate(() => { window.__cap.draw(); });
  await p.screenshot({ path: `${out}/f${String(k++).padStart(5, '0')}.png` });
  if (k % 50 === 0) console.log('t', st.t.toFixed(2), 'fed', JSON.stringify(st.fed));
  if (st.t >= +maxT || st.fed.every(x => x !== null)) break;
}
// hold the last frame for 2 s
for (let i = 0; i < 50; i++) { await p.evaluate(() => window.__cap.draw()); await p.screenshot({ path: `${out}/f${String(k++).padStart(5, '0')}.png` }); }
console.log('done', k, 'frames, t', st.t.toFixed(2), JSON.stringify(st.fed));
await b.close();
