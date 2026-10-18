import { readFileSync } from 'fs';
import { gunzipSync } from 'zlib';
import { parseNet, Brain } from './brain.js';
const raw = gunzipSync(Buffer.from(readFileSync('brain.b64.txt', 'utf8'), 'base64'));
const net = parseNet(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
const [base, con] = process.argv.slice(2).map(Number);
const cases = { L: [base + con, base - con], E: [base, base], R: [base - con, base + con] };
for (const [k, d] of Object.entries(cases)) {
  const xs = []; let act = 0;
  for (const seed of [1, 2, 3]) {
    const b = new Brain(net, { dt: 0.5, seed, individuality: 0.5 });
    for (let i = 0; i < 300; i++) {
      const r = b.run(20, [[net.foodL, d[0]], [net.foodR, d[1]]], false);
      if (i >= 50) { xs.push(6 * r.reading / 0.01); act += r.regCount.reduce((a, b) => a + b); }
    }
  }
  const m = xs.reduce((a, b) => a + b) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  console.log(base, con, k, 'mean', m.toFixed(2), 'sd', sd.toFixed(1), 'D', (sd * sd * 0.01 / 2).toFixed(1), 'spikes/10ms', (act / xs.length).toFixed(0));
}
