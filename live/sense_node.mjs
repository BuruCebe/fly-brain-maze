import { readFileSync } from 'fs';
import { gunzipSync } from 'zlib';
import { parseNet, Brain } from './brain.js';
import { ornRates } from './world.js';
const raw = gunzipSync(Buffer.from(readFileSync('brain.b64.txt', 'utf8'), 'base64'));
const net = parseNet(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
const cases = {
  none: [0,0,0,0,0,0],
  foodLeft: [...ornRates(0.26,0.24,1),0,0,0,0], foodRight: [...ornRates(0.24,0.26,1),0,0,0,0],
  foodEven: [...ornRates(0.25,0.25,1),0,0,0,0],
  trailLeft: [0,0,...ornRates(0.26,0.24,1),0,0], trailRight: [0,0,...ornRates(0.24,0.26,1),0,0],
  seeLeft: [0,0,0,0,8,0], seeRight: [0,0,0,0,0,8],
};
for (const [k, d] of Object.entries(cases)) {
  const b = new Brain(net, { dt: 0.5, seed: 3, individuality: 0.5 });
  const xs = [];
  for (let i = 0; i < 400; i++) { // 10 ms steps, 4 s
    const r = b.run(20, [[net.foodL,d[0]],[net.foodR,d[1]],[net.trailL,d[2]],[net.trailR,d[3]],[net.visL,d[4]],[net.visR,d[5]]], false);
    if (i >= 100) xs.push(6 * r.reading / 0.01); // rad/s before smoothing
  }
  const m = xs.reduce((a,b)=>a+b)/xs.length, sd = Math.sqrt(xs.reduce((a,b)=>a+(b-m)**2,0)/xs.length);
  console.log(k.padEnd(11), 'turn rate mean', m.toFixed(2), 'rad/s  sd', sd.toFixed(2));
}
