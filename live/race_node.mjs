// One race in node, same loop as the page. Prints JSON: first arrival, all arrivals.
// usage: node race_node.mjs <n flies> <smellEach 0/1> <seeEach 0/1> <maze seed> <fly seed> [size] [limit s]
import { readFileSync } from 'fs';
import { gunzipSync } from 'zlib';
import { parseNet, Brain } from './brain.js';
import { World, Trails, Fly, senses } from './world.js';
const raw = gunzipSync(Buffer.from(readFileSync(new URL('./brain.b64.txt', import.meta.url), 'utf8'), 'base64'));
const net = parseNet(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
const [n, smellEach, seeEach, mazeSeed, flySeed, size = 3, limit = 90] = process.argv.slice(2).map(Number);
const P = { smellFood: process.env.NOSMELL ? false : true, smellEach: !!smellEach, seeEach: !!seeEach, trail: 100, sens: 100, follow: !!process.env.FOLLOW };
const p = { steerGain: +(process.env.GAIN ?? 2), wander: +(process.env.WANDER ?? 0.5), speed: 8, followGain: process.env.FOLLOW ? +process.env.FOLLOW : 0 };
const BODY_DT = 0.01, nSteps = 20;
const world = new World({ w: size, h: size, complexity: 0.1, seed: mazeSeed });
const flies = [], brains = [];
for (let i = 0; i < n; i++) {
  flies.push(new Fly(i, world, flySeed + i * 101));
  brains.push(new Brain(net, { dt: 0.5, fatigue: 0.5, individuality: 0.5, seed: flySeed + i * 977 }));
}
const arena = { world, flies, trails: new Trails(world, n), social: n > 1, t: 0 };
let t = 0, first = null, step = 0;
const rec = process.env.REC ? [] : null;
while (t < limit && flies.some(f => f.done === null)) {
  // WEAK=1: fly 0 cannot smell the food (it can still smell and see the other flies)
  const d = flies.map((f, i) => f.done === null ? senses(arena, f, i, (process.env.WEAK && i === 0) ? { ...P, smellFood: false } : P) : null);
  t += BODY_DT; arena.t = t;
  flies.forEach((f, i) => {
    if (!d[i]) return;
    const x = d[i];
    const r = brains[i].run(nSteps, [[net.foodL, x[0]], [net.foodR, x[1]], [net.trailL, x[2]], [net.trailR, x[3]], [net.visL, x[4]], [net.visR, x[5]]], false);
    f.move(r, BODY_DT, p);
    if (f.done === true) { f.done = t; if (first === null) { first = t; } }
    if (arena.social) arena.trails.deposit(i, f.pos, 0.05);
  });
  if (arena.social) arena.trails.decay(BODY_DT);
  if (rec && step++ % 4 === 0) rec.push(flies.map(f => [+f.pos[0].toFixed(2), +f.pos[1].toFixed(2), +f.head.toFixed(2), f.done === null ? 0 : 1]));
  if (process.env.FIRST && first !== null) break;
  if (process.env.WEAK && flies[0].done !== null) break;
}
if (rec) { const { writeFileSync } = await import('fs'); writeFileSync(process.env.REC, JSON.stringify({ grid: Array.from(world.grid), W: world.W, H: world.H, cell: 5, exit: world.exit, start: world.start, frames: rec })); }
console.log(JSON.stringify({ n, smellEach, seeEach, mazeSeed, flySeed, first, times: flies.map(f => f.done), route: world.routeLength }));
