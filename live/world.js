// The maze, the food smell, the smell trails flies leave, and the flies' bodies.
// Units: millimetres and seconds. A fruit fly is about 2.5 mm long.
import { rng } from './brain.js';

export const CELL_PX = 5;   // mm per grid square (a maze cell is one open square plus walls)
const RES = 1;              // mm per smell-map pixel

// Random maze. complexity 0 = a perfect maze (one route); higher values knock out
// extra walls, which adds loops and shortcuts.
export function makeMaze(w, h, complexity, seed) {
  const rand = rng(seed);
  const H = 2 * h + 1, W = 2 * w + 1;
  const grid = new Uint8Array(H * W);
  const seen = new Uint8Array(h * w);
  const stack = [[0, 0]];
  seen[0] = 1; grid[1 * W + 1] = 1;
  while (stack.length) {
    const [y, x] = stack[stack.length - 1];
    const nb = [[0, 1], [1, 0], [0, -1], [-1, 0]].map(([dy, dx]) => [y + dy, x + dx])
      .filter(([a, b]) => a >= 0 && a < h && b >= 0 && b < w && !seen[a * w + b]);
    if (!nb.length) { stack.pop(); continue; }
    const [ny, nx] = nb[Math.floor(rand() * nb.length)];
    grid[(2 * ny + 1) * W + 2 * nx + 1] = 1;
    grid[(y + ny + 1) * W + (x + nx + 1)] = 1;
    seen[ny * w + nx] = 1;
    stack.push([ny, nx]);
  }
  // knock out some interior walls between two open cells
  const candidates = [];
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    if (grid[y * W + x]) continue;
    const horiz = y % 2 === 1 && x % 2 === 0, vert = y % 2 === 0 && x % 2 === 1;
    if (horiz || vert) candidates.push(y * W + x);
  }
  for (const c of candidates) if (rand() < complexity) grid[c] = 1;
  return { grid, W, H };
}

export class World {
  constructor({ w, h, complexity, seed }) {
    const m = makeMaze(w, h, complexity, seed);
    Object.assign(this, m);
    this.size = [m.W * CELL_PX, m.H * CELL_PX];
    this.nx = Math.round(this.size[0] / RES); this.ny = Math.round(this.size[1] / RES);
    this.open = new Uint8Array(this.nx * this.ny);
    for (let y = 0; y < this.ny; y++) for (let x = 0; x < this.nx; x++)
      this.open[y * this.nx + x] = m.grid[Math.floor(y * RES / CELL_PX) * m.W + Math.floor(x * RES / CELL_PX)];
    this.start = [1.5 * CELL_PX, 1.5 * CELL_PX];
    // food goes in the cell farthest (by walking) from the start
    const fromStart = this.walk(this.start);
    let best = 0, bi = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const px = [(2 * x + 1.5) * CELL_PX, (2 * y + 1.5) * CELL_PX];
      const d = fromStart[this.pix(px)];
      if (isFinite(d) && d > best) { best = d; bi = [px[0], px[1]]; }
    }
    this.exit = bi;
    this.routeLength = best;
    // food smell falls off with walking distance to the food, not straight-line distance
    const dist = this.walk(this.exit);
    this.distExit = dist;
    const len = Math.max(40, 0.5 * best);
    this.odor = new Float32Array(this.nx * this.ny);
    for (let i = 0; i < dist.length; i++) this.odor[i] = isFinite(dist[i]) ? Math.exp(-dist[i] / len) : 0;
  }

  pix([x, y]) {
    const ix = Math.min(this.nx - 1, Math.max(0, Math.floor(x / RES)));
    const iy = Math.min(this.ny - 1, Math.max(0, Math.floor(y / RES)));
    return iy * this.nx + ix;
  }
  isOpen(p) { return this.open[this.pix(p)] === 1; }

  walk(p) {
    const d = new Float32Array(this.nx * this.ny).fill(Infinity);
    const q = new Int32Array(this.nx * this.ny);
    let head = 0, tail = 0;
    const s = this.pix(p); d[s] = 0; q[tail++] = s;
    while (head < tail) {
      const i = q[head++], x = i % this.nx, y = (i - x) / this.nx;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const a = x + dx, b = y + dy;
        if (a < 0 || b < 0 || a >= this.nx || b >= this.ny) continue;
        const j = b * this.nx + a;
        if (this.open[j] && d[j] === Infinity) { d[j] = d[i] + RES; q[tail++] = j; }
      }
    }
    return d;
  }
}

// Each fly leaves a smell trail that fades. A fly smells everyone's trail except its own.
export class Trails {
  constructor(world, n, tau = 20) {
    this.w = world; this.tau = tau;
    this.total = new Float32Array(world.nx * world.ny);
    this.own = Array.from({ length: n }, () => new Float32Array(world.nx * world.ny));
    this.steps = 0;
  }
  deposit(i, p, amount) { const k = this.w.pix(p); this.total[k] += amount; this.own[i][k] += amount; }
  decay(dt) {
    // fading is applied every 0.25 s to keep it cheap
    this.steps++;
    if (this.steps % 25) return;
    const f = Math.exp(-0.25 / this.tau);
    for (let k = 0; k < this.total.length; k++) this.total[k] *= f;
    for (const o of this.own) for (let k = 0; k < o.length; k++) o[k] *= f;
  }
  others(i, p) { const k = this.w.pix(p); return Math.max(0, this.total[k] - this.own[i][k]); }
}

const ANT_AHEAD = 1, ANT_OFFSET = 1, BASE_HZ = 12, CONTRAST = 10, CONTRAST_AMP = 25, TAU_STEER = 0.3;
// The steering signal adapts: a turn signal that never changes (a fly's own left/right bias, or something
// it has stared at for a while) fades over a few seconds, the way sensory neurons stop reporting a constant input.
const TAU_ADAPT = 3, MAX_TURN = 6;

// Smell at each antenna -> receptor firing rate on that side. The smell differs by only a few
// percent across a 2 mm head, so the left/right difference is amplified first. Strength is felt
// on a log scale (like real receptors), so the smell keeps getting stronger all the way to the food.
export function ornRates(cL, cR, gain) {
  const k = Math.tanh(CONTRAST_AMP * (cL - cR) / (cL + cR + 1e-9));
  const level = Math.min(1, Math.max(0, 1 + Math.log(Math.max(cL, cR) + 1e-9) / 4)) * gain;
  return [level * (BASE_HZ + CONTRAST * k), level * (BASE_HZ - CONTRAST * k)];
}

export class Fly {
  constructor(id, world, seed) {
    this.id = id; this.w = world;
    this.rand = rng(seed * 7919 + 13);
    this.pos = [world.start[0] + (this.rand() - .5) * 1.5, world.start[1] + (this.rand() - .5) * 1.5];
    this.head = this.rand() * 2 * Math.PI;
    this.steer = 0; this.bias = 0; this.followCue = 0; this.dnL = 0; this.dnR = 0;
    this.done = null;
    this.path = [this.pos.slice()];
  }
  antennae() {
    const c = Math.cos(this.head), s = Math.sin(this.head);
    const tx = this.pos[0] + ANT_AHEAD * c, ty = this.pos[1] + ANT_AHEAD * s;
    return [[tx - ANT_OFFSET * s, ty + ANT_OFFSET * c], [tx + ANT_OFFSET * s, ty - ANT_OFFSET * c]];
  }
  gauss() { return Math.sqrt(-2 * Math.log(this.rand() + 1e-12)) * Math.cos(2 * Math.PI * this.rand()); }
  // reading: steering signal in spikes/s from the brain; positive turns left
  move(result, bodyDt, p) {
    const a = bodyDt / TAU_STEER;
    this.steer += a * (result.reading / bodyDt - this.steer);
    this.dnL += a * (result.dnL / bodyDt - this.dnL);
    this.dnR += a * (result.dnR / bodyDt - this.dnR);
    if (this.done !== null) return;
    this.bias += bodyDt / TAU_ADAPT * (this.steer - this.bias);
    // the brain's own steering, plus (if switched on) the added follow channel toward the food mark
    const turn = Math.max(-MAX_TURN, Math.min(MAX_TURN, p.steerGain * (this.steer - this.bias) + (p.followGain ?? 0) * this.followCue));
    this.head += turn * bodyDt + p.wander * Math.sqrt(bodyDt) * this.gauss();
    const nx = this.pos[0] + p.speed * bodyDt * Math.cos(this.head);
    const ny = this.pos[1] + p.speed * bodyDt * Math.sin(this.head);
    if (this.w.isOpen([nx, ny])) this.pos = [nx, ny];
    else {
      if (this.w.isOpen([nx, this.pos[1]])) this.pos = [nx, this.pos[1]];
      else if (this.w.isOpen([this.pos[0], ny])) this.pos = [this.pos[0], ny];
      this.head += (this.rand() < .5 ? -1 : 1) * Math.PI / 2;  // turn away from the wall
    }
    if (Math.hypot(this.pos[0] - this.w.exit[0], this.pos[1] - this.w.exit[1]) < CELL_PX * .8) this.done = true;
  }
}

// ---------- vision ----------
// Other flies in view, as summed angular size (radians) on the left and right eye.
// Walls block the view; a fly cannot see straight behind itself.
const VIEW_RANGE = 40, BLIND_BEHIND = 20 * Math.PI / 180, FLY_HALF_LENGTH = 1.25;
export function seeFlies(world, flies, me) {
  let L = 0, R = 0;
  for (const o of flies) {
    if (o === me) continue;
    const dx = o.pos[0] - me.pos[0], dy = o.pos[1] - me.pos[1], d = Math.hypot(dx, dy);
    if (d > VIEW_RANGE || d < 1e-6) continue;
    let rel = Math.atan2(dy, dx) - me.head;
    rel = Math.atan2(Math.sin(rel), Math.cos(rel));
    if (Math.abs(rel) > Math.PI - BLIND_BEHIND) continue;
    let clear = true;
    for (let s = 1; s < d; s += 1) if (!world.isOpen([me.pos[0] + dx * s / d, me.pos[1] + dy * s / d])) { clear = false; break; }
    if (!clear) continue;
    const size = 2 * Math.atan(FLY_HALF_LENGTH / Math.max(d, 1.5));
    if (rel > 0) L += size; else R += size;   // positive angle is to the fly's left
  }
  return [L, R];
}

// Everything a fly senses this instant, as firing rates (Hz) for the six input groups the brain takes:
// food smell left/right, other flies' smell left/right, other flies seen left/right.
const VIS_HZ = 8, VIS_FULL = 0.5;  // a fly 5 mm away (about 28 degrees) drives the visual neurons fully
export function senses(arena, fly, local, P) {
  const w = arena.world;
  const [l, r] = fly.antennae();
  let food = [0, 0], trail = [0, 0], vis = [0, 0];
  fly.followCue = 0;
  if (P.smellFood) food = ornRates(w.odor[w.pix(l)], w.odor[w.pix(r)], P.sens / 100);
  if (arena.social && arena.flies.length > 1) {
    if (P.smellEach) {
      const s = P.trail / 100;
      let mL = 0, mR = 0;
      if (P.follow) { mL = foodMark(arena, l); mR = foodMark(arena, r); }
      trail = ornRates(arena.trails.others(local, l) * s + mL, arena.trails.others(local, r) * s + mR, P.sens / 100);
      if (mL + mR > 1e-4) {
        const level = Math.min(1, Math.max(0, 1 + Math.log(Math.max(mL, mR)) / 4));
        fly.followCue = level * Math.tanh(CONTRAST_AMP * (mL - mR) / (mL + mR));
      }
    }
    if (P.seeEach) {
      const [aL, aR] = seeFlies(w, arena.flies, fly);
      vis = [VIS_HZ * Math.min(1, aL / VIS_FULL), VIS_HZ * Math.min(1, aR / VIS_FULL)];
    }
  }
  return [food[0], food[1], trail[0], trail[1], vis[0], vis[1]];
}

// ---------- food mark (the added "follow" channel) ----------
// Real fruit flies that have fed leave aggregation pheromones on the food, and other flies are drawn
// to food that carries them. Here every fly that reaches the food marks it; the mark spreads out along
// the corridors over the next several seconds. The brain still smells it through its pheromone neurons,
// but in the FlyWire wiring those barely reach the steering neurons, so turning toward the mark is an
// added rule (P.follow), not something the copied brain does on its own.
const MARK_LEN = 30, MARK_RISE = 8;
export function foodMark(arena, p) {
  const t = arena.t ?? 0, d = arena.world.distExit[arena.world.pix(p)];
  if (!isFinite(d)) return 0;
  let m = 0;
  for (const f of arena.flies) if (typeof f.done === 'number') m += 1 - Math.exp(-Math.max(0, t - f.done) / MARK_RISE);
  return m * Math.exp(-d / MARK_LEN);
}
