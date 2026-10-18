// Whole-brain spiking model of one fruit fly, in JavaScript.
// Same leaky integrate-and-fire model and parameters as Shiu et al. 2024 (Nature),
// on the part of the FlyWire connectome that smell and steering use.
// Runs inside a Web Worker (worker.js) and in node for testing.

const V0 = -52, VRST = -52, VTH = -45, T_MBR = 20, TAU = 5, T_RFC = 2.2, T_DLY = 1.8;
const W_SYN = 0.275, KICK = W_SYN * 250, REST_TOL = 0.05, TAU_REC = 500;

export function parseNet(buf) {
  const h = new Int32Array(buf, 0, 4);
  const K = h[0], E = h[1], R = h[2];
  let o = 16;
  const take = (Type, n) => { const a = new Type(buf.slice(o, o + n * Type.BYTES_PER_ELEMENT)); o += n * Type.BYTES_PER_ELEMENT; return a; };
  const fullIndex = take(Int32Array, K);
  const region = take(Uint8Array, K);
  const flags = take(Uint16Array, K);
  const readout = take(Float32Array, K);
  const indptr = take(Int32Array, K + 1);
  const post = take(Uint16Array, E);
  const syn = take(Int16Array, E);
  const pick = bit => Int32Array.from([...flags.keys()].filter(i => flags[i] & bit));
  return {
    K, E, R, fullIndex, region, flags, readout, indptr, post, syn,
    foodL: pick(1), foodR: pick(2), trailL: pick(4), trailR: pick(8), visL: pick(128), visR: pick(256),
    roIdx: Int32Array.from([...readout.keys()].filter(i => readout[i] !== 0)),
  };
}

// small fast seeded RNG (mulberry32)
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Brain {
  // opts: dt (ms), seed, fatigue (U), individuality (mV of threshold spread), silenced (Set of region ids)
  constructor(net, opts = {}) {
    this.net = net;
    const K = net.K;
    this.dt = opts.dt ?? 0.25;
    this.D = Math.max(1, Math.round(T_DLY / this.dt));
    this.U = opts.fatigue ?? 0.5;
    this.rand = rng(opts.seed ?? 1);
    this.v = new Float32Array(K).fill(V0);
    this.g = new Float32Array(K);
    this.rfc = new Float32Array(K);
    this.x = new Float32Array(K).fill(1);
    this.tlast = new Float64Array(K);
    this.isLive = new Uint8Array(K);
    this.live = new Int32Array(K);
    this.nLive = 0;
    this.cap = 80000;
    this.qi = Array.from({ length: this.D }, () => new Int32Array(this.cap));
    this.qv = Array.from({ length: this.D }, () => new Float32Array(this.cap));
    this.qn = new Int32Array(this.D);
    this.t = 0;
    // each fly is an individual: every neuron's firing threshold is shifted a little
    const spread = opts.individuality ?? 0;
    this.vth = new Float32Array(K);
    for (let i = 0; i < K; i++) {
      const gauss = Math.sqrt(-2 * Math.log(this.rand() + 1e-12)) * Math.cos(2 * Math.PI * this.rand());
      this.vth[i] = VTH + spread * gauss;
    }
    this.off = new Uint8Array(K);
    const sil = opts.silenced ?? new Set();
    for (let i = 0; i < K; i++) if (sil.has(net.region[i])) this.off[i] = 1;
    this.dep = new Uint8Array(K);
    for (let i = 0; i < K; i++) this.dep[i] = (net.flags[i] & 16) ? 1 : 0;
  }

  wake(k) {
    if (!this.isLive[k]) { this.isLive[k] = 1; this.live[this.nLive++] = k; }
  }

  // Run nSteps of dt with Poisson input. drive: array of [Int32Array idx, rateHz].
  // Returns spike counts per region, per readout neuron, DNa02 L/R, and (if record) the neurons that spiked.
  run(nSteps, drive, record) {
    const { indptr, post, syn, region, flags } = this.net;
    const dt = this.dt, D = this.D;
    const regCount = new Float32Array(this.net.R);
    const roCount = new Float32Array(this.net.K);
    let dnL = 0, dnR = 0;
    const spiked = record ? new Set() : null;
    for (let s = 0; s < nSteps; s++) {
      const t = this.t + s, slot = t % D;
      const qi = this.qi[slot], qv = this.qv[slot], n = this.qn[slot];
      for (let e = 0; e < n; e++) { const k = qi[e]; this.g[k] += qv[e]; this.wake(k); }
      this.qn[slot] = 0;
      for (const [idx, rate] of drive) {
        const p = rate * dt / 1000;
        if (p <= 0) continue;
        for (let m = 0; m < idx.length; m++) if (this.rand() < p) { const k = idx[m]; this.v[k] += KICK; this.wake(k); }
      }
      let keep = 0;
      const live = this.live;
      for (let i = 0; i < this.nLive; i++) {
        const k = live[i];
        if (this.rfc[k] > 0) { this.rfc[k] -= dt; live[keep++] = k; continue; }
        let vk = this.v[k] + (V0 - this.v[k] + this.g[k]) / T_MBR * dt;
        let gk = this.g[k] - this.g[k] / TAU * dt;
        if (vk > this.vth[k] && !this.off[k]) {
          vk = VRST; gk = 0; this.rfc[k] = T_RFC - dt;
          let strength = 1;
          if (this.dep[k]) {  // short-term depression, recovered exactly since the last spike
            this.x[k] = 1 - (1 - this.x[k]) * Math.exp(-(t - this.tlast[k]) * dt / TAU_REC);
            this.tlast[k] = t;
            strength = this.x[k];
            this.x[k] -= this.U * this.x[k];
          }
          let c = this.qn[slot];
          const w = W_SYN * strength;
          for (let j = indptr[k], end = indptr[k + 1]; j < end; j++) {
            const target = post[j];
            if (this.off[target] || c >= this.cap) continue;
            qi[c] = target; qv[c] = syn[j] * w; c++;
          }
          this.qn[slot] = c;
          regCount[region[k]]++;
          roCount[k]++;
          if (flags[k] & 32) dnL++;
          else if (flags[k] & 64) dnR++;
          if (spiked) spiked.add(k);
        }
        this.v[k] = vk; this.g[k] = gk;
        if (this.rfc[k] <= 0 && Math.abs(vk - V0) < REST_TOL && Math.abs(gk) < REST_TOL) {
          this.v[k] = V0; this.g[k] = 0; this.isLive[k] = 0;
        } else live[keep++] = k;
      }
      this.nLive = keep;
    }
    this.t += nSteps;
    let reading = 0;
    for (const i of this.net.roIdx) reading += roCount[i] * this.net.readout[i];
    return { regCount, reading, dnL, dnR, spiked: spiked ? Int32Array.from(spiked) : null };
  }
}
