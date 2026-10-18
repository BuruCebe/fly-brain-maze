// Runs a few fly brains. The page sends each fly's smell input every 10 ms of
// simulated time; the worker runs the brains and sends back their output.
import { parseNet, Brain } from './brain.js';

let net = null;
const brains = new Map();

onmessage = e => {
  const m = e.data;
  if (m.type === 'net') {
    net = parseNet(m.buf);
    postMessage({ type: 'ready' });
  } else if (m.type === 'flies') {
    brains.clear();
    for (const f of m.flies) brains.set(f.id, new Brain(net, { ...f.opts, silenced: new Set(f.opts.silenced) }));
    postMessage({ type: 'built', ids: m.flies.map(f => f.id) });
  } else if (m.type === 'step') {
    const out = {};
    for (const [id, d] of Object.entries(m.drives)) {
      const b = brains.get(+id);
      if (!b) continue;
      const r = b.run(m.nSteps, [[net.foodL, d[0]], [net.foodR, d[1]], [net.trailL, d[2]], [net.trailR, d[3]], [net.visL, d[4]], [net.visR, d[5]]], +id === m.record);
      out[id] = { reading: r.reading, dnL: r.dnL, dnR: r.dnR, reg: Array.from(r.regCount), spiked: r.spiked };
    }
    postMessage({ type: 'stepped', seq: m.seq, out });
  }
};
