"""Record a run in the app's replay format (replays/<name>.jsonl.gz)."""
import gzip, json, sys
from pathlib import Path
import numpy as np
import engine
from server import maze_info

name, flies, seed, smell, gain, trail = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == '1', float(sys.argv[5]), sys.argv[6] == '1'
t_max = float(sys.argv[7]) if len(sys.argv) > 7 else 90
sim = engine.Simulation(flies, seed=seed, smell=smell, steer_gain=gain, pack_trail=trail)
p = {'flies': flies, 'seed': seed, 'smell': smell, 'trail': trail, 'steer_gain': gain, 't_max': t_max}
header = {'type': 'start', 'name': name, 'params': p, 'maze': maze_info(sim.arena), 'flies': flies, 'regions': list(engine.REGIONS)}
Path('replays').mkdir(exist_ok=True)
with gzip.open(f'replays/{name}.jsonl.gz', 'wt') as f:
    f.write(json.dumps(header) + '\n')
    while sim.t < t_max and np.isnan(sim.done).any():
        out = sim.step(record_fly=0)
        f.write(json.dumps({'type': 'frame', 't': round(sim.t, 3), 'pos': np.round(sim.pos, 2).tolist(),
                            'head': np.round(sim.head, 3).tolist(),
                            'done': [None if np.isnan(d) else round(d, 2) for d in sim.done],
                            'dna02': np.round(np.column_stack([sim.rL, sim.rR]), 1).tolist(),
                            'reg': np.round(out['reg'], 2).tolist(), 'focus': 0,
                            'spiked': out['spiked'].tolist()}, separators=(',', ':')) + '\n')
    f.write(json.dumps({'type': 'end', 'name': name, 'done': [None if np.isnan(d) else round(d, 2) for d in sim.done], 't': round(sim.t, 2)}) + '\n')
print(name, sim.done)
