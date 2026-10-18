"""Many independent single-fly runs in parallel; saves paths and exit times per condition."""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import Simulation

T_MAX = 90.0


def one(job):
    cond, seed = job
    sim = Simulation(1, seed=seed, smell=(cond != 'blind'), steer_gain=6.0 if cond == 'smell6' else 3.0,
                     silence={'no_mb': ['Mushroom body'], 'no_lh': ['Lateral horn']}.get(cond, []))
    rows = []
    while sim.t < T_MAX and np.isnan(sim.done).any():
        sim.step()
        rows.append((sim.t, seed, *sim.pos[0], sim.head[0]))
    print(cond, seed, sim.done[0], flush=True)
    return cond, seed, sim.done[0], rows


if __name__ == '__main__':
    conds = sys.argv[1].split(',')
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    with Pool(int(sys.argv[4])) as p:
        res = p.map(one, [(c, s) for c in conds for s in seeds], chunksize=1)
    for c in conds:
        rr = [r for r in res if r[0] == c]
        pd.DataFrame([row for r in rr for row in r[3]], columns=['t', 'fly', 'x', 'y', 'heading']).to_parquet(f'b_{c}.parquet')
        np.save(f'b_{c}_done.npy', np.array([r[2] for r in rr]))
