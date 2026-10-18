"""Run one maze experiment from the command line and save the paths."""
import argparse
import time

import numpy as np
import pandas as pd

from engine import Simulation

p = argparse.ArgumentParser()
p.add_argument('--flies', type=int, default=1)
p.add_argument('--t', type=float, default=90)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--maze-seed', type=int, default=0)
p.add_argument('--blind', action='store_true')
p.add_argument('--trail', action='store_true')
p.add_argument('--silence', nargs='*', default=[])
p.add_argument('--out', default='run')
a = p.parse_args()

sim = Simulation(a.flies, maze_seed=a.maze_seed, seed=a.seed, smell=not a.blind,
                 pack_trail=a.trail, silence=a.silence)
rows = []
t0 = time.time()
while sim.t < a.t and np.isnan(sim.done).any():
    sim.step()
    for i in range(sim.B):
        rows.append((sim.t, i, *sim.pos[i], sim.head[i], sim.rL[i], sim.rR[i]))
    if round(sim.t / sim.body_dt) % 100 == 0:
        print(f'sim {sim.t:5.1f}s  wall {time.time() - t0:5.0f}s  '
              f'done {np.isfinite(sim.done).sum()}/{sim.B}', flush=True)
pd.DataFrame(rows, columns=['t', 'fly', 'x', 'y', 'heading', 'dna02_L', 'dna02_R']).to_parquet(f'{a.out}.parquet')
np.save(f'{a.out}_done.npy', sim.done)
print('exit times', sim.done)
