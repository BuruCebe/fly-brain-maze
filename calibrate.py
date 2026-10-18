"""Open-loop recording of every descending neuron while smell comes from the left, right or both sides.

Saves per-fly, per-100 ms spike counts of each DN so a steering readout can be
fit on some seeds and checked on others.
"""
import sys
import numpy as np
from engine import Simulation, BASE_HZ, annotations

seed = int(sys.argv[1])
ks = np.repeat([-1.0, 0.0, 1.0], 3)
sim = Simulation(len(ks), seed=seed)
b = sim.brain
ann = annotations().set_index('root_id')
is_dn = (ann.reindex(b.ids).super_class == 'descending').to_numpy()
dn_idx = np.flatnonzero(is_dn)
region = np.full(b.N, len(dn_idx), np.int64)
region[dn_idx] = np.arange(len(dn_idx))
hzL, hzR = BASE_HZ + 10 * ks, BASE_HZ - 10 * ks
counts = []
for i in range(120):  # 12 s in 100 ms bins
    reg, dn, rec = b.run(400, [(sim.food[0], hzL), (sim.food[1], hzR)], region, len(dn_idx) + 1, sim.dnL, sim.dnR, -1)
    counts.append(reg[:, :-1])
np.savez(f'calib_{seed}.npz', counts=np.array(counts), ks=ks, dn_ids=b.ids[dn_idx])
print('done', seed)
