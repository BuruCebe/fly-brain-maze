"""Neurons that fire when a fly sees another fly (small-object visual neurons, left/right/both), with and without food smell."""
import sys
import numpy as np
from engine import Simulation, annotations

VIS = ['LC10a', 'LC11', 'LC12', 'LC15', 'LC17', 'LC18', 'LC9', 'LC10d']
ann = annotations()
seed = int(sys.argv[1])
sim = Simulation(8, seed=seed)
b = sim.brain
rng = np.random.default_rng(seed)
VL = b.ids_to_idx(ann[ann.cell_type.isin(VIS) & (ann.side == 'left')].root_id)
VR = b.ids_to_idx(ann[ann.cell_type.isin(VIS) & (ann.side == 'right')].root_id)
# fly: (vis left, vis right, food left, food right) Hz; each fly drives a different random share of the visual neurons
conds = np.array([[40, 0, 0, 0], [0, 40, 0, 0], [40, 40, 0, 0], [20, 0, 12, 12], [0, 20, 22, 2], [40, 40, 24, 24], [10, 10, 0, 0], [40, 0, 2, 22]], float)
ever = np.zeros(b.N, bool)
for rep in range(3):
    subL = rng.choice(VL, int(.3 * len(VL)), replace=False); subR = rng.choice(VR, int(.3 * len(VR)), replace=False)
    for i in range(150):
        drive = [(subL, conds[:, 0]), (subR, conds[:, 1]), (sim.food[0], conds[:, 2]), (sim.food[1], conds[:, 3])]
        _, _, rec = b.run(40, drive, sim.regions, 11, sim.dnL, sim.dnR, i % 8)
        ever |= rec
np.save(f'active_vis_{seed}.npy', ever)
print(seed, ever.sum(), flush=True)
