"""Which neurons ever fire when a fly smells food and/or other flies? Union over conditions and seeds."""
import sys
import numpy as np
from engine import Simulation, BASE_HZ

seed = int(sys.argv[1])
# per fly: (food left, food right, trail left, trail right) in Hz
conds = np.array([[22, 2, 0, 0], [2, 22, 0, 0], [12, 12, 0, 0], [0, 0, 22, 2], [0, 0, 2, 22],
                  [22, 2, 22, 2], [2, 22, 22, 2], [24, 24, 24, 24], [6, 6, 0, 0], [0, 0, 6, 6]], float)
sim = Simulation(len(conds), seed=seed)
b = sim.brain
ever = np.zeros(b.N, bool)
per = np.zeros((len(conds), b.N), bool)
for f in range(len(conds)):
    pass
for i in range(400):  # 4 s
    drive = [(sim.food[0], conds[:, 0]), (sim.food[1], conds[:, 1]),
             (sim.trail_orn[0], conds[:, 2]), (sim.trail_orn[1], conds[:, 3])]
    for f in range(len(conds)):
        pass
    _, _, rec = b.run(40, drive, sim.regions, 11, sim.dnL, sim.dnR, i % len(conds))
    ever |= rec
np.save(f'active_{seed}.npy', ever)
print(seed, ever.sum())
