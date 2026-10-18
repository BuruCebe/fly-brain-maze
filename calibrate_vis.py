"""Record every descending neuron while another fly is seen on the left, right or both sides (no smell)."""
import sys
import numpy as np
from engine import Simulation, annotations

VIS = ['LC10a', 'LC11', 'LC12', 'LC15', 'LC17', 'LC18', 'LC9', 'LC10d']
seed = int(sys.argv[1])
ann = annotations()
ks = np.repeat([-1.0, 0.0, 1.0], 3)
sim = Simulation(len(ks), seed=seed)
b = sim.brain
VL = b.ids_to_idx(ann[ann.cell_type.isin(VIS) & (ann.side == 'left')].root_id)
VR = b.ids_to_idx(ann[ann.cell_type.isin(VIS) & (ann.side == 'right')].root_id)
is_dn = (ann.set_index('root_id').reindex(b.ids).super_class == 'descending').to_numpy()
dn_idx = np.flatnonzero(is_dn)
region = np.full(b.N, len(dn_idx), np.int64); region[dn_idx] = np.arange(len(dn_idx))
hzL = np.where(ks >= 0, 8.0, 0); hzR = np.where(ks <= 0, 8.0, 0)
counts = []
for i in range(120):
    reg, dn, rec = b.run(400, [(VL, hzL), (VR, hzR)], region, len(dn_idx) + 1, sim.dnL, sim.dnR, -1)
    counts.append(reg[:, :-1])
np.savez(f'calibvis_{seed}.npz', counts=np.array(counts), ks=ks, dn_ids=b.ids[dn_idx])
print('done', seed, flush=True)
