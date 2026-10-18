"""Does the trimmed brain (active neurons plus everything with >=20 excitatory synapses from them) steer like the full one?"""
import sys
import numpy as np
import fastbrain
import engine
from flybrain import connectome

ids, W = connectome()
act = np.zeros(len(ids), bool)
for s in (1, 2, 3, 4):
    act |= np.load(f'active_{s}.npy')
exc = W.copy(); exc.data = np.maximum(exc.data, 0)
inp = np.asarray(exc[act].sum(0)).ravel() / 0.275
keep = act | (inp >= 20)
ro = np.load('steering_readout.npz')
keep |= np.isin(ids, ro['ids'])
np.save('browser_keep.npy', keep)
full = sys.argv[1] == 'full'
sim = engine.Simulation(6, seed=7)
if not full:  # rebuild the brain on the trimmed network
    ann = engine.annotations()
    al = ann[ann.cell_class.isin(['olfactory', 'ALLN'])].root_id.to_numpy()
    sim.brain = fastbrain.FastBrain(6, dt=0.25, seed=7, depress=al, U=0.5, tau_rec=500., silence=ids[~keep])
ks = np.array([1, 1, 0, 0, -1, -1.])
hzL, hzR = 12 + 10 * ks, 12 - 10 * ks
cnt_sum = np.zeros((6, len(sim.ro_idx)))
for i in range(600):
    cnt, dn, rec = sim.brain.run(40, [(sim.food[0], hzL), (sim.food[1], hzR)], sim.count_map, 11 + len(sim.ro_idx), sim.dnL, sim.dnR, -1)
    if i >= 100:
        cnt_sum += cnt[:, 11:]
rates = cnt_sum / 5.0
print('full' if full else 'trimmed', keep.sum(), 'neurons; readout per fly (left, left, even, even, right, right):', (rates @ sim.ro_w).round(2))
