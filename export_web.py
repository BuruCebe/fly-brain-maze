"""Export the trimmed brain for the in-browser simulator (../live/).

brain.b64.txt: base64 of gzip of one binary blob:
  header int32[4] = K, E, n_regions, 0
  full_index int32[K]   index into the full 138,639-neuron display cloud
  region uint8[K]       display / silencing region
  flags uint16[K]       1 food ORN left, 2 food ORN right, 4 trail ORN left, 8 trail ORN right,
                        16 short-term depression, 32 DNa02 left, 64 DNa02 right,
                        128 small-object visual neuron, left eye, 256 same, right eye
  readout float32[K]    steering weight (0 for most)
  indptr int32[K+1]     CSR rows = presynaptic
  post uint16[E]
  syn int16[E]          signed synapse count (multiply by w_syn = 0.275 mV)
cloud.b64.txt: base64 of float32 xyz[138639*3] + uint8 region[138639] for the brain picture.
"""
import base64, gzip, json
import numpy as np
import engine
from flybrain import connectome
from server import brain_geometry

ids, W = connectome()
VIS = ['LC10a', 'LC11', 'LC12', 'LC15', 'LC17', 'LC18', 'LC9', 'LC10d']
act = np.zeros(len(ids), bool)
for f in ['active_%d.npy' % s for s in (1, 2, 3, 4)] + ['active_vis_%d.npy' % s for s in (1, 2, 3, 4)]:
    act |= np.load(f)
exc = W.copy(); exc.data = np.maximum(exc.data, 0)
inp = np.asarray(exc[act].sum(0)).ravel() / 0.275
ann0 = engine.annotations().set_index('root_id').reindex(ids)
ro = np.load('steering_readout.npz')
keep = act | (inp >= 20) | np.isin(ids, ro['ids']) | ann0.cell_type.isin(VIS).to_numpy()
np.save('browser_keep.npy', keep)
K = int(keep.sum())
full_index = np.flatnonzero(keep).astype(np.int32)
sub = W[keep][:, keep].tocsr()
syn = np.rint(sub.data / 0.275).astype(np.int16)
regions = engine.neuron_regions(ids)
ann = engine.annotations()
sim_ann = ann.set_index('root_id').reindex(ids)
flags = np.zeros(len(ids), np.uint16)
def mark(types, side, bit):
    m = sim_ann.cell_type.isin(types).to_numpy() & (sim_ann.side == side).to_numpy()
    flags[m] |= bit
mark(engine.FOOD_GLOM, 'left', 1); mark(engine.FOOD_GLOM, 'right', 2)
mark(engine.TRAIL_GLOM, 'left', 4); mark(engine.TRAIL_GLOM, 'right', 8)
flags[sim_ann.cell_class.isin(['olfactory', 'ALLN']).to_numpy()] |= 16
flags[(sim_ann.cell_type == 'DNa02').to_numpy() & (sim_ann.side == 'left').to_numpy()] |= 32
flags[(sim_ann.cell_type == 'DNa02').to_numpy() & (sim_ann.side == 'right').to_numpy()] |= 64
mark(VIS, 'left', 128); mark(VIS, 'right', 256)
readout = np.zeros(len(ids), np.float32)
idx = {r: i for i, r in enumerate(ids)}
readout[[idx[r] for r in ro['ids']]] = ro['w']
assert (readout[~keep] == 0).all()
blob = b''.join([np.array([K, sub.nnz, len(engine.REGIONS), 0], np.int32).tobytes(), full_index.tobytes(),
                 regions[keep].astype(np.uint8).tobytes(), flags[keep].tobytes(), readout[keep].tobytes(),
                 sub.indptr.astype(np.int32).tobytes(), sub.indices.astype(np.uint16).tobytes(), syn.tobytes()])
open('../live/brain.b64.txt', 'w').write(base64.b64encode(gzip.compress(blob, 9)).decode())
xyz, reg = brain_geometry()
open('../live/cloud.b64.txt', 'w').write(base64.b64encode(gzip.compress(xyz.tobytes() + reg.tobytes(), 9)).decode())
json.dump({'regions': list(engine.REGIONS), 'K': K, 'E': int(sub.nnz),
           'counts': {k: int(((flags[keep] & b) > 0).sum()) for k, b in [('foodL', 1), ('foodR', 2), ('trailL', 4), ('trailR', 8), ('dep', 16), ('visL', 128), ('visR', 256)]}},
          open('../live/meta.json', 'w'), indent=1)
print(K, sub.nnz, 'max |syn|', np.abs(syn).max())
