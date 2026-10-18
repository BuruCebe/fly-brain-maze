"""Turn the open-loop recordings into steering_readout.npz used by engine.py.

Picks the 40 descending neurons whose firing best separates smell-on-the-left
from smell-on-the-right (on seeds 101 and 102), weights them, and removes any
part of the readout that responds to overall activity, so a fly smelling
nothing or smelling both sides evenly gets no turning signal.
"""
import numpy as np

c1, c2 = [np.load(f'calib_{s}.npz', allow_pickle=True) for s in (101, 102)]
ids, ks = c1['dn_ids'], c1['ks']
X = np.concatenate([c['counts'][10:].reshape(-1, len(ids)) for c in (c1, c2)]) * 10.0
K = np.concatenate([np.tile(ks, c['counts'][10:].shape[0]) for c in (c1, c2)])
left, right, even = X[K > 0], X[K < 0], X[K == 0]
diff = left.mean(0) - right.mean(0)
sd = np.sqrt((left.var(0) + right.var(0)) / 2) + 1.0
top = np.argsort(-np.abs(diff / sd))[:40]
w = np.zeros(len(ids))
w[top] = diff[top] / sd[top] ** 2
# Symmetric input should not turn the fly. Remove the readout's response to even smell
# and to seeing flies on both sides (calibrate_vis.py); silence reads zero already.
v1, v2 = [np.load(f'calibvis_{s}.npz', allow_pickle=True) for s in (201, 202)]
assert (v1['dn_ids'] == ids).all()
V = np.concatenate([c['counts'][10:].reshape(-1, len(ids)) for c in (v1, v2)]) * 10.0
VK = np.concatenate([np.tile(v1['ks'], c['counts'][10:].shape[0]) for c in (v1, v2)])
mask = w != 0
B = np.stack([even.mean(0) * mask, V[VK == 0].mean(0) * mask], 1)
w -= B @ np.linalg.lstsq(B, w, rcond=None)[0]
w /= (left.mean(0) - right.mean(0)) @ w / 2       # full left contrast reads +1, full right -1
keep = w != 0
np.savez('steering_readout.npz', ids=ids[keep], w=w[keep])
print(keep.sum(), 'neurons; even smell', round(float(even.mean(0) @ w), 3), 'flies seen both sides', round(float(V[VK == 0].mean(0) @ w), 3))
print('seen left (train)', round(float(V[VK > 0].mean(0) @ w), 2), 'seen right (train)', round(float(V[VK < 0].mean(0) @ w), 2))
t = np.load('calibvis_203.npz', allow_pickle=True); T = t['counts'][10:] * 10.0
for k, name in ((1, 'left'), (0, 'both'), (-1, 'right')):
    print('held-out seen', name, round(float((T[:, t['ks'] == k] @ w[keep] if False else T[:, t['ks'] == k].mean((0, 1)) @ w)), 2))
c3 = np.load('calib_103.npz', allow_pickle=True); F = c3['counts'][10:] * 10.0
for k, name in ((1, 'left'), (0, 'even'), (-1, 'right')):
    print('held-out food', name, round(float(F[:, c3['ks'] == k].mean((0, 1)) @ w), 2))
