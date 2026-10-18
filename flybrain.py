"""Whole-brain spiking model of the adult fruit fly, batched over many flies.

A numpy/scipy port of the leaky integrate-and-fire model in Shiu et al. 2024
(Nature, github.com/philshiu/Drosophila_brain_model, MIT), run on the FlyWire
v783 connectome (CC BY 4.0). Same parameters as the original; the only change
is that B independent brains are stepped together so a whole pack can share
one simulation loop.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent / "data" / "Drosophila_brain_model"

# Shiu et al. 2024 defaults (ms, mV)
V0, VRST, VTH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN = 0.275
F_POI = 250  # Poisson input kick = W_SYN * F_POI
REST_TOL = 0.05  # mV; closer than this to rest counts as resting and stops being updated


_CACHE = {}


def connectome(w_syn=W_SYN):
    """FlyWire neuron ids and the synapse matrix (rows = presynaptic), loaded once."""
    if w_syn not in _CACHE:
        comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
        ids = comp.index.to_numpy()
        con = pd.read_parquet(DATA / "Connectivity_783.parquet")
        W = sp.csr_matrix(
            (con["Excitatory x Connectivity"].to_numpy(np.float32) * w_syn,
             (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy())),
            shape=(len(ids), len(ids)))
        _CACHE[w_syn] = ids, W
    return _CACHE[w_syn]


class Brain:
    def __init__(self, n_flies=1, dt=0.1, seed=0, w_syn=W_SYN, silence=(), depress=(), U=0.1, tau_rec=200.0):
        self.ids, W = connectome(w_syn)
        self.index = {rid: i for i, rid in enumerate(self.ids)}
        self.N = len(self.ids)
        if len(silence):  # cut every synapse to and from these neurons
            keep = sp.diags((~np.isin(self.ids, silence)).astype(np.float32))
            W = (keep @ W @ keep).tocsr()
            W.eliminate_zeros()
        self.W = W
        self.B, self.dt = n_flies, dt
        # short-term depression: these neurons' synapses weaken as they keep firing
        self.dep = np.isin(self.ids, depress)
        self.dep_flat = np.flatnonzero(np.repeat(self.dep, n_flies))
        self.U, self.tau_rec = U, tau_rec
        self.rng = np.random.default_rng(seed)
        self.delay = int(round(T_DLY / dt))
        self.reset()

    def reset(self):
        n = self.N * self.B  # state is flat: element k = neuron k // B, fly k % B
        self.v = np.full(n, V0, np.float32)
        self.g = np.zeros(n, np.float32)
        self.rfc = np.zeros(n, np.float32)  # ms of refractory left
        # synaptic input in flight, one (indices, values) pair per delay slot
        self.buf = [(np.zeros(0, int), np.zeros(0, np.float32))] * self.delay
        self.is_live = np.zeros(n, bool)  # anything away from rest
        self.live = np.zeros(0, int)
        self.x = np.ones(self.N * self.B, np.float32)  # synaptic resources left
        self.t = 0

    def ids_to_idx(self, root_ids):
        return np.array([self.index[r] for r in root_ids if r in self.index], int)

    def _wake(self, k):
        k = np.unique(k[~self.is_live[k]])
        self.is_live[k] = True
        self.live = np.concatenate([self.live, k])

    def step(self, drive):
        """Advance one dt.

        drive: list of (idx_array, rates_Hz) pairs giving Poisson input
        rates, rates either a scalar or one value per fly. Returns the flat
        indices (neuron * B + fly) that spiked this step.

        Most of the brain sits at rest at any moment, so only elements that
        have received input since they last settled are updated.
        """
        dt, B = self.dt, self.B
        slot = self.t % self.delay
        inc, val = self.buf[slot]
        np.add.at(self.g, inc, val)
        self._wake(inc)

        # Poisson sensory input, applied straight to v as in the original
        for idx, rate in drive:
            p = np.broadcast_to(np.asarray(rate, np.float32) * dt / 1000.0, (B,))
            hit = self.rng.random((len(idx), B)) < p
            k = (idx[:, None] * B + np.arange(B))[hit]
            self.v[k] += W_SYN * F_POI
            self._wake(k)

        k = self.live
        v, g, r = self.v[k], self.g[k], self.rfc[k]
        on = r <= 0
        v = np.where(on, v + (V0 - v + g) / T_MBR * dt, v)
        g = np.where(on, g - g / TAU * dt, g)
        spk = v > VTH
        v[spk] = VRST
        g[spk] = 0
        r = np.where(spk, T_RFC, r - dt)
        rest = (np.abs(v - V0) < REST_TOL) & (np.abs(g) < REST_TOL) & (r <= 0)
        v[rest], g[rest] = V0, 0
        self.v[k], self.g[k], self.rfc[k] = v, g, r
        self.is_live[k[rest]] = False
        self.live = k[~rest]

        fired = k[spk]
        d = self.dep_flat
        self.x[d] += (1 - self.x[d]) * dt / self.tau_rec
        strength = self.x[fired].copy()
        self.x[fired] -= self.U * self.x[fired] * self.dep[fired // B]
        out = (np.zeros(0, int), np.zeros(0, np.float32))
        if len(fired):
            # gather each spiking neuron's row of the synapse matrix
            pre, fly = fired // B, fired % B
            starts = self.W.indptr[pre]
            lens = self.W.indptr[pre + 1] - starts
            seg = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
            out = (self.W.indices[seg].astype(np.int64) * B + np.repeat(fly, lens),
                   self.W.data[seg] * np.repeat(strength, lens))
        self.buf[slot] = out  # slot frees up now and lands again after `delay` steps
        self.t += 1
        return fired
