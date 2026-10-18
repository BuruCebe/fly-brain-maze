"""Compiled (numba) version of flybrain.Brain for closed-loop runs.

Same model and parameters as flybrain.py, but the whole inner loop runs as
machine code and a body step (40 brain steps) is one call.
"""
import numpy as np
from numba import njit

from flybrain import V0, VRST, VTH, T_MBR, TAU, T_RFC, T_DLY, W_SYN, F_POI, REST_TOL, connectome
import scipy.sparse as sp


@njit(cache=True)
def _run(n_steps, t0, dt, B, v, g, rfc, x, tlast, bt, bv, bn, live, is_live, n_live,
         indptr, indices, data, dep, U, tau_rec, drv_idx, drv_rate, region, n_regions, dnL, dnR, rec_fly, seed):
    """Event-driven: only elements away from rest (the live list) are updated.
    Spikes are queued per delay slot in bt/bv and land D steps later."""
    np.random.seed(seed)
    D, cap = bt.shape
    kick = W_SYN * F_POI
    reg = np.zeros((B, n_regions))
    dn = np.zeros((B, 2))
    rec = np.zeros(v.shape[0] // B, np.bool_)
    dropped = 0
    for s in range(n_steps):
        t = t0 + s
        slot = t % D
        for e in range(bn[slot]):
            k = bt[slot, e]
            g[k] += bv[slot, e]
            if not is_live[k]:
                is_live[k] = True
                live[n_live] = k
                n_live += 1
        bn[slot] = 0
        # Poisson sensory input, applied straight to v as in the original
        for m in range(drv_idx.shape[0]):
            base = drv_idx[m] * B
            for f in range(B):
                p = drv_rate[m, f] * dt / 1000.0
                if p > 0 and np.random.random() < p:
                    k = base + f
                    v[k] += kick
                    if not is_live[k]:
                        is_live[k] = True
                        live[n_live] = k
                        n_live += 1
        keep = 0
        for i in range(n_live):
            k = live[i]
            if rfc[k] > 0:
                rfc[k] -= dt
                live[keep] = k
                keep += 1
                continue
            vk = v[k] + (V0 - v[k] + g[k]) / T_MBR * dt
            gk = g[k] - g[k] / TAU * dt
            if vk > VTH:
                vk = VRST
                gk = 0.0
                rfc[k] = T_RFC - dt
                n = k // B
                f = k % B
                strength = 1.0
                if dep[n]:  # short-term depression, recovered exactly since the last spike
                    x[k] = 1.0 - (1.0 - x[k]) * np.exp(-(t - tlast[k]) * dt / tau_rec)
                    tlast[k] = t
                    strength = x[k]
                    x[k] -= U * x[k]
                for j in range(indptr[n], indptr[n + 1]):
                    c = bn[slot]
                    if c < cap:
                        bt[slot, c] = indices[j] * B + f
                        bv[slot, c] = data[j] * strength
                        bn[slot] = c + 1
                    else:
                        dropped += 1
                reg[f, region[n]] += 1
                if n == dnL:
                    dn[f, 0] += 1
                elif n == dnR:
                    dn[f, 1] += 1
                if f == rec_fly:
                    rec[n] = True
            v[k] = vk
            g[k] = gk
            if rfc[k] <= 0 and abs(vk - V0) < REST_TOL and abs(gk) < REST_TOL:
                v[k] = V0
                g[k] = 0.0
                is_live[k] = False
            else:
                live[keep] = k
                keep += 1
        n_live = keep
    return reg, dn, rec, n_live, dropped


class FastBrain:
    def __init__(self, n_flies=1, dt=0.25, seed=0, silence=(), depress=(), U=0.5, tau_rec=500.0):
        self.ids, W = connectome()
        self.index = {rid: i for i, rid in enumerate(self.ids)}
        self.N = len(self.ids)
        if len(silence):  # cut every synapse to and from these neurons
            keep = sp.diags((~np.isin(self.ids, silence)).astype(np.float32))
            W = (keep @ W @ keep).tocsr()
            W.eliminate_zeros()
        self.W = W
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int64)
        self.data = W.data.astype(np.float32)
        self.B, self.dt = n_flies, dt
        self.dep = np.isin(self.ids, depress)
        self.U, self.tau_rec = U, tau_rec
        self.delay = int(round(T_DLY / dt))
        self.seed = seed
        n = self.N * self.B  # element k = neuron k // B, fly k % B
        self.v = np.full(n, V0, np.float32)
        self.g = np.zeros(n, np.float32)
        self.rfc = np.zeros(n, np.float32)
        self.x = np.ones(n, np.float32)
        self.tlast = np.zeros(n, np.int64)
        cap = 400_000 * n_flies  # queued synaptic events per delay slot
        self.bt = np.zeros((self.delay, cap), np.int64)
        self.bv = np.zeros((self.delay, cap), np.float32)
        self.bn = np.zeros(self.delay, np.int64)
        self.live = np.zeros(n, np.int64)
        self.is_live = np.zeros(n, np.bool_)
        self.n_live = 0
        self.dropped = 0
        self.t = 0

    def ids_to_idx(self, root_ids):
        return np.array([self.index[r] for r in root_ids if r in self.index], np.int64)

    def run(self, n_steps, drive, region, n_regions, dnL, dnR, rec_fly=-1):
        """drive: list of (neuron idx array, per-fly rates in Hz). Returns spike
        counts per fly and region, DNa02 left/right spike counts per fly, and a
        bool per neuron saying whether it spiked in rec_fly."""
        idx = np.concatenate([d[0] for d in drive]).astype(np.int64)
        rate = np.concatenate([np.broadcast_to(np.asarray(r, np.float64), (len(i), self.B)) for i, r in drive])
        self.seed += 1
        reg, dn, rec, self.n_live, dropped = _run(
            n_steps, self.t, self.dt, self.B, self.v, self.g, self.rfc, self.x, self.tlast,
            self.bt, self.bv, self.bn, self.live, self.is_live, self.n_live,
            self.indptr, self.indices, self.data, self.dep, self.U, self.tau_rec,
            idx, np.ascontiguousarray(rate), region, n_regions, dnL, dnR, rec_fly, self.seed)
        self.dropped += dropped
        self.t += n_steps
        return reg, dn, rec
