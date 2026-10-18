"""Closed loop: a pack of FlyWire brains walking a maze.

Each fly smells the food odor at its left and right antenna, plus (in pack
mode) the trail other flies leave behind. Those smells drive the fly's real
olfactory receptor neurons, the whole-brain model runs, and the left and
right DNa02 descending neurons (which steer walking in real flies,
Rayshubskiy et al. 2020) set how the fly turns. Forward speed is fixed; the
brain only steers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from arena import Arena
from fastbrain import FastBrain

ANN_PATH = (Path(__file__).resolve().parent / 'data' / 'flywire_annotations' / 'supplemental_files'
            / 'Supplemental_file1_neuron_annotations.tsv')
_ANN = None


def annotations():
    global _ANN
    if _ANN is None:
        _ANN = pd.read_csv(ANN_PATH, sep='\t', low_memory=False)
    return _ANN


FOOD_GLOM = ['ORN_DM1', 'ORN_DM4', 'ORN_VA2', 'ORN_VM7d', 'ORN_DM2', 'ORN_DP1m']  # vinegar-tuned
TRAIL_GLOM = ['ORN_DA1', 'ORN_VA1v', 'ORN_VA1d']  # fly pheromone (cVA and others)

# Brain regions shown in the UI and offered for silencing, as annotation filters
REGIONS = {
    'Smell receptors':     lambda a: a.cell_class == 'olfactory',
    'Antennal lobe':       lambda a: a.cell_class.isin(['ALLN', 'ALPN', 'ALIN', 'ALON']),
    'Mushroom body':       lambda a: a.cell_class.isin(['Kenyon_Cell', 'MBON', 'DAN', 'MBIN']),
    'Lateral horn':        lambda a: a.cell_class.isin(['LHLN', 'LHCENT']),
    'Central complex':     lambda a: a.cell_class == 'CX',
    'Descending neurons':  lambda a: a.super_class == 'descending',
    'Steering (DNa02)':    lambda a: a.cell_type == 'DNa02',
    'Other central brain': lambda a: a.super_class == 'central',
    'Optic lobes':         lambda a: a.super_class.isin(['optic', 'visual_projection', 'visual_centrifugal']),
    'Other senses':        lambda a: a.super_class.isin(['sensory', 'sensory_ascending']),
    'Motor / other':       lambda a: pd.Series(True, index=a.index),
}

SPEED = 8.0          # mm/s
ANT_OFFSET = 1.0     # mm, antenna sample points left/right of the head
ANT_AHEAD = 1.0      # mm ahead of the body centre
BASE_HZ = 12.0       # ORN rate with no left/right contrast
CONTRAST_AMP = 25.0  # amplification of the left/right odor difference
TAU_DN = 0.3         # s, smoothing of descending neuron rates
STEER_GAIN = 6.0     # rad/s of turning at a steering reading of 1 (full smell contrast)
# Steering is read from 40 descending neurons (DNa02, DNa01, DNa04 and others)
# whose firing tells left smell from right smell; see make_readout.py
READOUT = Path(__file__).resolve().parent / 'steering_readout.npz'
TURN_NOISE = 1.5     # rad/s^0.5, wandering that real walking flies show


def neuron_regions(ids):
    """Region index (into REGIONS) for each neuron id; first matching filter wins."""
    a = annotations().set_index('root_id').reindex(ids).reset_index()
    out = np.full(len(ids), len(REGIONS) - 1, np.uint8)
    unset = np.ones(len(ids), bool)
    for r, f in enumerate(REGIONS.values()):
        m = f(a).fillna(False).to_numpy(bool) & unset
        out[m] = r
        unset &= ~m
    return out


class Simulation:
    def __init__(self, n_flies=1, maze=(3, 3), maze_seed=0, seed=0, smell=True, contrast=10.0,
                 pack_trail=False, trail_gain=1.0, silence=(), dt=0.25, body_dt=0.01, steer_gain=STEER_GAIN):
        ann = annotations()
        self.rng = np.random.default_rng(seed)
        self.arena = Arena(*maze, seed=maze_seed)
        ids_all = ann.root_id.to_numpy()
        reg_all = neuron_regions(ids_all)
        names = list(REGIONS)
        silenced = ids_all[np.isin(reg_all, [names.index(s) for s in silence])]
        al = ann[ann.cell_class.isin(['olfactory', 'ALLN'])].root_id.to_numpy()
        self.brain = FastBrain(n_flies=n_flies, dt=dt, seed=seed, depress=al, U=0.5, tau_rec=500.,
                               silence=silenced)
        b = self.brain
        self.regions = neuron_regions(b.ids).astype(np.int64)
        ro = np.load(READOUT)
        self.ro_idx = b.ids_to_idx(ro['ids'])
        self.ro_w = ro['w']
        # spikes of readout neurons are counted per neuron, after the display regions
        self.count_map = self.regions.copy()
        self.count_map[self.ro_idx] = len(REGIONS) + np.arange(len(self.ro_idx))
        self.region_size = np.bincount(self.regions, minlength=len(REGIONS))

        def side_idx(types, side):
            return b.ids_to_idx(ann[ann.cell_type.isin(types) & (ann.side == side)].root_id)
        self.food = side_idx(FOOD_GLOM, 'left'), side_idx(FOOD_GLOM, 'right')
        self.trail_orn = side_idx(TRAIL_GLOM, 'left'), side_idx(TRAIL_GLOM, 'right')
        dn = ann[ann.cell_type == 'DNa02'].set_index('side').root_id
        self.dnL, self.dnR = b.index[dn['left']], b.index[dn['right']]

        B = self.B = n_flies
        self.smell, self.contrast = smell, contrast
        self.pack_trail, self.trail_gain = pack_trail, trail_gain
        self.body_dt = body_dt
        self.steer_gain = steer_gain
        self.steps_per_body = int(round(body_dt * 1000 / dt))
        self.pos = np.tile(self.arena.start, (B, 1)) + self.rng.normal(0, 0.3, (B, 2))
        self.head = self.rng.uniform(0, 2 * np.pi, B)
        self.rL = np.zeros(B)  # smoothed DNa02 rates, for display
        self.rR = np.zeros(B)
        self.steer = np.zeros(B)  # smoothed steering reading: +1 turn left, -1 turn right
        self.done = np.full(B, np.nan)
        self.t = 0.0

    def _antennae(self, field):
        fwd = np.column_stack([np.cos(self.head), np.sin(self.head)])
        left = np.column_stack([-np.sin(self.head), np.cos(self.head)])
        tip = self.pos + ANT_AHEAD * fwd
        return (self.arena.sample(field, tip + ANT_OFFSET * left),
                self.arena.sample(field, tip - ANT_OFFSET * left))

    def _orn_rates(self, cL, cR, gain=1.0):
        """Smell at each antenna -> ORN firing rate on that side. The odor differs
        by only a few percent across a 2 mm head, so the left/right difference
        is amplified (CONTRAST_AMP) before it reaches the receptors."""
        k = np.tanh(CONTRAST_AMP * (cL - cR) / (cL + cR + 1e-9))
        level = np.clip(np.maximum(cL, cR) * 4 * gain, 0, 1)  # fades out far from the source
        return level * (BASE_HZ + self.contrast * k), level * (BASE_HZ - self.contrast * k)

    def step(self, record_fly=None):
        """Advance one body step. Returns per-fly region spike counts (B, R) and,
        if record_fly is given, the neuron indices that fly spiked."""
        B = self.B
        moving = np.isnan(self.done)
        drive = []
        cL, cR = self._antennae(self.arena.odor)
        if not self.smell:
            cL, cR = cL * 0, cR * 0
        hzL, hzR = self._orn_rates(cL, cR)
        hzL[~moving] = hzR[~moving] = 0
        drive += [(self.food[0], hzL), (self.food[1], hzR)]
        if self.pack_trail:
            tL, tR = self._antennae(self.arena.trail)
            pL, pR = self._orn_rates(tL, tR, self.trail_gain)
            pL[~moving] = pR[~moving] = 0
            drive += [(self.trail_orn[0], pL), (self.trail_orn[1], pR)]

        nR_ = len(REGIONS)
        cnt, dn, rec = self.brain.run(self.steps_per_body, drive, self.count_map, nR_ + len(self.ro_idx),
                                      self.dnL, self.dnR, -1 if record_fly is None else record_fly)
        reg = cnt[:, :nR_].copy()
        np.add.at(reg.T, self.regions[self.ro_idx], cnt[:, nR_:].T)
        a = self.body_dt / TAU_DN
        self.rL += a * (dn[:, 0] / self.body_dt - self.rL)
        self.rR += a * (dn[:, 1] / self.body_dt - self.rR)
        reading = (cnt[:, nR_:] / self.body_dt) @ self.ro_w
        self.steer += a * (reading - self.steer)

        # heading is counter-clockwise, so a positive reading turns the fly left
        turn = self.steer_gain * self.steer
        turn += TURN_NOISE * self.rng.normal(size=B) / np.sqrt(self.body_dt)
        self.head[moving] += turn[moving] * self.body_dt
        new = self.pos + SPEED * self.body_dt * np.column_stack([np.cos(self.head), np.sin(self.head)])
        new[~moving] = self.pos[~moving]
        self.pos, bumped = self.arena.move(self.pos, new)
        hit = bumped & moving
        self.head[hit] += self.rng.choice([-1, 1], hit.sum()) * np.pi / 2  # turn away from walls
        if self.pack_trail:
            self.arena.deposit(self.pos[moving], 0.05, self.body_dt)
        self.t += self.body_dt
        self.done[self.arena.at_exit(self.pos) & moving] = self.t
        out = {'reg': reg / self.body_dt / np.maximum(self.region_size, 1)}  # mean Hz per neuron
        if record_fly is not None:
            out['spiked'] = np.flatnonzero(rec)
        return out
