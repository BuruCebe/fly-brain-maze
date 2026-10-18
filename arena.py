"""2D maze for simulated flies: walls, a food odor that spreads from the exit
along the corridors, and an odor trail each fly leaves behind.

Units are millimetres and seconds. A fruit fly is about 2.5 mm long and
walks at up to ~20 mm/s.
"""
from collections import deque

import numpy as np

CELL = 10.0  # mm per maze cell


def make_maze(w, h, seed=0):
    """Random perfect maze as a grid of open (True) / wall (False) pixels,
    each maze cell becoming 2x2 pixels plus walls."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((2 * h + 1, 2 * w + 1), bool)
    seen = np.zeros((h, w), bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    grid[1, 1] = True
    while stack:
        y, x = stack[-1]
        nbrs = [(y + dy, x + dx) for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= y + dy < h and 0 <= x + dx < w and not seen[y + dy, x + dx]]
        if not nbrs:
            stack.pop()
            continue
        ny, nx = nbrs[rng.integers(len(nbrs))]
        grid[2 * ny + 1, 2 * nx + 1] = True
        grid[y + ny + 1, x + nx + 1] = True
        seen[ny, nx] = True
        stack.append((ny, nx))
    return grid


class Arena:
    def __init__(self, w=4, h=4, seed=0, res=1.0, odor_len=60.0, trail_tau=20.0):
        self.grid = make_maze(w, h, seed)
        self.px = CELL / 2  # mm per grid square
        self.res = res      # mm per odor/trail pixel
        H, W = self.grid.shape
        self.size = (W * self.px, H * self.px)  # mm (x, y)
        ny, nx = int(self.size[1] / res), int(self.size[0] / res)
        yy, xx = np.mgrid[0:ny, 0:nx]
        self.open = self.grid[(yy * res / self.px).astype(int), (xx * res / self.px).astype(int)]
        self.start = np.array([1.5 * self.px, 1.5 * self.px])
        self.exit = np.array([(W - 1.5) * self.px, (H - 1.5) * self.px])
        # food odor: falls off with walking distance to the exit, not straight-line distance
        dist = self._walk_dist(self.exit)
        self.odor = np.where(np.isfinite(dist), np.exp(-dist / odor_len), 0.0)
        self.trail = np.zeros_like(self.odor)
        self.trail_tau = trail_tau

    def _walk_dist(self, p):
        ny, nx = self.open.shape
        d = np.full((ny, nx), np.inf)
        y0, x0 = int(p[1] / self.res), int(p[0] / self.res)
        d[y0, x0] = 0
        q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = y + dy, x + dx
                if 0 <= a < ny and 0 <= b < nx and self.open[a, b] and d[a, b] == np.inf:
                    d[a, b] = d[y, x] + self.res
                    q.append((a, b))
        return d

    def _pix(self, pts):
        pts = np.atleast_2d(pts)
        ix = np.clip((pts[:, 0] / self.res).astype(int), 0, self.open.shape[1] - 1)
        iy = np.clip((pts[:, 1] / self.res).astype(int), 0, self.open.shape[0] - 1)
        return iy, ix

    def is_open(self, pts):
        return self.open[self._pix(pts)]

    def sample(self, field, pts):
        return field[self._pix(pts)]

    def move(self, pos, new):
        """Walls stop the fly; it slides along them if one axis is still free."""
        out = pos.copy()
        ok = self.is_open(new)
        out[ok] = new[ok]
        tx = np.column_stack([new[:, 0], pos[:, 1]])
        ty = np.column_stack([pos[:, 0], new[:, 1]])
        okx = ~ok & self.is_open(tx)
        oky = ~ok & ~okx & self.is_open(ty)
        out[okx] = tx[okx]
        out[oky] = ty[oky]
        return out, ~ok

    def deposit(self, pos, amount, dt):
        self.trail *= np.exp(-dt / self.trail_tau)
        iy, ix = self._pix(pos)
        np.add.at(self.trail, (iy, ix), amount)

    def at_exit(self, pos):
        return np.linalg.norm(pos - self.exit, axis=1) < self.px * 0.8
