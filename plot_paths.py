"""Draw maze runs: walls, food smell, and each fly's path."""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arena import Arena


def draw(ax, arena, df, done, title):
    ext = [0, arena.size[0], arena.size[1], 0]
    ax.imshow(np.where(arena.open, arena.odor, np.nan), extent=ext, cmap='YlGn', vmin=0, vmax=1.3)
    ax.imshow(np.where(arena.open, np.nan, 1.0), extent=ext, cmap='gray_r', vmin=0, vmax=1.2)
    for i, g in df.groupby('fly'):
        ax.plot(g.x, g.y, lw=1.2, color=plt.cm.tab10(i % 10))
    ax.plot(*arena.start, 'o', color='#2b6cb0', ms=7)
    ax.plot(*arena.exit, '*', color='#c05621', ms=14)
    ok = np.isfinite(done)
    sub = f'reached food: {ok.sum()}/{len(done)}' + (f', {np.nanmean(done):.1f} s avg' if ok.any() else '')
    ax.set_title(f'{title}\n{sub}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':
    names = sys.argv[1:-1]
    out = sys.argv[-1]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.4))
    for ax, n in zip(np.atleast_1d(axes), names):
        draw(ax, Arena(3, 3, seed=0), pd.read_parquet(f'{n}.parquet'), np.load(f'{n}_done.npy'), n)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
