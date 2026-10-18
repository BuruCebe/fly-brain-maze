# Top-down animation of a recorded race. usage: python3 render_clip.py rec.json out.mp4 "title" [weak]
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse, Circle

rec = json.load(open(sys.argv[1]))
out, title = sys.argv[2], sys.argv[3]
weak = len(sys.argv) > 4
W, H, c = rec['W'], rec['H'], rec['cell']
grid = np.array(rec['grid']).reshape(H, W)
frames = rec['frames']
n = len(frames[0])
COLORS = ['#d6336c', '#2f7fd1', '#3fa36a', '#8a63c9', '#e07a3a', '#1fa59a']
if not weak: COLORS[0] = '#e6a61c'

fig = plt.figure(figsize=(6, 6.6), dpi=180)
fig.patch.set_facecolor('#f4f1ea')
ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
ax.set_xlim(0, W * c); ax.set_ylim(H * c, 0); ax.set_aspect('equal'); ax.axis('off')
ax.imshow(np.where(grid == 1, 1.0, 0.0), extent=[0, W * c, H * c, 0], cmap=matplotlib.colors.ListedColormap(['#4a5560', '#efe9d8']), interpolation='nearest')
ax.add_patch(Circle(rec['exit'], 1.8, color='#e8890c', zorder=3))
ax.text(rec['exit'][0], rec['exit'][1] - 2.4, 'food', ha='center', va='bottom', fontsize=8, color='#7a4a06', zorder=3)
ax.add_patch(Circle(rec['start'], 1.8, color='#9fb7cf', zorder=2))
fig.text(0.05, 0.955, title, fontsize=12, weight='bold', color='#22262a', va='top')
sub = fig.text(0.05, 0.905, '', fontsize=9, color='#4b5259', va='top')
tails = [ax.plot([], [], '-', lw=1.4, color=COLORS[i % 6], alpha=.55, zorder=4)[0] for i in range(n)]
bodies = [Ellipse((0, 0), 2.6, 1.3, color=COLORS[i % 6], zorder=5) for i in range(n)]
for b in bodies: ax.add_patch(b)
if weak:
    label = ax.text(0, 0, "can't smell food", fontsize=7, color=COLORS[0], weight='bold', zorder=6)
STEP = 2   # draw every 2nd recorded frame (recorded at 25/s) and play at 12.5 fps -> real time

def draw(k):
    f = frames[min(k * STEP, len(frames) - 1)]
    for i, (x, y, h, done) in enumerate(f):
        bodies[i].set_center((x, y)); bodies[i].angle = np.degrees(h)
        lo = max(0, k * STEP - 150)
        pts = np.array([fr[i][:2] for fr in frames[lo:k * STEP + 1]])
        tails[i].set_data(pts[:, 0], pts[:, 1])
    if weak:
        label.set_position((f[0][0] + 2, f[0][1] - 2))
    sub.set_text(f'{k * STEP * 0.04:4.1f} s   spiking connectome brains (FlyWire, 139k neurons)')
    return []

nf = (len(frames) + STEP - 1) // STEP + 12
anim = FuncAnimation(fig, draw, frames=nf, blit=False)
anim.save(out, writer=FFMpegWriter(fps=12.5, bitrate=4000, extra_args=['-pix_fmt', 'yuv420p']))
draw(nf - 1); fig.savefig(out.replace('.mp4', '.png'), facecolor=fig.get_facecolor())
print('saved', out)
