"""Fly Brain Maze: local web app.

    python3 server.py        then open http://localhost:8000

Runs the maze simulation in the background and streams it to the browser.
Every run is also saved under runs/ so it can be replayed at full speed.
"""
import gzip
import json
import queue
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

import engine
from arena import Arena
from flybrain import connectome

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
RUNS.mkdir(exist_ok=True)


def brain_geometry():
    """Soma position (scaled to about -1..1) and region of every neuron, in model order."""
    ids, _ = connectome()
    ann = engine.annotations().set_index('root_id').reindex(ids)
    xyz = np.column_stack([ann.soma_x.fillna(ann.pos_x) * 4, ann.soma_y.fillna(ann.pos_y) * 4,
                           ann.soma_z.fillna(ann.pos_z) * 40]).astype(np.float32)  # voxels -> nm
    xyz = np.nan_to_num(xyz, nan=float(np.nanmean(xyz)))
    xyz -= xyz.mean(0)
    xyz /= np.abs(xyz).max()
    xyz[:, 1] *= -1  # FlyWire y points down
    return xyz, engine.neuron_regions(ids)


class Runner:
    def __init__(self):
        self.sim = None
        self.thread = None
        self.stop_flag = threading.Event()
        self.paused = threading.Event()
        self.focus = 0
        self.listeners = []
        self.lock = threading.Lock()
        self.status = {'state': 'idle'}
        self.header = None

    def broadcast(self, msg):
        data = json.dumps(msg, separators=(',', ':'))
        with self.lock:
            for q in self.listeners:
                q.put(data)

    def start(self, p):
        self.stop()
        self.stop_flag.clear()
        self.paused.clear()
        self.focus = 0
        self.thread = threading.Thread(target=self._run, args=(p,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.paused.clear()
        if self.thread:
            self.thread.join()

    def _run(self, p):
        self.status = {'state': 'loading'}
        self.broadcast({'type': 'status', **self.status})
        maze = (int(p.get('maze_w', 3)), int(p.get('maze_h', 3)))
        sim = engine.Simulation(
            n_flies=int(p.get('flies', 1)), maze=maze, maze_seed=int(p.get('maze_seed', 0)),
            seed=int(p.get('seed', 0)), smell=bool(p.get('smell', True)),
            contrast=float(p.get('contrast', 10)), pack_trail=bool(p.get('trail', False)),
            silence=p.get('silence', []))
        self.sim = sim
        t_max = float(p.get('t_max', 60))
        name = time.strftime('%Y%m%d-%H%M%S')
        header = {'type': 'start', 'name': name, 'params': p, 'maze': maze_info(sim.arena),
                  'flies': sim.B, 'regions': list(engine.REGIONS)}
        self.header = header
        self.broadcast(header)
        rec = gzip.open(RUNS / f'{name}.jsonl.gz', 'wt')
        rec.write(json.dumps(header) + '\n')
        self.status = {'state': 'running', 'name': name}
        t0 = time.time()
        while not self.stop_flag.is_set() and sim.t < t_max and np.isnan(sim.done).any():
            if self.paused.is_set():
                time.sleep(0.1)
                continue
            focus = min(self.focus, sim.B - 1)
            out = sim.step(record_fly=focus)
            frame = {
                'type': 'frame', 't': round(sim.t, 3), 'wall': round(time.time() - t0, 1),
                'pos': np.round(sim.pos, 2).tolist(), 'head': np.round(sim.head, 3).tolist(),
                'done': [None if np.isnan(d) else round(d, 2) for d in sim.done],
                'dna02': np.round(np.column_stack([sim.rL, sim.rR]), 1).tolist(),
                'reg': np.round(out['reg'], 2).tolist(), 'focus': focus,
                'spiked': out['spiked'].tolist(),
            }
            self.broadcast(frame)
            rec.write(json.dumps(frame, separators=(',', ':')) + '\n')
        end = {'type': 'end', 'name': name, 'done': [None if np.isnan(d) else round(d, 2) for d in sim.done],
               't': round(sim.t, 2)}
        rec.write(json.dumps(end) + '\n')
        rec.close()
        self.broadcast(end)
        self.status = {'state': 'idle', 'last': name}


def maze_info(arena):
    return {'grid': arena.grid.astype(int).tolist(), 'px': arena.px, 'size': arena.size,
            'start': arena.start.tolist(), 'exit': arena.exit.tolist(),
            'odor': np.round(np.where(arena.open, arena.odor, -1), 3).tolist(), 'res': arena.res}


RUNNER = Runner()
GEOM = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype='application/json', code=200, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            self._send((HERE / 'web' / 'index.html').read_bytes(), 'text/html')
        elif path.startswith('/vendor/'):
            f = HERE / 'web' / 'vendor' / Path(path).name
            if not f.exists():
                return self._send('{}', code=404)
            self._send(f.read_bytes(), 'text/javascript')
        elif path == '/brain.bin':
            xyz, reg = GEOM
            self._send(xyz.tobytes() + reg.tobytes(), 'application/octet-stream')
        elif path == '/meta':
            self._send(json.dumps({'n': len(GEOM[1]), 'regions': list(engine.REGIONS),
                                   'status': RUNNER.status, 'maze': maze_info(Arena(3, 3, seed=0))}))
        elif path == '/runs':
            self._send(json.dumps(sorted((p.name[:-9] for p in RUNS.glob('*.jsonl.gz')), reverse=True)))
        elif path.startswith('/runs/'):
            f = RUNS / (Path(path).name + '.jsonl.gz')
            if not f.exists():
                return self._send('{}', code=404)
            self._send(f.read_bytes(), 'application/x-ndjson', extra={'Content-Encoding': 'gzip'})
        elif path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            q = queue.Queue(maxsize=2000)
            with RUNNER.lock:
                RUNNER.listeners.append(q)
            try:
                self.wfile.write(f'data: {json.dumps({"type": "status", **RUNNER.status})}\n\n'.encode())
                if RUNNER.status.get('state') == 'running' and RUNNER.header:  # joined mid-run
                    self.wfile.write(f'data: {json.dumps(RUNNER.header)}\n\n'.encode())
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = None
                    self.wfile.write((f'data: {msg}\n\n' if msg else ': ping\n\n').encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with RUNNER.lock:
                    RUNNER.listeners.remove(q)
        else:
            self._send('{}', code=404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/start':
            RUNNER.start(body)
        elif self.path == '/stop':
            RUNNER.stop()
        elif self.path == '/pause':
            (RUNNER.paused.clear if RUNNER.paused.is_set() else RUNNER.paused.set)()
        elif self.path == '/focus':
            RUNNER.focus = int(body.get('fly', 0))
        self._send(json.dumps({'ok': True, 'paused': RUNNER.paused.is_set()}))


if __name__ == '__main__':
    print('Loading the FlyWire connectome...')
    GEOM = brain_geometry()
    print('Ready: open http://localhost:8000')
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
