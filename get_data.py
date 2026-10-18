"""Download the free FlyWire data the app needs (about 230 MB) into data/."""
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
DATA.mkdir(exist_ok=True)
REPOS = {
    'Drosophila_brain_model': 'https://github.com/philshiu/Drosophila_brain_model',  # Shiu et al. 2024, MIT
    'flywire_annotations': 'https://github.com/flyconnectome/flywire_annotations',    # Schlegel et al. 2024
}
for name, url in REPOS.items():
    if (DATA / name).exists():
        print(f'{name}: already there')
        continue
    print(f'Downloading {name}...')
    subprocess.run(['git', 'clone', '--depth', '1', url, str(DATA / name)], check=True)
print('Done. Now run: python3 server.py')
