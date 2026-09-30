"""Rebuild the public revision offline, preserving frozen inputs and outputs.

This does not run model inference, download datasets or change freeze manifests.
Compilation is optional and uses an existing Tectonic installation/cache.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    commands = [
        ['src/analysis/rebuild_cc_statistics_v4.py'],
        ['src/analysis/canary_controls_v2.py'],
        ['src/analysis/canary_controls_v2.py', '--root', 'results/canary_stability_v2',
         '--out', 'results/canary_stability_v2/analysis'],
        ['src/analysis/audit_canary_raw_v2.py'],
        ['src/analysis/render_canary_stability_v2.py'],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    source = ROOT / 'results/industrial_interface_focused_overleaf'
    if args.compile:
        exe = shutil.which('tectonic')
        if not exe:
            raise RuntimeError('Tectonic not found; numeric reconstruction has completed.')
        for name in ['main.tex', 'supplement.tex']:
            subprocess.run([exe, '--only-cached', name], cwd=source, check=True)
    target = ROOT / 'manuscript'
    target.mkdir(exist_ok=True)
    for p in source.rglob('*'):
        if p.is_file() and p.suffix in {'.tex', '.bib', '.pdf', '.png', '.txt'} and not p.name.endswith('.build.txt'):
            dest = target / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    print('Offline reconstruction complete; manuscript/ mirrors the canonical editorial directory.')

if __name__ == '__main__':
    main()
