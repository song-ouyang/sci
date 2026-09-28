"""Offline review rebuild: no model calls, downloads, protocol creation or re-freezing.

Use a Python environment with requirements-revision-v3.txt and cached Tectonic.
This command rebuilds derived scores and synchronized manuscript sources/PDF.
"""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
for command in [
    ['src/analysis/score_revision_v3.py'],
    ['src/analysis/export_revision_v3_cases.py'],
    ['src/experiments/mechanism_extension.py','score'],
    ['src/analysis/export_mechanism_cases.py'],
    ['src/analysis/render_revision_v3_paper.py'],
]:
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
subprocess.run(['tectonic','--only-cached','--keep-logs','main_inline.tex'],cwd=ROOT/'paper',check=True)
print('Offline rebuild complete. No inference was invoked; inspect the generated PDF before releasing changes.')
