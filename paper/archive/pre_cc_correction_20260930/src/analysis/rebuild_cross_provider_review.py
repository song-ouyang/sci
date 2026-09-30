"""Offline rebuild of four evidence stages: no inference, downloads or re-freezing."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]

def main():
    for command in [
        ['src/analysis/score_revision_v3.py'],
        ['src/analysis/export_revision_v3_cases.py'],
        ['src/experiments/mechanism_extension.py', 'score'],
        ['src/analysis/export_mechanism_cases.py'],
        ['src/analysis/audit_domestic_extension.py', 'score', '--base', 'results/domestic_extension_v2'],
        ['src/experiments/independent_validation.py', 'score'],
        ['src/analysis/verify_independent_results.py'],
        ['src/analysis/render_revision_v3_paper.py'],
    ]:
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    subprocess.run(['tectonic','--only-cached','--keep-logs','main_inline.tex'],cwd=ROOT/'paper',check=True)
    print('Offline rebuild complete; no model calls, data downloads or new freezes.')

if __name__=='__main__':
    main()
