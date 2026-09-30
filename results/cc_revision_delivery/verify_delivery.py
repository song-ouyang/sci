"""Internal independently implemented extracted-package verification, not independent human peer review. Run ONLY after final freeze/package."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def snapshot(folder, suffixes):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file() and p.suffix in suffixes}

def normalize_json(value):
    # Actual request timestamps, elapsed durations and IDs are preserved. Only
    # newly written provenance wall-clock fields are excluded from comparison.
    if isinstance(value, dict):
        return {k: normalize_json(v) for k, v in value.items() if k not in {'created_utc', 'generated_at'}}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    return value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--interpreter', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'package': str(args.package.resolve()),
              'package_sha256': sha(args.package), 'steps': []}
    with zipfile.ZipFile(args.package) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for name in names:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe ZIP path')
        manifest = json.loads(z.read('industrial_interface_review/CC_PACKAGE_MANIFEST.json'))
        for name, expected in manifest.items():
            got = hashlib.sha256(z.read('industrial_interface_review/' + name)).hexdigest()
            if got != expected:
                raise ValueError('Package manifest mismatch: ' + name)
        z.extractall(dest)
    root = dest / 'industrial_interface_review'
    report['package_manifest_files_verified'] = len(manifest)
    source_dirs = ['results/statistics_v2', 'results/error_structure_v3', 'results/canary_controls_v2',
                   'results/canary_stability_v2/analysis']
    before = {f'{d}/{p}': b for d in source_dirs for p, b in snapshot(root / d, {'.csv', '.json'}).items()}
    tex_dirs = ['results/industrial_interface_focused_overleaf', 'results/qrei_submission']
    before_tex = {f'{d}/{p.name}': p.read_bytes() for d in tex_dirs for p in (root / d).glob('*.tex')}
    frozen = {}
    for directory in ['independent_validation', 'context_blocked_v1', 'hydraulic_validation_v1',
                      'hydraulic_validation_v1_completion', 'policy_hydraulic_v1', 'policy_paderborn_v1',
                      'canary_hydraulic_v1', 'canary_stability_v2']:
        base = root / 'results' / directory
        decl = json.loads((base / 'predictions/frozen_manifest.json').read_text())
        protocol = json.loads((base / 'protocol.json').read_text())
        checks = {base / 'predictions' / name: wanted for name, wanted in decl['files'].items()}
        checks[base / 'protocol.json'] = decl['protocol_sha256']
        checks.update({root / name: wanted for name, wanted in protocol['code_sha256'].items()})
        for path, expected in checks.items():
            if sha(path) != expected:
                raise ValueError('Frozen mismatch before rebuilding: ' + str(path))
            frozen[str(path.relative_to(root))] = expected
    report['unique_frozen_files_verified'] = len(frozen)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=str(root), PYTHONNOUSERSITE='1')
    commands = [
        ['src/analysis/rebuild_cc_statistics_v4.py'],
        ['src/analysis/canary_controls_v2.py'],
        ['src/analysis/canary_controls_v2.py', '--root', 'results/canary_stability_v2', '--out', 'results/canary_stability_v2/analysis'],
        ['src/analysis/audit_canary_raw_v2.py'],
        ['src/analysis/render_canary_stability_v2.py'],
        ['paper/reviews/cc_q2_2026-09-30/statistical_recomputations.py'],
        ['src/analysis/sync_cc_submission.py', '--compile'],
    ]
    for index, command in enumerate(commands):
        # Preserve the virtualenv launcher symlink: resolving it would bypass
        # pyvenv.cfg and incorrectly invoke the dependency-free global Python.
        proc = subprocess.run([str(args.interpreter.absolute()), *command], cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (dest / f'step_{index + 1}.log').write_text(proc.stdout)
        report['steps'].append({'command': command, 'returncode': proc.returncode})
        (dest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        if proc.returncode:
            raise RuntimeError('Reconstruction failed: ' + ' '.join(command))
    mismatches = []
    for name, old_bytes in before.items():
        new_bytes = (root / name).read_bytes()
        if name.endswith('.json'):
            same = normalize_json(json.loads(old_bytes)) == normalize_json(json.loads(new_bytes))
        else:
            same = old_bytes == new_bytes
        if not same:
            mismatches.append(name)
    tex_mismatches = [name for name, b in before_tex.items() if (root / name).read_bytes() != b]
    changed_frozen = [name for name, expected in frozen.items() if sha(root / name) != expected]
    with (root / 'results/error_structure_v3/decision_policies.csv').open(newline='') as f:
        policies = {(r['study'], r['policy']): r for r in csv.DictReader(f)}
    expected_critical = {
        ('TEP', 'single'): (64, 20, 0, 1.0),
        ('TEP ABBA', 'single'): (40, 2, 0, 1.0),
        ('Hydraulic (5 blocks)', 'single'): (49, 23, 0, 1.0),
        ('Hydraulic (5 blocks)', 'canary'): (66, 6, 0, 1.75),
        ('Hydraulic (prospective)', 'single'): (105, 57, 0, 1.0),
        ('Hydraulic (prospective)', 'canary'): (123, 3, 36, 17/9),
        ('Hydraulic (prospective)', 'vote_tie_first'): (102, 60, 0, 3.0),
    }
    for key, expected in expected_critical.items():
        row = policies[key]
        actual = tuple(float(row[k]) for k in ['accepted_correct', 'accepted_error', 'abstained', 'calls_per_window'])
        if actual != expected:
            raise ValueError(f'Critical policy mismatch {key}: {actual}')
    with (root / 'results/statistics_v2/abba_contrast.csv').open(newline='') as f:
        abba = next(csv.DictReader(f))
    if int(abba['call_relabellings']) != 6 or float(abba['call_perm_p_acc']) != 4/6:
        raise ValueError('ABBA design regression')
    report.update({'scientific_outputs_compared': len(before), 'scientific_output_mismatches': mismatches,
                   'canonical_tex_compared': len(before_tex), 'canonical_tex_mismatches': tex_mismatches,
                   'changed_frozen_files': changed_frozen, 'critical_policy_rows_checked': len(expected_critical),
                   'abba_exact_design': {'assignments': 6, 'p': 4/6},
                   'complete': not (mismatches or tex_mismatches or changed_frozen),
                   'completed_utc': datetime.now(timezone.utc).isoformat()})
    (dest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['complete']:
        raise RuntimeError('Reconstructed package has mismatches; see verification.json')

if __name__ == '__main__':
    main()
