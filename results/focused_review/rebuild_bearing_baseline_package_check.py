#!/usr/bin/env python3
"""Explicitly gated, offline final-package reconstruction and preservation check.

Preparation does not inspect any archive or prediction. Execute only after the
root agent has finished and frozen the new experiment and created both final
ZIPs. Includes the frozen supervised bearing reference; no training. All reconstruction writes are confined to a new extraction directory.
No model, prepare, run, freeze, network download, or broad test suite is invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'paper/focused_submission').is_dir())
BUNDLED_PYTHON = Path('/Users/guoba/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
SCIENTIFIC_JSON = {
    'results/policy_error_classes_v1/audit.json': {'generated_at'},
    'results/repeat_condition_v2/audit.json': {'created_at'},
    'results/call_timing_v2/timing.json': {'created_at_utc'},
    'results/context_blocked_v1/analysis/scores.json': {'scored_at'},
    'results/policy_hydraulic_v1/analysis/scores.json': {'scored_at'},
    'results/policy_paderborn_v1/analysis/scores.json': {'scored_at'},
    'results/bearing_baseline_v1/analysis/scores.json': {'generated_at_utc'},
}
CSV_DIRS = ('results/repeat_condition_v2', 'results/call_timing_v2',
            'results/context_blocked_v1/analysis','results/policy_hydraulic_v1/analysis','results/policy_paderborn_v1/analysis','results/bearing_baseline_v1/analysis')
EXPECTED_CSV = {
    'results/repeat_condition_v2/' + name + '.csv'
    for name in ('contrasts', 'within_condition_pairs', 'cross_condition_pairs', 'case_stability')
} | {'results/call_timing_v2/' + name + '.csv' for name in ('calls', 'contrasts')} | {
    'results/context_blocked_v1/analysis/' + name + '.csv'
    for name in ('requests', 'focal_metrics', 'paired', 'case_stability', 'block_summaries')
}
EXPECTED_CSV |= {f'results/policy_{d}_v1/analysis/{n}.csv' for d in ['hydraulic','paderborn'] for n in ['requests','randomized_pairs','predictions','policy_cases','policy_costs','flips','incremental_gate_cases']}
EXPECTED_CSV |= {'results/bearing_baseline_v1/analysis/'+n+'.csv' for n in ['scored_predictions','llm_call_comparisons']}
TEX_FILES = ('paper/focused_submission/main.tex', 'paper/focused_submission/supplement.tex')
PDF_FILES = ('paper/focused_submission/main.pdf', 'paper/focused_submission/supplement.pdf')
OLD_PAPER = ('paper/main.tex', 'paper/main_inline.tex', 'paper/main_compile.tex', 'paper/main_inline.pdf')
OLD_STAGES = ('revision_v3', 'mechanism_extension', 'domestic_extension_v2', 'independent_validation')
COMMANDS = (
    ('repeat_condition_v2', ('src/analysis/audit_repeat_condition_v2.py', '--include-hydraulic')),
    ('call_timing_v2', ('src/analysis/audit_call_timing_v2.py',)),
    ('context_blocked_audit', ('src/experiments/context_blocked_v1.py', 'audit')),
    ('context_blocked_score', ('src/analysis/score_context_blocked_v1.py',)),
    ('policy_hydraulic_audit', ('src/experiments/policy_validation_v1.py','audit','--root','results/policy_hydraulic_v1')),
    ('policy_paderborn_audit', ('src/experiments/policy_validation_v1.py','audit','--root','results/policy_paderborn_v1')),
    ('policy_hydraulic_score', ('src/analysis/score_policy_validation_v1.py','--root','results/policy_hydraulic_v1')),
    ('policy_paderborn_score', ('src/analysis/score_policy_validation_v1.py','--root','results/policy_paderborn_v1')),
    ('policy_error_class_audit', ('src/analysis/audit_policy_error_classes_v1.py',)),
    ('bearing_baseline_audit', ('src/analysis/bearing_baseline_v1.py','audit')),
    ('bearing_baseline_score', ('src/analysis/bearing_baseline_v1.py','score')),
    ('build_focused_manuscript', ('src/analysis/build_focused_manuscript.py',)),
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), f'Missing or symlinked file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def byte_sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative_path(value):
    path = PurePosixPath(value)
    require(isinstance(value, str) and value and not path.is_absolute()
            and '..' not in path.parts and '\\' not in value
            and '\x00' not in value, f'Unsafe archive/manifest path: {value!r}')
    require(not any(part == '.env' or part.startswith('.env.') for part in path.parts),
            f'Environment file in package: {value}')
    return path


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    files, seen = [], set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = relative_path(info.filename)
            require(info.filename not in seen, f'Duplicate ZIP member: {info.filename}')
            seen.add(info.filename)
            require(not stat.S_ISLNK(info.external_attr >> 16), f'Symlink ZIP member: {info.filename}')
            if info.is_dir():
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            files.append(info.filename)
    return files


def csv_inventory(root):
    return {p.relative_to(root).as_posix() for rel in CSV_DIRS
            for p in (root / rel).rglob('*.csv') if p.is_file()}


def scientific_value(path, ignored):
    value = read_json(path)
    require(isinstance(value, dict), f'Expected JSON object: {path}')
    # Ignore ONLY the top-level analysis-generation timestamps. All measured
    # dispatch/start/end/duration/midpoint/offset values remain exact comparisons.
    require(ignored <= set(value), f'Expected generation timestamp absent: {path}')
    return {key: item for key, item in value.items() if key not in ignored}


def compare_science(reference, rebuilt, phase):
    rows = []
    for rel, ignored in SCIENTIFIC_JSON.items():
        left = scientific_value(reference / rel, ignored)
        right = scientific_value(rebuilt / rel, ignored)
        require(left == right, f'{phase}: scientific JSON differs: {rel}')
        canonical = json.dumps(left, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        rows.append({'path': rel, 'status': 'PASS', 'ignored_top_level_fields': sorted(ignored),
                     'scientific_content_sha256': byte_sha(canonical)})
    return rows


def compare_bytes(reference, candidate, paths, phase):
    rows = []
    for rel in sorted(paths):
        expected, actual = sha(reference / rel), sha(candidate / rel)
        require(expected == actual, f'{phase}: bytes differ: {rel}')
        rows.append({'path': rel, 'sha256': actual, 'status': 'PASS'})
    return rows


def archive_member(z, rel):
    names = [n for n in z.namelist() if n == rel or n.endswith('/' + rel)]
    require(len(names) == 1, f'Expected one historical member {rel}, got {len(names)}')
    return names[0]


def check_original_preservation(archive, source, extracted):
    result = {'historical_zip_sha256': sha(archive), 'paper_files': [], 'stages': []}
    with zipfile.ZipFile(archive) as z:
        def compare_three(rel, expected=None):
            member = archive_member(z, rel)
            historic = byte_sha(z.read(member))
            if expected is not None:
                require(historic == expected, f'Historical frozen inventory mismatch: {rel}')
            require(sha(source / rel) == historic, f'Original project artifact changed: {rel}')
            require(sha(extracted / rel) == historic, f'Original artifact not preserved in final package: {rel}')
            return {'path': rel, 'sha256': historic, 'status': 'PASS'}

        for rel in OLD_PAPER:
            result['paper_files'].append(compare_three(rel))
        for stage in OLD_STAGES:
            # The domestic API study froze batches and sessions relative to
            # its stage root; the three CLI studies use predictions/ roots.
            prefix = f'results/{stage}' + ('' if stage == 'domestic_extension_v2' else '/predictions')
            rel = prefix + '/frozen_manifest.json'
            manifest_record = compare_three(rel)
            manifest = json.loads(z.read(archive_member(z, rel)))
            require(isinstance(manifest.get('files'), dict), f'Missing frozen files mapping: {stage}')
            outputs = []
            for member, expected in sorted(manifest['files'].items()):
                relative_path(member)
                outputs.append(compare_three(prefix + '/' + member, expected))
            result['stages'].append({'stage': stage, 'manifest': manifest_record,
                                     'output_count': len(outputs), 'outputs': outputs})
    return result


def command(argv, cwd, label, work, report, timeout=900):
    require('prepare' not in argv and 'run' not in argv and 'freeze' not in argv,
            f'Forbidden execution command: {argv}')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    started = time.monotonic()
    completed = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=timeout)
    log = work / 'logs' / f'{label}.log'
    log.parent.mkdir(exist_ok=True)
    log.write_text(completed.stdout + '\n--- STDERR ---\n' + completed.stderr)
    report.setdefault('commands', []).append({'name': label, 'argv': [str(x) for x in argv],
        'cwd': str(cwd), 'exit_code': completed.returncode,
        'elapsed_seconds': round(time.monotonic() - started, 3), 'log': str(log)})
    require(completed.returncode == 0, f'{label} failed; see {log}')


PDF_COMPARISON_CODE = r'''
import hashlib, json, sys
from pathlib import Path
from pypdf import PdfReader
source, full, compiled, destination = map(Path, sys.argv[1:])
result = {'tool': 'pypdf.PdfReader.extract_text', 'comparison': 'exact per-page text; no normalization', 'documents': []}
for name in ['main', 'supplement']:
    paths = [source / 'paper/focused_submission' / (name + '.pdf'),
             full / 'paper/focused_submission' / (name + '.pdf'), compiled / (name + '.pdf')]
    readers = [PdfReader(p) for p in paths]
    texts = [[page.extract_text() or '' for page in reader.pages] for reader in readers]
    rows = []
    for i in range(max(map(len, texts))):
        values = [pages[i] if i < len(pages) else None for pages in texts]
        rows.append({'page': i + 1, 'equal': values[0] == values[1] == values[2],
            'character_counts': [None if v is None else len(v) for v in values],
            'text_sha256': [None if v is None else hashlib.sha256(v.encode()).hexdigest() for v in values]})
    result['documents'].append({'name': name, 'paths': list(map(str, paths)),
        'page_counts': list(map(len, texts)), 'pages': rows,
        'equal': len({len(v) for v in texts}) == 1 and all(r['equal'] for r in rows),
        'nonempty_pages': all(t.strip() for pages in texts for t in pages)})
result['status'] = 'PASS' if all(x['equal'] and x['nonempty_pages'] for x in result['documents']) else 'FAIL'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'documents': [{k:v for k,v in d.items() if k != 'pages'} for d in result['documents']]}))
sys.exit(0 if result['status'] == 'PASS' else 1)
'''


def execute(args, work, report):
    source = args.project_root.resolve()
    full_zip = args.full_zip.resolve()
    overleaf_zip = args.overleaf_zip.resolve()
    require(not (source / 'results/context_blocked_v1/runner.lock').exists(), 'New runner is still active/interrupted')
    require((source / 'results/context_blocked_v1/predictions/frozen_manifest.json').is_file(), 'New experiment is not frozen')
    require(args.python.is_file(), f'Missing project Python: {args.python}')
    require(args.pdf_python.is_file(), f'Missing PDF Python: {args.pdf_python}')
    for path, expected in [(full_zip, args.full_sha256), (overleaf_zip, args.overleaf_sha256)]:
        require(sha(path) == expected, f'Final ZIP hash does not match explicit expected value: {path}')
    report['archives'] = {'full': {'path': str(full_zip), 'sha256': args.full_sha256},
                          'overleaf': {'path': str(overleaf_zip), 'sha256': args.overleaf_sha256}}
    for study in ['policy_hydraulic_v1','policy_paderborn_v1']:
        require(not (source/'results'/study/'runner.lock').exists(), 'New policy runner active')
        require((source/'results'/study/'predictions/frozen_manifest.json').is_file(), 'New policy study not frozen')
    source_csv = csv_inventory(source)
    require(EXPECTED_CSV <= source_csv, 'Required newly generated CSV files are missing')
    reference_files = set(SCIENTIFIC_JSON) | source_csv | set(TEX_FILES) | set(PDF_FILES)
    before = {rel: sha(source / rel) for rel in sorted(reference_files)}
    safe_extract(full_zip, work / 'review')
    safe_extract(overleaf_zip, work / 'overleaf')
    manifests = list((work / 'review').rglob('FOCUSED_PACKAGE_MANIFEST.json'))
    require(len(manifests) == 1, 'Expected exactly one FOCUSED_PACKAGE_MANIFEST.json')
    root = manifests[0].parent
    manifest = read_json(manifests[0])
    require(isinstance(manifest, dict) and manifest, 'Invalid or empty focused package manifest')
    actual_inventory = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual_inventory == set(manifest) | {'FOCUSED_PACKAGE_MANIFEST.json'}, 'Full package inventory differs from manifest')
    for rel, expected in manifest.items():
        relative_path(rel)
        require(sha(root / rel) == expected, f'FOCUSED_PACKAGE_MANIFEST mismatch: {rel}')
    report['focused_manifest'] = {'status': 'PASS', 'files_verified': len(manifest),
                                   'manifest_sha256': sha(manifests[0])}
    report['original_artifact_preservation'] = check_original_preservation(args.historical_zip, source, root)
    require(csv_inventory(root) == source_csv, 'Packaged new CSV inventory differs from current project')
    report['packaged_scientific_json'] = compare_science(source, root, 'packaged')
    report['packaged_current_files'] = compare_bytes(source, root, source_csv | set(TEX_FILES) | set(PDF_FILES), 'packaged')
    require(not (root / 'results/context_blocked_v1/runner.lock').exists(), 'Packaged new runner has a lock')
    require((root / 'results/context_blocked_v1/predictions/frozen_manifest.json').is_file(), 'Packaged experiment is not frozen')
    for label, argv in COMMANDS:
        command([args.python, *argv], root, label, work, report)
    report['rebuilt_scientific_json'] = compare_science(source, root, 'rebuilt')
    require(csv_inventory(root) == source_csv, 'Rebuild changed new CSV inventory')
    report['rebuilt_csv_and_tex'] = compare_bytes(source, root, source_csv | set(TEX_FILES), 'rebuilt')
    build = read_json(root / 'paper/focused_submission/build_manifest.json')
    for category in ('inputs', 'outputs'):
        for rel, expected in build[category].items():
            relative_path(rel)
            require(sha(root / rel) == expected, f'Rebuilt build_manifest {category} mismatch: {rel}')
    report['rebuilt_build_manifest'] = {'status': 'PASS', 'inputs_verified': len(build['inputs']),
                                        'outputs_verified': len(build['outputs'])}
    permitted = set(SCIENTIFIC_JSON) | source_csv | set(TEX_FILES) | {
        'results/repeat_condition_v2/README.md', 'results/call_timing_v2/README.md',
        'results/context_blocked_v1/analysis/summary.md', 'paper/focused_submission/build_manifest.json'}
    changed = [rel for rel, expected in manifest.items() if sha(root / rel) != expected]
    require(set(changed) <= permitted, f'Reconstruction changed frozen or unrelated files: {set(changed) - permitted}')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == actual_inventory,
            'Unexpected files created in reconstructed full package')
    report['rebuild_write_scope'] = {'status': 'PASS', 'changed_derived_files': sorted(changed),
                                      'all_other_manifest_files_unchanged': True}
    overleaf = work / 'overleaf'
    overleaf_rows = []
    for name in ('main.tex', 'supplement.tex'):
        expected = (root / 'paper/focused_submission' / name).read_text().replace(
            r'\graphicspath{{../}}', r'\graphicspath{{./}}').replace(
            r'\bibliography{../references}', r'\bibliography{references}')
        require((overleaf / name).read_text() == expected, f'Overleaf transformed TeX differs: {name}')
        overleaf_rows.append({'path': name, 'sha256': sha(overleaf / name), 'status': 'PASS'})
    for path in sorted(overleaf.rglob('*')):
        if path.is_file() and (path.name == 'references.bib' or 'figures' in path.relative_to(overleaf).parts):
            relative = path.relative_to(overleaf).as_posix()
            require(sha(path) == sha(root / 'paper' / relative), f'Overleaf reference/figure differs: {relative}')
            overleaf_rows.append({'path': relative, 'sha256': sha(path), 'status': 'PASS'})
    report['overleaf_sources'] = overleaf_rows
    tectonic = shutil.which(args.tectonic)
    require(tectonic is not None, f'Tectonic unavailable: {args.tectonic}')
    for name in ('main', 'supplement'):
        command([tectonic, '--only-cached', name + '.tex'], overleaf,
                'overleaf_compile_' + name, work, report)
    command([args.pdf_python, '-c', PDF_COMPARISON_CODE, source, root, overleaf, work / 'pdf_text_comparison.json'],
            work, 'pypdf_exact_page_comparison', work, report)
    report['pdf_text_comparison'] = read_json(work / 'pdf_text_comparison.json')
    require(all(sha(source / rel) == expected for rel, expected in before.items()), 'Current project reference files changed during check')
    require(sha(full_zip) == args.full_sha256 and sha(overleaf_zip) == args.overleaf_sha256,
            'Final ZIP changed during check')
    report['current_project_reference_files_unchanged'] = True
    report['new_csv_count'] = len(source_csv)
    report['new_model_calls'] = 0
    report['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Required explicit final-package execution gate')
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--full-zip', type=Path, default=PROJECT / 'results/industrial_interface_focused_review.zip')
    parser.add_argument('--overleaf-zip', type=Path, default=PROJECT / 'results/industrial_interface_focused_overleaf.zip')
    parser.add_argument('--historical-zip', type=Path, default=PROJECT / 'results/cross_provider_revision_review.zip')
    parser.add_argument('--full-sha256', required=True, help='Root-confirmed SHA-256 of the newly generated final full ZIP')
    parser.add_argument('--overleaf-sha256', required=True, help='Root-confirmed SHA-256 of the newly generated final Overleaf ZIP')
    parser.add_argument('--work-dir', type=Path, default=PROJECT / 'tmp/bearing_baseline_package_check')
    parser.add_argument('--python', type=Path, default=PROJECT / '.venv/bin/python')
    parser.add_argument('--pdf-python', type=Path, default=BUNDLED_PYTHON)
    parser.add_argument('--tectonic', default='tectonic')
    args = parser.parse_args()
    require(args.execute, 'Preparation only: supply --execute after root confirms both final archives')
    require(all(re.fullmatch(r'[0-9a-f]{64}', x) for x in (args.full_sha256, args.overleaf_sha256)), 'Invalid expected ZIP SHA-256')
    work = args.work_dir.resolve()
    require(not work.exists(), f'Refusing to reuse or remove existing check directory: {work}')
    require(work.is_relative_to(args.project_root.resolve() / 'tmp'),
            'Extraction/check output must remain inside project tmp/')
    work.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'bearing-baseline-final-package-check-v1',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
              'execution_scope': 'Frozen-output offline reconstruction in an independent extraction; no inference or refreezing.',
              'ignored_time_fields': {k: sorted(v) for k, v in SCIENTIFIC_JSON.items()},
              'measured_call_times_retained_in_comparison': True,
              'validation_inherited_from_historical_quality': False,
              'script_sha256': sha(Path(__file__))}
    code = 0
    try:
        execute(args, work, report)
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        code = 1
    finally:
        report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        (work / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': report['status'], 'result': str(work / 'result.json'),
                          'error': report.get('error')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
