"""Stage the reproducibility subset of the verified archive into an existing clone.

No Git commit/push is performed here. Internal submission correspondence and
downloaded third-party papers are excluded. Existing unrelated files are retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def selected(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or any(x.startswith('.env') or x == '.DS_Store' or x.startswith('._') for x in p.parts):
        return False
    if any(x in p.parts for x in ['__pycache__','source_cache']) or p.suffix in {'.zip','.pyc','.log','.aux','.bbl','.blg'}:
        return False
    if name.startswith('results/qrei_submission/'):
        return False
    if name.startswith('results/reference_audit_20260930/'):
        return p.name in {'reference_audit.json','REFERENCE_AUDIT.md','corrections_applied.json'}
    if name.startswith('results/submission_finalization_20260930/'):
        return p.name in {'public_reproduction_verification.json','public_export_review.json','publication_manifest.json'}
    if p.parts[0] in {'data','src','tests','results'}:
        return True
    if name.startswith('paper/archive/pre_cc_correction_20260930/src/analysis/'):
        return p.suffix == '.py'
    return name in {'requirements.txt','requirements-revision-v3.txt'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT/'results/industrial_interface_focused_review.zip')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo/'.git').exists():
        raise RuntimeError('Destination must be an existing checked Git clone.')
    written = {}
    def save(name, data):
        target = repo/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written[name] = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(args.package) as z:
        prefix = 'industrial_interface_review/'
        manifest = json.loads(z.read(prefix+'CC_PACKAGE_MANIFEST.json'))
        for name, expected in manifest.items():
            if not selected(name):
                continue
            data = z.read(prefix+name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Archive checksum mismatch: '+name)
            save(name, data)
            source = 'results/industrial_interface_focused_overleaf/'
            if name.startswith(source):
                save('manuscript/'+name[len(source):], data)
    save('README.md', (ROOT/'GITHUB_README.md').read_bytes())
    frozen_checks = 0
    immutable_files = {}
    for study in ['independent_validation','context_blocked_v1','hydraulic_validation_v1',
                  'hydraulic_validation_v1_completion','policy_hydraulic_v1','policy_paderborn_v1',
                  'canary_hydraulic_v1','canary_stability_v2']:
        base = repo/'results'/study
        protocol = json.loads((base/'protocol.json').read_text())
        freeze = json.loads((base/'predictions/frozen_manifest.json').read_text())
        checks = {base/'predictions'/n: h for n,h in freeze['files'].items()}
        checks[base/'protocol.json'] = freeze['protocol_sha256']
        checks[base/'inference/batches.json'] = protocol['input_sha256']
        schedule = 'session_schedule.json' if study == 'independent_validation' else 'schedule.json'
        checks[base/schedule] = protocol['schedule_sha256']
        checks.update({repo/n: h for n,h in protocol['code_sha256'].items()})
        for path, expected in checks.items():
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Incomplete or changed frozen dependency: '+str(path))
            immutable_files[str(path.relative_to(repo))] = expected
            frozen_checks += 1
        # Preserve the release bytes of manifests and all published study inputs
        # across reconstruction, including answer keys and prompt files.
        evidence = [base/'predictions/frozen_manifest.json',base/'protocol.sha256']
        evidence += [p for folder in ['inference','answers'] for p in (base/folder).rglob('*') if p.is_file()]
        evidence += [base/n for n in ['source_audit.json','data_audit.json','contract_audit.json'] if (base/n).exists()]
        for path in evidence:
            immutable_files[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    retained = {}
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
    for name in tracked:
        if not name or name in written or name == 'PUBLICATION_MANIFEST.json':
            continue
        path = repo/name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Unexpected retained Git entry: '+name)
        retained[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    public_manifest = {'release_tag':'qrei-2026-09-30','files':dict(sorted({**retained,**written}.items())),
                       'new_export_file_count':len(written),'retained_files':dict(sorted(retained.items())),
                       'immutable_files':dict(sorted(immutable_files.items())),
                       'frozen_checks':frozen_checks,'journal_submission_claimed':False,
                       'canonical_editorial_source':'results/industrial_interface_focused_overleaf',
                       'reading_mirror':'manuscript','command':'python src/analysis/reproduce_public_revision.py --compile'}
    (repo/'PUBLICATION_MANIFEST.json').write_text(json.dumps(public_manifest,indent=2)+'\n')
    print(json.dumps({'staged_files':len(written),'retained_files':len(retained),'manifest_files':len(public_manifest['files']),'frozen_checks':frozen_checks,'repo':str(repo)}))

if __name__ == '__main__':
    main()
