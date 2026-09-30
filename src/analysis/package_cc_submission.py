"""Package the current CC/QREI revision, without inference or publication."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results'
CC=RESULTS/'industrial_interface_focused_overleaf'
ARCHIVE=ROOT/'paper/archive/pre_cc_correction_20260930'

def allowed(name):
    p=Path(name)
    if name.startswith('results/reference_audit_20260930/') and len(p.parts)>3:
        return False  # Local third-party source caches are not redistributed.
    return not (p.is_absolute() or '..' in p.parts or '__pycache__' in p.parts or 'source_cache' in p.parts
                or any(x == '.DS_Store' or x.startswith('._') for x in p.parts)
                or any(x.startswith('.env') for x in p.parts)
                or p.suffix in {'.pyc','.log','.blg','.aux','.bbl','.lock'}
                or p.name.endswith('.build.txt'))

def main():
    record=json.loads((RESULTS/'qrei_submission/sync_record.json').read_text())
    if not record.get('compiled_this_run') or not record.get('pdf_sha256'):
        raise RuntimeError('Run sync_cc_submission.py --compile successfully before packaging.')
    for name,wanted in {**record['sources'],**record['pdf_sha256']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Source/PDF changed after successful compilation: '+name)
    for name in ['industrial_interface_focused_overleaf.zip','industrial_interface_focused_review.zip']:
        old=RESULTS/name; saved=ARCHIVE/('legacy_'+name)
        if old.exists() and not saved.exists():shutil.copy2(old,saved)
    with zipfile.ZipFile(RESULTS/'industrial_interface_focused_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(CC.rglob('*')):
            if p.is_file() and p.name!='README.txt' and allowed(p.relative_to(CC).as_posix()) and p.suffix in {'.tex','.bib','.pdf','.png','.txt'}:
                z.write(p,p.relative_to(CC))
        z.writestr('README.txt','Current CC revision. Editorial source: main.tex; supplement.tex compiles separately. Cached Tectonic or a normal LaTeX/BibTeX installation can compile both. The complete evidence/code archive is separate. Author confirmation is recorded in the submission metadata; no journal submission is implied.\n')
    payload={}
    old=ARCHIVE/'legacy_industrial_interface_focused_review.zip'
    if not old.exists():
        raise FileNotFoundError('The preserved earlier evidence archive is required to retain historical study dependencies.')
    if old.exists():
        with zipfile.ZipFile(old) as z:
            for info in z.infolist():
                if info.is_dir():continue
                name=info.filename
                if name.startswith('industrial_interface_review/'):name=name[len('industrial_interface_review/'):]
                if not allowed(name) or name.startswith('paper/focused_submission/'):continue
                payload[name]=z.read(info)
    directories=['src','tests','results/industrial_interface_focused_overleaf','results/qrei_submission',
        'results/independent_validation','results/context_blocked_v1','results/hydraulic_validation_v1',
        'results/policy_hydraulic_v1','results/policy_paderborn_v1',
        'results/canary_hydraulic_v1','results/canary_controls_v2','results/canary_stability_v2',
        'results/hydraulic_validation_v1_completion','results/error_structure_v3','results/statistics_v2',
        'results/baselines_v2','results/cc_statistical_correction_v4','paper/archive/pre_cc_correction_20260930',
        'paper/reviews/cc_q2_2026-09-30','results/cc_revision_delivery',
        'results/reference_audit_20260930','results/submission_finalization_20260930']
    for directory in directories:
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file():continue
            name=p.relative_to(ROOT).as_posix()
            if allowed(name) and p.suffix!='.zip':payload[name]=p.read_bytes()
    for name in ['paper/CURRENT_STATUS.md','paper/CC_REVISION_2026-09-30.md','paper/REFERENCE_PUBLICATION_2026-09-30.md','requirements.txt']:
        p=ROOT/name
        if p.exists():payload[name]=p.read_bytes()
    for study in ['independent_validation','context_blocked_v1','hydraulic_validation_v1',
                  'hydraulic_validation_v1_completion','policy_hydraulic_v1','policy_paderborn_v1',
                  'canary_hydraulic_v1','canary_stability_v2']:
        for suffix in ['protocol.json','predictions/frozen_manifest.json']:
            if f'results/{study}/{suffix}' not in payload:
                raise FileNotFoundError(f'Missing required frozen evidence: {study}/{suffix}')
    payload['CC_REVISION_README.txt']=b'''Current manuscript: results/industrial_interface_focused_overleaf/main.pdf
Submission set: results/qrei_submission/upload/
Target: Quality and Reliability Engineering International, subscription route.
The user confirmed all authors approved the corrected revision on 30 September 2026.
No journal submission is claimed. Repository publication is tracked separately in
results/submission_finalization_20260930/; archive creation alone does not push files.
Earlier pipeline manuscripts are historical and must not replace this source.

Offline reconstruction (numpy/scipy/pandas/sklearn/matplotlib environment):
python src/analysis/rebuild_cc_statistics_v4.py
python src/analysis/canary_controls_v2.py
python src/analysis/canary_controls_v2.py --root results/canary_stability_v2 --out results/canary_stability_v2/analysis
python src/analysis/audit_canary_raw_v2.py
python src/analysis/render_canary_stability_v2.py
python src/analysis/sync_cc_submission.py --compile
Do not run prepare/run/freeze commands for existing studies. Those are not rebuild commands.
The review recomputations are an internal arithmetic audit, not independent peer review.
Original protocol/response manifests and a pre-correction manuscript snapshot are retained.
'''
    manifest={k:hashlib.sha256(v).hexdigest() for k,v in sorted(payload.items())}
    payload['CC_PACKAGE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    with zipfile.ZipFile(RESULTS/'industrial_interface_focused_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):z.writestr('industrial_interface_review/'+name,data)
    with zipfile.ZipFile(RESULTS/'qrei_submission_current.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((RESULTS/'qrei_submission').rglob('*')):
            if p.is_file() and allowed(p.relative_to(RESULTS/'qrei_submission').as_posix()):
                z.write(p,p.relative_to(RESULTS/'qrei_submission'))
    print(json.dumps({'review_files':len(manifest),'source':str(CC),'public_upload':False}))

if __name__=='__main__':main()
