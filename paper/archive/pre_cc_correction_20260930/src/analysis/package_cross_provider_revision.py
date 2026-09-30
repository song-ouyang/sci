"""Build local Overleaf and review archives from verified, visually reviewed artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.audit_domestic_extension import verify
from src.analysis.score_revision_v3 import verify_frozen_runs
from src.experiments.mechanism_extension import audit_outputs,BASE

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tree(p):
    return [f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in {'.pyc','.zip'}
            and f.name not in {'package_manifest.json','runner.lock','review_package_manifest.json','verification_workspace.txt'}]

def safe_files(files):
    secrets=[]
    env=ROOT/'.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1)
                if 'KEY' in k and len(v.strip().strip('\"\''))>12:
                    secrets.append(v.strip().strip('\"\'').encode())
    for f in files:
        if f.is_symlink() or not f.is_file() or f.name.startswith('.env'):
            raise ValueError('Unsafe package path: '+str(f))
        if any(secret in f.read_bytes() for secret in secrets):
            raise ValueError('Credential material detected in package file: '+str(f.relative_to(ROOT)))

def archive(path,entries):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f,name in entries:
            z.write(f,name)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for f,name in entries:
            if hashlib.sha256(z.read(name)).hexdigest()!=digest(f):raise ValueError('ZIP content changed: '+name)
    print(path.name, 'files:',len(entries),'bytes:',path.stat().st_size)

def main():
    verify_frozen_runs(ROOT/'results/revision_v3')
    verify(ROOT/'results/domestic_extension_v2')
    audit_outputs(BASE)
    from src.experiments.independent_validation import audit_outputs as audit_independent
    audit_independent(ROOT/'results/independent_validation')
    iv=json.loads((ROOT/'results/independent_validation/analysis/scores.json').read_text())['coverage']
    initial=ROOT/'results/domestic_extension'
    interrupted=json.loads((initial/'interruption_manifest.json').read_text())
    for n,h in interrupted['files'].items():
        if digest(initial/n)!=h:raise ValueError('Interrupted record changed: '+n)
    quality=json.loads((ROOT/'results/cross_provider_revision/manuscript_quality.json').read_text())
    if quality['pdf_sha256']!=digest(ROOT/'paper/main_inline.pdf'):raise ValueError('Unreviewed PDF')
    provenance=json.loads((ROOT/'results/revision_v3/analysis/paper_provenance.json').read_text())
    for path,key in [('paper/main.tex','main_sha256'),('paper/main_inline.tex','main_inline_sha256'),('paper/main_compile.tex','main_compile_sha256'),
                     ('paper/revision_v3_template.tex','template_sha256'),
                     ('paper/references.bib','references_sha256'),
                     ('src/analysis/render_revision_v3_paper.py','renderer_sha256'),
                     ('results/revision_v3/analysis/scores.json','analysis_sha256')]:
        if digest(ROOT/path)!=provenance[key]:raise ValueError('Stale manuscript input: '+path)
    for group in ['extension_files','domestic_files','figure_files','figure_data_files','claude_reported_files','usable_output_files','independent_files']:
        if not provenance.get(group):raise ValueError('Missing paper provenance: '+group)
        for n,h in provenance[group].items():
            if digest(ROOT/n)!=h:raise ValueError('Stale paper provenance: '+n)
    if any((ROOT/'paper'/n).read_bytes()!=(ROOT/'paper/main_inline.tex').read_bytes() for n in ['main.tex','main_compile.tex']):
        raise ValueError('Unsynchronized sources')
    files=[]
    for name in ['results/revision_v3','results/mechanism_extension','results/domestic_extension',
                 'results/domestic_extension_v2','results/revision_v3_claude_extension','results/revision_v3_claude_extension_v2',
                 'results/claude_extension_audit_archive','results/claude_manual_recheck_2026-09-27',
                 'results/claude_descriptive_recheck_2026-09-28','results/claude_conversation_exploratory_v1',
                 'results/cross_provider_revision','results/claude_reported_summary_2026-09-27','results/independent_validation','paper/figures/revision_v3','paper/figures/mechanism_extension','paper/figures/independent_validation']:
        files+=tree(ROOT/name)
    # Code-only allowlists exclude configuration, runtime directories and credentials.
    for name in ['src/analysis','src/experiments','tests']:
        files += list((ROOT/name).glob('*.py'))
    files += [ROOT/'paper'/n for n in ['main.tex','main_inline.tex','main_compile.tex','main_inline.pdf','references.bib',
        'revision_v3_template.tex','independent_validation_text.json','mechanism_extension_text.json','domestic_extension_text.json','claude_reported_text.json',
        'CROSS_PROVIDER_REVISION_CHANGELOG.md','PAPER_SUMMARY_ZH.md','CURRENT_STATUS.md','CLAUDE_RERUN_HANDOFF_2026-09-27.md',
        'MECHANISM_REVISION_CHANGELOG.md','REVISION_PLAN_2026-09-27.md','CLAUDE_EXTENSION_REMEDIATION_2026-09-27.md']]
    files += list((ROOT/'paper').glob('*.md'))
    files += [ROOT/n for n in ['requirements-revision-v3.txt','REVIEW_PACKAGE_README.md','DOMESTIC_MODELS.md']]
    files=sorted(set(files));safe_files(files)
    manifest={'schema':'cross-provider-review-v1','created_at':datetime.now(timezone.utc).isoformat(),
              'scope':'Local review artifact; unpublished. Claude reported summaries are included separately and remain noneligible.',
              'claude_reported_arrays_excluded_from_controlled_counts':27,
              'claude_conversation_planned_records_excluded_from_controlled_counts':18,
              'claude_conversation_strict_arrays':15,
              'claude_conversation_posthoc_id_aligned_arrays':16,
              'scheduled_requests':129+iv['planned'],'valid_requests':125+iv['valid'],'valid_classifications':5334+iv['valid_labels'],
              'unique_source_units':302,'aborted_api_trial_included_separately':True,
              'credentials_included':False,'raw_source_simulation_downloads_included':False,
              'offline_rebuild':'python src/analysis/rebuild_cross_provider_review.py',
              'files':{str(f.relative_to(ROOT)):digest(f) for f in files}}
    mp=ROOT/'results/cross_provider_revision/review_package_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive(ROOT/'results/cross_provider_revision_review.zip',[(f,'cross_provider_revision_review/'+str(f.relative_to(ROOT))) for f in files+[mp]])
    ol=[ROOT/'paper'/n for n in ['main.tex','main_inline.tex','main_compile.tex','references.bib']]
    ol += [p for d in ['revision_v3','mechanism_extension','independent_validation'] for p in (ROOT/'paper/figures'/d).glob('*.pdf')]
    safe_files(ol)
    readme=ROOT/'results/cross_provider_revision/OVERLEAF_README.txt'
    readme.write_text('Upload this ZIP to Overleaf. Set main.tex as the main document and use XeLaTeX with BibTeX.\n'
                      'This is the synchronized local review manuscript; author details and journal formatting are pending.\n'
                      'The full analysis and raw-response audit trail are in cross_provider_revision_review.zip.\n')
    archive(ROOT/'results/paper_cross_provider_overleaf.zip',[(f,str(f.relative_to(ROOT/'paper'))) for f in ol]+[(readme,'README.txt')])

if __name__=='__main__':main()
