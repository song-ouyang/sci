"""Package verified original results plus the separately frozen follow-up for review."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.mechanism_extension import audit_outputs,BASE,digest

def main():
    p,g,records,_=audit_outputs(BASE)
    original=ROOT/'results/revision_v3'
    scores=json.loads((BASE/'analysis/scores.json').read_text())
    if scores['protocol_sha256']!=digest(BASE/'protocol.json') or scores['frozen_manifest_sha256']!=digest(BASE/'predictions/frozen_manifest.json'):
        raise ValueError('Extension analysis provenance mismatch')
    exported=json.loads((BASE/'analysis/cases/manifest.json').read_text())
    if exported['scores_sha256']!=digest(BASE/'analysis/scores.json'):
        raise ValueError('Stale case exports')
    for n,h in exported['files'].items():
        if digest(BASE/'analysis/cases'/n)!=h:raise ValueError('Case hash mismatch')
    quality=json.loads((BASE/'analysis/manuscript_quality.json').read_text())
    pdf=ROOT/'paper/main_inline.pdf'
    if digest(pdf)!=quality['pdf_sha256']:
        raise ValueError('PDF differs from visually reviewed version')
    provenance=json.loads((original/'analysis/paper_provenance.json').read_text())
    for filename,key in [('paper/main.tex','main_sha256'),('paper/main_inline.tex','main_inline_sha256'),
                         ('paper/revision_v3_template.tex','template_sha256'),
                         ('src/analysis/render_revision_v3_paper.py','renderer_sha256'),
                         ('results/revision_v3/analysis/scores.json','analysis_sha256')]:
        if digest(ROOT/filename)!=provenance[key]:raise ValueError('Unbuilt paper input: '+filename)
    for name,h in provenance.get('extension_files',{}).items():
        if digest(ROOT/name)!=h:raise ValueError('Extension paper provenance mismatch: '+name)
    if not provenance.get('extension_files'):
        raise ValueError('Paper does not include the frozen extension')
    if (ROOT/'paper/main.tex').read_bytes()!=(ROOT/'paper/main_inline.tex').read_bytes():
        raise ValueError('Unsynchronized TeX sources')
    files=[]
    for folder in [original,BASE,ROOT/'tests',ROOT/'paper/figures/revision_v3',ROOT/'paper/figures/mechanism_extension']:
        files += [f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts
                  and f.suffix!='.pyc' and f.name not in {'package_manifest.json','runner.lock','review_package_manifest.json','final_quality_check.json','visual_polish_reproduction_check.json'}]
    files += [ROOT/'paper'/n for n in ['main.tex','main_inline.tex','main_inline.pdf','references.bib','revision_v3_template.tex',
              'mechanism_extension_text.json','REVISION_PLAN_2026-09-27.md','MECHANISM_REVISION_CHANGELOG.md']]
    # Only reproduction dependencies; historical API clients are not needed.
    files += [ROOT/'src/experiments'/n for n in [
        'build_revision_v3.py','prepare_revision_v3.py','run_revision_v3.py',
        'continue_revision_v3.py','run_v2_experiments.py']]
    files += [ROOT/'src/analysis'/n for n in [
        'freeze_revision_v3.py','score_revision_v3.py','render_revision_v3_paper.py',
        'package_revision_v3.py','draw_revision_v3_workflow.py',
        'plot_revision_v3_diagnostics.py','export_revision_v3_cases.py']]
    files += [ROOT/n for n in [
        'src/experiments/mechanism_extension.py',
        'src/analysis/render_mechanism_extension.py',
        'src/analysis/plot_mechanism_context.py',
        'src/analysis/export_mechanism_cases.py',
        'src/analysis/package_mechanism_revision.py',
        'src/analysis/rebuild_mechanism_review.py']]
    files += [ROOT/'requirements-revision-v3.txt',ROOT/'REVIEW_PACKAGE_README.md']
    files=sorted(set(files))
    for f in files:
        if not f.is_file() or f.is_symlink():raise ValueError('Missing or linked file: '+str(f))
    manifest={'schema':'mechanism-revision-review-package-v1','created_at':datetime.now(timezone.utc).isoformat(),
              'status':'Local review package, not publicly published; author and license details remain pending',
              'original_sessions':27,'extension_sessions':len(records),'unique_sources_across_both_studies':130,
              'raw_source_datasets_included':False,
              'offline_reproduce':'python src/analysis/rebuild_mechanism_review.py',
              'files':{str(f.relative_to(ROOT)):digest(f) for f in files}}
    mp=BASE/'review_package_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    target=ROOT/'results/mechanism_revision_review.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in files+[mp]:z.write(f,'mechanism_revision_review/'+str(f.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for name,h in manifest['files'].items():
            if hashlib.sha256(z.read('mechanism_revision_review/'+name)).hexdigest()!=h:
                raise ValueError('ZIP content hash mismatch: '+name)
    print('Verified review archive:',target,'files:',len(files)+1,'bytes:',target.stat().st_size)

if __name__=='__main__':main()
