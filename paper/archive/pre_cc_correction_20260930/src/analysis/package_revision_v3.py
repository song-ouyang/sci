"""Assemble a compact local reproduction archive without duplicating raw datasets."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_v3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scores = json.loads((BASE/'analysis/scores.json').read_text())
    frozen_path = BASE/'predictions/frozen_manifest.json'
    frozen = json.loads(frozen_path.read_text())
    if digest(frozen_path) != scores['audit']['frozen_manifest_sha256']:
        raise ValueError('Analysis/frozen-prediction provenance mismatch')
    for name, value in frozen['files'].items():
        if digest(BASE/'predictions'/name) != value:
            raise ValueError(f'Frozen result changed: {name}')
    paper = ROOT/'paper'
    provenance = json.loads((BASE/'analysis/paper_provenance.json').read_text())
    for path, key in [(paper/'main_inline.tex','main_inline_sha256'),(paper/'main.tex','main_sha256'),
                      (BASE/'analysis/scores.json','analysis_sha256'),
                      (paper/'revision_v3_template.tex','template_sha256'),
                      (ROOT/'src/analysis/render_revision_v3_paper.py','renderer_sha256')]:
        if digest(path) != provenance[key]:
            raise ValueError(f'Manuscript provenance mismatch: {path}')
    for name, value in provenance.get('figure_builders', {}).items():
        if digest(ROOT/'src/analysis'/name) != value:
            raise ValueError(f'Figure builder changed: {name}')
    for category in ['figure_files', 'figure_data_files']:
        for name, value in provenance.get(category, {}).items():
            if digest(ROOT/name) != value:
                raise ValueError(f'Figure provenance mismatch: {name}')
    reuse = json.loads((BASE/'analysis/reuse/manifest.json').read_text())
    if digest(ROOT/reuse['exporter']) != reuse['exporter_sha256']:
        raise ValueError('Rebuild case exports after changing the exporter')
    for name, value in reuse['sources_relative_to_revision_root'].items():
        if digest(BASE/name) != value:
            raise ValueError(f'Case export source changed: {name}')
    for name, metadata in reuse['outputs'].items():
        if digest(BASE/'analysis/reuse'/name) != metadata['sha256']:
            raise ValueError(f'Case export changed: {name}')
    if (paper/'main.tex').read_bytes() != (paper/'main_inline.tex').read_bytes():
        raise ValueError('Manuscript sources are not synchronized')
    pdf = paper/'main_inline.pdf'
    if not pdf.exists() or pdf.stat().st_mtime < (paper/'main_inline.tex').stat().st_mtime:
        raise ValueError('Compile the current manuscript before packaging')
    quality = json.loads((BASE/'analysis/final_quality_check.json').read_text())
    if quality['pdf_sha256'] != digest(pdf):
        raise ValueError('Final PDF quality review does not match the current manuscript')
    files = [p for p in BASE.rglob('*') if p.is_file() and p.name not in {'package_manifest.json'}]
    files += [paper/name for name in ['main_inline.tex','main.tex','main_inline.pdf','references.bib',
              'revision_v3_template.tex','REVISION_CHANGELOG_2026-09-26.md','REVISION_RESEARCH_NOTES.md',
              'VISUAL_POLISH_2026-09-27.md']]
    files += list((paper/'figures/revision_v3').glob('*'))
    files += [ROOT/'src/experiments'/name for name in ['build_revision_v3.py','prepare_revision_v3.py','run_revision_v3.py','continue_revision_v3.py','run_v2_experiments.py']]
    files += [ROOT/'src/analysis'/name for name in ['freeze_revision_v3.py','score_revision_v3.py','render_revision_v3_paper.py','package_revision_v3.py',
              'draw_revision_v3_workflow.py','plot_revision_v3_diagnostics.py','export_revision_v3_cases.py']]
    files += [ROOT/'tests'/name for name in ['test_response_parsing.py','test_score_revision_v3.py']]
    files += [ROOT/'requirements-revision-v3.txt']
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise ValueError(f'Missing reproduction files: {missing}')
    files = sorted(set(files))
    manifest = {'schema':'revision-v3-local-reproduction-v1',
                'status':'local review package; not publicly published or assigned a DOI/license',
                'planned_sessions':scores['audit']['planned_sessions'],
                'valid_sessions':scores['audit']['completed_sessions'], 'invalid_sessions':scores['audit']['failed_sessions'],
                'valid_predictions':scores['audit']['total_valid_predictions'],
                'requested_predictions':scores['audit']['expected_predictions'],
                'unique_test_source_units':130,
                'source_raw_files_included':False,
                'reproduce':'See results/revision_v3/README.md; frozen-output analysis needs no external model calls.',
                'files':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    manifest_path = BASE/'package_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive = ROOT/'results/revision_v3_reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for path in files + [manifest_path]:
            zf.write(path, 'revision_v3_reproduction/'+str(path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for name, value in manifest['files'].items():
            if hashlib.sha256(zf.read('revision_v3_reproduction/'+name)).hexdigest() != value:
                raise ValueError(f'Archive SHA mismatch: {name}')
    print(f'Packaged {len(files)+1} files, {archive.stat().st_size:,} bytes: {archive}')


if __name__ == '__main__':
    main()
