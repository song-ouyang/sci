"""Package the reviewed TeX and local audit record; no inference or downloads."""
from pathlib import Path
import zipfile
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]


def main():
    paper=ROOT/'paper/focused_submission'
    overleaf=ROOT/'results/industrial_interface_focused_overleaf.zip'
    figure_paths=set()
    with zipfile.ZipFile(overleaf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','supplement.tex']:
            s=(paper/name).read_text().replace(r'\graphicspath{{../}}',r'\graphicspath{{./}}').replace(r'\bibliography{../references}',r'\bibliography{references}')
            z.writestr(name,s)
            figure_paths.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s))
        z.write(ROOT/'paper/references.bib','references.bib')
        for rel in sorted(figure_paths):
            if not rel.startswith('figures/') or '..' in Path(rel).parts:
                raise ValueError('Unsafe figure path: '+rel)
            z.write(ROOT/'paper'/rel,rel)
        z.writestr('README.txt','Author-review draft. Compile main.tex and supplement.tex separately with XeLaTeX/Tectonic and BibTeX. No inference is needed. The supplement preserves the four-stage simulation report and adds a measured follow-up. Author details and final scientific approval remain pending.\n')
    overrides={}
    dirs=['paper/focused_submission','results/hydraulic_validation_v1','results/focused_review','data/uci_hydraulic_processed',
          'results/repeat_condition_v2','results/call_timing_v2','results/context_blocked_v1',
          'results/policy_hydraulic_v1','results/policy_paderborn_v1','data/policy_hydraulic_v1',
          'results/bearing_baseline_v1']
    for directory in dirs:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.log','.blg','.pyc'}:
                overrides[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    # Retain processed bearing data and source manifests, but not the separately
    # downloadable raw MAT/RAR payloads or transport cache.
    for p in (ROOT/'data/paderborn_processed_v1').rglob('*.json'):
        if not {'range_cache','range_header_cache'} & set(p.parts):
            overrides[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    for p in (ROOT/'data/paderborn_processed_v1').iterdir():
        if p.suffix not in {'.sha256','.npz','.py'} or not p.is_file():
            continue
        overrides[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    # The baseline extension retains numerical features and full provenance,
    # including any extraction amendment, but excludes raw signal payloads.
    for p in (ROOT/'data/bearing_baseline_v1').rglob('*'):
        if p.is_file() and p.suffix in {'.json','.sha256','.npz'} and not {'range_cache','range_header_cache'} & set(p.parts):
            overrides[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    files=['src/analysis/prepare_hydraulic_data.py','src/analysis/score_hydraulic_v1.py',
           'src/analysis/build_focused_manuscript.py','src/analysis/hydraulic_manuscript_fragments.py',
           'src/analysis/package_focused_review.py','src/analysis/audit_hydraulic_preprocessing.py','results/hydraulic_independent_preprocessing_audit.json','src/experiments/prepare_hydraulic_v1.py',
           'src/analysis/audit_repeat_condition_v2.py','src/analysis/audit_call_timing_v2.py','src/analysis/repeat_condition_manuscript_v2.py',
           'src/experiments/context_blocked_v1.py','src/analysis/score_context_blocked_v1.py','src/analysis/context_blocked_manuscript_v1.py',
           'src/experiments/run_hydraulic_v1.py','src/experiments/interface_contract.py',
           'tests/test_hydraulic_design.py','tests/test_run_hydraulic_v1.py','tests/test_interface_contract.py',
           'tests/test_repeat_condition_v2.py','tests/test_context_blocked_v1.py',
           'paper/references.bib','paper/JOURNAL_VERIFICATION_2026-09-28.md',
           'paper/FOCUSED_SCIENTIFIC_REVIEW_2026-09-28.md','paper/HYDRAULIC_SOURCE_REVIEW_2026-09-28.md',
           'paper/HYDRAULIC_VALIDATION_EXECUTION_2026-09-28.md','paper/HYDRAULIC_RUNNER_AUDIT_2026-09-28.md',
           'paper/FREE_SCIE_OPTIMIZATION_PLAN_2026-09-28.md','paper/CURRENT_STATUS.md','paper/PAPER_SUMMARY_ZH.md','paper/TABULAR_CONTEXT_LITERATURE_2026-09-28.md',
           'paper/NOVELTY_COMPARISON_2026-09-28.md','paper/CONTEXT_BLOCKED_REVIEW_2026-09-28.md','paper/POINTS_1_2_REVISION_2026-09-28.md',
           'results/points12_independent_numeric_check.json','results/context_blocked_independent_check.json',
           'src/experiments/prepare_policy_hydraulic_v1.py','src/analysis/prepare_paderborn_v1.py',
           'src/experiments/policy_validation_v1.py','src/analysis/score_policy_validation_v1.py','src/analysis/policy_manuscript_v1.py',
           'src/analysis/audit_policy_error_classes_v1.py','results/policy_error_classes_v1/audit.json','src/analysis/independent_policy_check_v1.py','src/analysis/plot_policy_design_v1.py','paper/figures/policy_design_v1.pdf','paper/POLICY_REPRODUCIBILITY_NOTES_2026-09-28.md','results/policy_environment.json','tests/test_policy_validation_v1.py','tests/test_policy_hydraulic_v1.py','tests/test_paderborn_v1.py',
           'paper/THREE_GAPS_EXECUTION_PLAN_2026-09-28.md','paper/THREE_GAPS_DESIGN_REVIEW_2026-09-28.md',
           'paper/ENGINEERING_POLICY_PRIOR_WORK_2026-09-28.md','paper/NEW_MEASURED_SOURCE_FEASIBILITY_2026-09-28.md',
           'paper/POLICY_VALIDATION_PREFLIGHT_2026-09-28.md','paper/POLICY_CLAIM_BOUNDARIES_2026-09-28.md',
           'paper/THREE_GAPS_REVISION_2026-09-28.md','paper/POLICY_HYDRAULIC_RESULT_REVIEW_2026-09-28.md','results/policy_hydraulic_independent_check.json','results/policy_paderborn_independent_check.json']
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'src/analysis').glob('*bearing_baseline*v*.py')]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_bearing_baseline*.py')]
    files += ['paper/BEARING_BASELINE_SOURCE_REVIEW_2026-09-28.md', 'paper/BEARING_BASELINE_CLAIM_PLAN_2026-09-28.md',
              'paper/BEARING_BASELINE_REVISION_2026-09-28.md', 'paper/BEARING_BASELINE_RESULT_REVIEW_2026-09-29.md', 'results/bearing_baseline_independent_check.json',
              'results/bearing_baseline_test_check.json', 'results/bearing_baseline_test_check.log',
              'paper/history/pre_bearing_baseline_v1_20260928/manifest.json',
              'paper/history/pre_bearing_baseline_v1_20260928/paper/focused_submission/quality.json']
    for rel in files:
        overrides[rel]=(ROOT/rel).read_bytes()
    overrides['FOCUSED_REVIEW_README.md']=b'''# Current focused review package

Current manuscript: paper/focused_submission/main.pdf and supplement.pdf.
This local archive includes the prior four-stage audit record and the new,
separately frozen initial hydraulic, sequence-balanced context and new randomized
fixed-policy follow-ups on hydraulic and bearing data, plus
post hoc all-pair and client-timing audits. A result-informed same-feature bearing
benchmark retains six development and nine test specimens, with four additional
joint specimen/regime holdouts. No public release or submission is claimed.

Offline rebuild (Python with numpy, scipy, pandas, sklearn; Tectonic for PDF):
    python results/focused_review/rebuild_context_repeat_audit.py
    python src/experiments/run_hydraulic_v1.py audit
    python src/analysis/score_hydraulic_v1.py
    python src/analysis/audit_repeat_condition_v2.py --include-hydraulic
    python src/analysis/audit_call_timing_v2.py
    python src/experiments/context_blocked_v1.py audit
    python src/analysis/score_context_blocked_v1.py
    python src/experiments/policy_validation_v1.py audit --root results/policy_hydraulic_v1
    python src/experiments/policy_validation_v1.py audit --root results/policy_paderborn_v1
    python src/analysis/score_policy_validation_v1.py --root results/policy_hydraulic_v1
    python src/analysis/score_policy_validation_v1.py --root results/policy_paderborn_v1
    python src/analysis/audit_policy_error_classes_v1.py
    python src/analysis/bearing_baseline_v1.py audit
    python src/analysis/bearing_baseline_v1.py score
    python src/analysis/build_focused_manuscript.py
    cd paper/focused_submission
    tectonic --only-cached main.tex
    tectonic --only-cached supplement.tex

These commands do not invoke a model, download data, or refreeze predictions.
JSON timestamps may change; compare scientific values and CSVs, not timestamp bytes.
The official raw bearing MAT/RAR members, public 73 MB hydraulic archive and original TEP/C-MAPSS raw files are not
duplicated here. This package supports reconstruction from processed features,
fixed inputs and frozen outputs, with source/member hashes and sampling records.
Supervised predictions, complete development-fold results and fitting environment
are frozen. Scoring is offline and does not retrain, change parameters or refreeze.
Independent selected-configuration refitting is documented in its check report.
Raw-data regeneration and the raw-to-prompt audit require separately obtaining
the official source files listed in their manifests.
The earlier hydraulic schedule was stopped: 29 valid responses, two network-related
timeouts, and 32 unattempted requests. It is not a completed five-block validation. New policy studies have separate
protocols, data and response freezes; their completion and failures are in their scores.
Earlier README files describe their historical stage, not current submission status.
The earlier review-package manifest is historical: its manuscript/status hashes
precede this edition. FOCUSED_PACKAGE_MANIFEST.json validates this merged archive.
The packaging script itself needs the separately retained older ZIP; repackaging
is not one of the self-contained scientific reconstruction commands above.
Human scientific verification, authorship/declarations and final approval are pending.
'''
    full=ROOT/'results/industrial_interface_focused_review.zip'
    manifest={}
    with zipfile.ZipFile(ROOT/'results/cross_provider_revision_review.zip') as old, zipfile.ZipFile(full,'w',zipfile.ZIP_DEFLATED) as out:
        for info in old.infolist():
            if info.is_dir():continue
            rel=info.filename.split('/',1)[1]
            if rel in overrides:continue
            if rel.startswith('/') or '..' in Path(rel).parts or Path(rel).name=='.env':
                raise ValueError('Unsafe archive path')
            data=old.read(info)
            out.writestr('industrial_interface_review/'+rel,data)
            manifest[rel]=hashlib.sha256(data).hexdigest()
        for rel,data in sorted(overrides.items()):
            out.writestr('industrial_interface_review/'+rel,data)
            manifest[rel]=hashlib.sha256(data).hexdigest()
        out.writestr('industrial_interface_review/FOCUSED_PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'overleaf':str(overleaf),'full_review':str(full),'full_entries':len(manifest),'zip_bytes':full.stat().st_size}))


if __name__=='__main__':main()
