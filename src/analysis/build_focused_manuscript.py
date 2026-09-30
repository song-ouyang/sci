"""Build an evidence-focused manuscript and complete companion report offline.

The existing full-report generator and frozen experiments are not modified.
All injected tables are copied from the synchronized full report; numerals in
the new abstract are bound to the fourth-stage score artifact.
"""
import hashlib
import json
from pathlib import Path
import re
from hydraulic_manuscript_fragments import make_fragments
from repeat_condition_manuscript_v2 import make_repeat_fragments
from context_blocked_manuscript_v1 import make_context_fragments
from policy_manuscript_v1 import make_policy_fragments
from bearing_baseline_manuscript_v1 import make_baseline_fragments
from evidence_value_manuscript import make_evidence_value_fragments

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/focused_submission'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(source, kind, label):
    matches = [x for x in re.findall(
        r'\\begin\{' + kind + r'\}.*?\\end\{' + kind + r'\}', source, re.S
    ) if r'\label{' + label + '}' in x]
    if len(matches) != 1:
        raise ValueError(f'Expected one {kind} with label {label}, got {len(matches)}')
    return matches[0]


def check_tex(text, bibkeys):
    if re.search(r'@@[A-Z_]+@@', text):
        raise ValueError('Unresolved template marker')
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate labels')
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text))
    if refs - set(labels):
        raise ValueError(f'Unresolved references: {refs - set(labels)}')
    cites = {key.strip() for group in re.findall(r'\\cite[pt]?\{([^}]+)\}', text) for key in group.split(',')}
    if cites - bibkeys:
        raise ValueError(f'Unknown citations: {cites-bibkeys}')
    return {'figures': text.count(r'\begin{figure}'), 'tables': text.count(r'\begin{table}'), 'cited_references': len(cites)}


def main():
    source_path = ROOT / 'paper/main.tex'
    source = source_path.read_text()
    if any((ROOT / 'paper' / name).read_text() != source for name in ['main_inline.tex', 'main_compile.tex']):
        raise ValueError('Full-report source aliases disagree')
    score_path = ROOT / 'results/independent_validation/analysis/scores.json'
    scores = json.loads(score_path.read_text())
    base = ROOT / 'results/independent_validation'
    if scores['protocol_sha256'] != digest(base / 'protocol.json') or scores['frozen_manifest_sha256'] != digest(base / 'predictions/frozen_manifest.json'):
        raise ValueError('Independent scores refer to different frozen artifacts')
    def row(kind, condition):
        return next(x for x in scores[kind] if x['model'] == 'gpt-5.6-sol' and x['condition'] == condition and x['repeat'] == 1)
    clean = row('metrics', 'clean')
    rotation = row('metrics', 'rotation')
    focal = row('focal', 'focal_only')
    back = row('focal', 'b_back')
    front = row('focal', 'b_front')
    audit_path = ROOT / 'results/focused_review/context_repeat_audit.json'
    audit = json.loads(audit_path.read_text())
    for name, value in audit['input_hashes'].items():
        if digest(ROOT/name) != value:
            raise ValueError('Repeat audit source changed: ' + name)
    labels = {'clean': 'Clean (70)', 'focal_only': 'Focal-only (21)', 'b_front': 'B-front (42)', 'b_back': 'B-back (42)'}
    table = [r'\begin{table}[htbp]\centering\small',
        r'\caption{Post hoc identical-input control: focal-label flips out of the same 21 cases across repeated requests. Parentheses give full request size. This reanalysis adds no calls or sources.}',
        r'\label{tab:repeat_control}', r'\begin{tabular}{@{}llrrr@{}}\toprule',
        r'Service & Input & R1--R2 & R1--R3 & R2--R3 \\\midrule']
    for model, short in [('gpt-5.6-sol','Sol'), ('gpt-6-astra','Astra')]:
        for condition, label in labels.items():
            counts = [next(x['flips'] for x in audit['identical_input_flips'] if x['model']==model and x['condition']==condition and (x['left_repeat'],x['right_repeat'])==pair) for pair in [(1,2),(1,3),(2,3)]]
            table.append(short+' & '+label+' & '+' & '.join(map(str,counts))+r' \\')
    table += [r'\bottomrule\end{tabular}\end{table}']
    repeat_table = '\n'.join(table)
    text = (OUT / 'main_template.tex').read_text()
    replacements = {
        'SOL_CLEAN': f"{clean['accuracy']*100:.1f}",
        'SOL_ROTATION': f"{rotation['accuracy']*100:.1f}",
        'SOL_DELTA': f"{(clean['accuracy']-rotation['accuracy'])*100:.1f}",
        'SOL_FOCAL': f"{focal['correct']}/{focal['n']}",
        'SOL_FRONT': f"{front['correct']}/{front['n']}",
        'REPEAT_CONTROL': repeat_table,
        'SOL_BACK': f"{back['correct']}/{back['n']}",
        'STAGES': environment(source, 'table', 'tab:stages'),
        'ORIGINAL_MAPPING': environment(source, 'table', 'tab:paired'),
        'NEW_MAPPING': environment(source, 'table', 'tab:ind_mapping'),
        'CONTEXT': environment(source, 'table', 'tab:ind_context'),
        'TRANSFER': environment(source, 'table', 'tab:ind_transfer'),
        'CONTROL_FIGURE': environment(source, 'figure', 'fig:independent'),
    }
    measured, measured_supplement = make_fragments()
    replacements.update(measured)
    repeat_fragments, repeat_supplement = make_repeat_fragments()
    replacements.update(repeat_fragments)
    blocked_fragments, blocked_supplement = make_context_fragments()
    replacements.update(blocked_fragments)
    policy_fragments, policy_supplement = make_policy_fragments()
    replacements.update(policy_fragments)
    baseline_fragments, baseline_supplement = make_baseline_fragments()
    replacements.update(baseline_fragments)
    evidence_fragments, evidence_supplement = make_evidence_value_fragments()
    replacements.update(evidence_fragments)
    replacements['CONTRIBUTION_ADDENDUM'] = (OUT/'contribution_addendum_2026-09-30.tex').read_text()
    replacements['PRIOR_WORK_COMPARISON'] = (OUT/'prior_work_comparison.tex').read_text()
    replacements = {key:value for key,value in replacements.items() if '@@'+key+'@@' in text}
    if set(re.findall(r'@@([A-Z_]+)@@', text)) != set(replacements):
        raise ValueError('Template and replacement keys disagree')
    for key, value in replacements.items():
        text = text.replace('@@' + key + '@@', value)
    supplement = source.replace(
        r'\title{Sensor Mapping and Context Effects in Large Language Models for Industrial Monitoring}',
        r'\title{Supplementary Report: Sensor Mapping and Context Effects in Industrial Monitoring}'
    ).replace(r'\maketitle', r'\maketitle' + '\n' +
        r'\noindent This companion preserves the complete four-stage report, including extended methods, all repeated outcomes and excluded exploratory records. Its original internal section and table numbering is retained. It accompanies the focused manuscript, ``Auditing Sensor Mapping and Context Dependence in LLM-Based Industrial Diagnosis.''' + '\n')
    supplement = supplement.replace(r'\begin{document}', r'\graphicspath{{../}}' + '\n' + r'\begin{document}')
    supplement = supplement.replace('pdftitle={Sensor Mapping and Context Effects in Large Language Models for Industrial Monitoring}',
                                    'pdftitle={Supplementary Report: Sensor Mapping and Context Effects in Industrial Monitoring}')
    supplement = supplement.replace(r'\bibliography{references}', r'\bibliography{../references}')
    addendum = measured_supplement + r'\subsection{Post hoc identical-input focal audit}' + '\n' + r'This analysis reuses frozen simulation outputs. It adds no observations or model calls. Identical-request variability can reach the primary context contrast, so a context effect separated from service variation is not estimated.' + '\n\n' + repeat_table + '\n' + r'\FloatBarrier' + '\n' + repeat_supplement
    addendum += '\n' + blocked_supplement + '\n' + policy_supplement + '\n' + baseline_supplement + '\n' + evidence_supplement
    anchor = supplement.rfind(r'\begingroup')
    if anchor < 0: raise ValueError('Missing supplement bibliography anchor')
    supplement = supplement[:anchor] + addendum + supplement[anchor:]
    supplement = supplement.replace('This companion preserves the complete four-stage report,', 'This companion preserves the archived four-stage simulation report and adds a separately frozen single-rig follow-up in Appendix C. The archived report below retains its original scope and language,')
    supplement = supplement.replace('Auditing Sensor Mapping and Context Dependence in LLM-Based Industrial Diagnosis', 'Auditing Sensor Mapping, Context and Repeatability in LLM-Based Industrial Diagnosis')
    supplement = supplement.replace('single-rig follow-up in Appendix C.', 'single-rig follow-up in Appendix C, plus a post hoc repetition and timing audit in Appendix D.')
    supplement = supplement.replace('timing audit in Appendix D.', 'timing audit in Appendix D and a separately frozen sequence-balanced follow-up in Appendix E.')
    supplement = supplement.replace('Auditing Sensor Mapping, Context and Repeatability in LLM-Based Industrial Diagnosis', 'Sensor Mapping, Context and Decision Checks in LLM-Based Industrial Diagnosis')
    supplement = supplement.replace('follow-up in Appendix E.', 'follow-up in Appendix E and new randomized fixed-policy studies in Appendix F.')
    supplement = supplement.replace('studies in Appendix F.', 'studies in Appendix F. Appendix G adds a result-informed, same-feature supervised bearing benchmark with fixed physical-specimen holdouts.')
    supplement = supplement.replace('fixed physical-specimen holdouts.', 'fixed physical-specimen holdouts. Appendix H adds result-informed class-transition and decision-loss audits of the same frozen records.')
    bib = ROOT / 'paper/references.bib'
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib.read_text()))
    info = {'main': check_tex(text, bibkeys), 'supplement': check_tex(supplement, bibkeys)}
    (OUT / 'main.tex').write_text(text)
    (OUT / 'supplement.tex').write_text(supplement)
    inputs = [source_path, score_path, bib, OUT / 'main_template.tex', Path(__file__), audit_path, ROOT/'src/analysis/hydraulic_manuscript_fragments.py', ROOT/'results/hydraulic_validation_v1/analysis/scores.json', ROOT/'results/hydraulic_validation_v1/protocol.json']
    inputs += [ROOT / f'results/{stage}/analysis/scores.json' for stage in ['revision_v3', 'mechanism_extension', 'domestic_extension_v2']]
    inputs += [OUT/'prior_work_comparison.tex', ROOT/'src/analysis/repeat_condition_manuscript_v2.py', ROOT/'src/analysis/audit_repeat_condition_v2.py', ROOT/'results/repeat_condition_v2/audit.json', ROOT/'src/analysis/audit_call_timing_v2.py', ROOT/'results/call_timing_v2/timing.json']
    inputs += [ROOT/'src/analysis/context_blocked_manuscript_v1.py', ROOT/'results/context_blocked_v1/protocol.json', ROOT/'results/context_blocked_v1/analysis/scores.json', ROOT/'results/context_blocked_v1/predictions/frozen_manifest.json']
    inputs += [ROOT/'src/analysis/audit_policy_error_classes_v1.py', ROOT/'results/policy_error_classes_v1/audit.json']
    inputs += [ROOT/'src/analysis/policy_manuscript_v1.py', ROOT/'src/analysis/plot_policy_design_v1.py', ROOT/'paper/figures/policy_design_v1.pdf']
    inputs += [ROOT/p for p in ['src/analysis/bearing_baseline_manuscript_v1.py', 'src/analysis/bearing_baseline_v1.py', 'results/bearing_baseline_v1/protocol.json', 'results/bearing_baseline_v1/prediction_manifest.json', 'results/bearing_baseline_v1/analysis/scores.json', 'results/bearing_baseline_v1/fit/selections.json', 'results/bearing_baseline_v1/fit/cv.json', 'results/bearing_baseline_v1/fit/structural_checks.json', 'data/bearing_baseline_v1/plan.json', 'data/bearing_baseline_v1/audit.json']]
    inputs += [OUT/'bearing_baseline_interpretation.tex']
    for dataset in ['hydraulic','paderborn']:
        inputs += [ROOT/f'results/policy_{dataset}_v1'/f for f in ['protocol.json','analysis/scores.json','predictions/frozen_manifest.json']]
        inputs += sorted((ROOT/f'results/policy_{dataset}_v1/analysis').glob('*.csv'))
    inputs += [ROOT/'src/analysis/evidence_value_manuscript.py', ROOT/'src/analysis/audit_evidence_boundaries.py', ROOT/'results/evidence_boundary_audit_2026-09-30.json', ROOT/'src/analysis/audit_decision_value.py', OUT/'contribution_addendum_2026-09-30.tex', OUT/'contribution_sources_2026-09-30.json']
    inputs += sorted((ROOT/'results/decision_value_audit_2026-09-29').glob('*'))
    inputs += [ROOT/'paper/figures/decision_loss_2026-09-30.pdf']
    manifest = {'status': 'author_review_draft', 'new_inference_requests': 0,
                'scope': 'Frozen simulation/measured evidence and specimen-disjoint baselines, supplemented by result-informed class-transition and decision-loss audits; no new inference or independent experimental validation.',
                'inputs': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
                'outputs': {str(p.relative_to(ROOT)): digest(p) for p in [OUT/'main.tex', OUT/'supplement.tex']},
                'checks': info, 'human_scientific_verification': 'pending',
                'visual_review': 'not_yet_performed', 'submission_ready': False}
    (OUT / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
