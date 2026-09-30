"""Render separate historical and tool-enabled Claude appendices from audited records.

No model calls or eligibility promotion. Historical snapshots remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.render_revision_v3_paper import latex_table
BASE = ROOT / 'results/claude_descriptive_recheck_2026-09-28'
CONVERSATION = ROOT / 'results/claude_conversation_exploratory_v1/audit_2026-09-28'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_checked():
    for base, prefix in [(BASE, ''), (CONVERSATION, 'snapshot/')]:
        manifest = json.loads((base / 'snapshot_manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if sha(base / (prefix + name)) != expected:
                raise ValueError('Changed post-score audit snapshot: ' + name)
    legacy = json.loads((BASE / 'descriptive_metrics.json').read_text())
    current = json.loads((CONVERSATION / 'corrected_descriptive_scores.json').read_text())
    if legacy['script_sha256'] != sha(ROOT / 'src/analysis/recheck_claude_descriptive.py'):
        raise ValueError('Legacy descriptive auditor changed; recheck required')
    if current['auditor_sha256'] != sha(ROOT / 'src/analysis/audit_conversation_exploratory.py'):
        raise ValueError('Conversation auditor changed; recheck required')
    if legacy['eligible_for_primary_comparison'] or current['eligible_for_controlled_primary']:
        raise ValueError('This renderer must not promote Claude eligibility')
    for report, base, key in [(legacy, BASE, 'source_snapshot_manifest_sha256'), (current, CONVERSATION, 'snapshot_manifest_sha256')]:
        if report[key] != sha(base / 'snapshot_manifest.json'):
            raise ValueError('Analysis/snapshot linkage mismatch')
    return legacy, current

def build():
    legacy, current = load_checked()
    names = {'sonnet': 'Sonnet alias', 'opus': 'Opus alias'}
    tasks = {'cmapss_clean': 'C-MAPSS', 'tep_clean': 'TEP clean', 'tep_misnamed': 'TEP corrupted'}
    def display(metric):
        return f"${metric['mean_pct']:.1f} \\pm {metric['sample_sd_pp']:.1f}$"
    rows = [[names[r['model_alias']], tasks[r['condition']], f"{r['accuracy']['n_available']}/3", display(r['accuracy']), display(r['macro_f1'])] for r in current['aggregates']]
    table = latex_table(
        'Tool-enabled conversational records: descriptive scores of arrays satisfying the exact-order output contract. '
        'Means and sample SDs (denominator $n-1$) are computed from unrounded session scores; accuracy and macro F1 are on a percentage scale. '
        'Available/planned counts describe output structure, not verified inference provenance. '
        'All records remain outside the controlled comparisons.',
        'tab:claude_conversation', 'lllrr', ['Requested alias', 'Condition', 'Available', 'Accuracy', 'Macro F1'], rows)
    pairs = []
    for scope, values in [('Exact order', current['paired_summaries']), ('ID-aligned audit', current['posthoc_id_alignment_sensitivity']['paired_summaries'])]:
        for model, r in values.items():
            pairs.append([scope, names[model], ', '.join('R'+str(x) for x in r['paired_repeats']),
                          f"{r['accuracy_delta_misnamed_minus_clean_pp']:.1f}", ', '.join(f'{x}/70' for x in r['flip_counts'])])
    paired_table = latex_table(
        'Matched-repeat contrasts for the separate conversational records. Delta is corrupted minus clean accuracy in percentage points. '
        'Only repeats with both arrays available enter each contrast. The post-hoc ID-aligned audit includes one complete but misordered Sonnet array; '
        'it does not repair the stored response or retrospectively waive its output-contract failure. Repeats reuse the same 70 runs.',
        'tab:claude_conversation_pairs', 'llllr', ['Analysis', 'Requested alias', 'Pairs', '$\\Delta$ (pp)', 'Label flips'], pairs)
    fable = next(x for x in legacy['aggregates'] if x['model_label_unverified'] == 'claude-fable-5-1' and x['condition'] == 'tep_misnamed')['accuracy']
    txt = r'''\section{Claude records outside the controlled comparisons}
\label{app:claude_reported}
\subsection{Historical collection and corrected arithmetic}
An earlier collection contains 27 arrays under caller-written Sonnet, Opus and Fable identifiers. A retrospective arithmetic check reproduces its supplied means, but only 23/27 local artifact chains are internally consistent; none has sufficient evidence for inclusion in the controlled comparison. The record labelled Fable uses the invocation alias \texttt{opus}, without a verified service-returned identity. It cannot establish a third distinct tested model. The updated Fable corrupted-condition accuracies are 91.4\%, 95.7\% and 100.0\% (mean ''' + f"{fable['mean_pct']:.1f}" + r'''\%, sample SD ''' + f"{fable['sample_sd_pp_ddof1']:.1f}" + r''' percentage points), correcting the earlier report of three perfect results. The original reported summaries and the subsequent correction are retained in separate audit snapshots. These post-score copies do not establish a prediction freeze before scoring.

\subsection{Separate tool-enabled conversational exploration}
A later exploration specifies 18 records: the requested \texttt{sonnet} and \texttt{opus} aliases, three conditions and three repeats. It reuses the original 60 C-MAPSS engines and 70 TEP runs. The organizer reports using Claude Code desktop Agent calls, with agents loading the prompt files through the Read tool. This is a tool-enabled conversational workflow, not the text-only, tool-disabled interface of the four controlled stages. Exact server-resolved model versions and exclusion of inherited context have not been verified.

The submitted directory contains 16 complete ID sets and two recorded failures: Sonnet corrupted R2 returned an off-task orchestration message, and Opus corrupted R3 returned 69/70 records. Their partial or off-task outputs are not scored. An additional format check finds that Sonnet clean R1 swaps the records at positions 17 and 18 despite the prompt's exact-order requirement. Consequently, 15/18 arrays satisfy the output contract, covering 990/1,200 planned labels. This is structural coverage only; it does not certify execution provenance. Table~\ref{tab:claude_conversation} reports descriptive statistics for those arrays.

''' + table + r'''

For transparency, a separate post-hoc ID-aligned calculation includes the complete but misordered Sonnet clean R1 array, without changing its saved bytes or protocol status. Under that sensitivity calculation, the Sonnet clean mean is 64.8\% (sample SD 4.6 percentage points; three arrays), and macro F1 is 61.1\% (sample SD 7.9). This recovers the supplied descriptive summary and raises arithmetically usable coverage to 1,060/1,200 labels; it is not an alternative success definition chosen for the main study.

''' + paired_table + r'''

The supplied 48.4-percentage-point Sonnet decrease subtracts means over different repeat sets. When the ID-aligned analysis uses R1 and R3 on both sides, clean and corrupted means are 62.1\% and 16.4\%, respectively: a 45.7-point decrease, with 47/70 and 37/70 label changes. The exact-order analysis retains only R3, with the same observed decrease but only one paired repeat. For the Opus alias, R1 and R2 each have perfect scores and zero paired label changes; corrupted R3 is unavailable. Neither two equal scores nor zero observed flips establishes deterministic inference, general robustness, or absence of descriptor use.

\subsection{Provenance limits and interpretation}
The submitted scorer labels the two documented failures as \texttt{not\_run}, while the session registry still labels all 18 jobs as running. The audit retains those original files and supplies a separate corrected terminal-status table. The submitted prompt hashes match the current files' MD5 digests; the audit additionally records SHA-256. Current-file agreement does not establish what each agent actually received. Complete original inputs, Read return contents, response transcripts and a prediction manifest demonstrably predating scoring are absent from the supplied directory. The off-task response mentions other agents and a retry, but without the underlying transcript this does not establish which additional calls occurred or whether their outputs entered another record.

Accordingly, these records are reported only as exploratory observations with unresolved execution provenance. They neither enlarge the four-stage request/source totals nor support a controlled cross-provider effect estimate. The observed Sonnet changes motivate further testing of this descriptor intervention; they do not quantify reliance on names or identify an internal mechanism. Opus invariance in the two available pairs is a property of those saved responses. Missingness, tools, inherited context, service identity and input delivery remain possible explanations that cannot be separated here. Retrospective arithmetic verification is distinct from prospective experiment validation.
'''
    return {'CLAUDE_REPORTED_APPENDIX': txt}

if __name__ == '__main__':
    (ROOT / 'paper/claude_reported_text.json').write_text(json.dumps(build(), indent=2) + '\n')
    print('Rendered historical correction and separately bounded conversation appendix.')
