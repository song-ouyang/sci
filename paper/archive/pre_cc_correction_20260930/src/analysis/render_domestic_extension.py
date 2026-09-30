"""Render the separately frozen cross-provider addition; never invoke inference."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.audit_domestic_extension import verify
from src.analysis.render_revision_v3_paper import sha, percent, interval, latex_table
BASE = ROOT / 'results/domestic_extension_v2'
NAMES = {'deepseek': 'DeepSeek V4 Pro', 'qwen': 'Qwen3.8 Max'}


def build():
    plan, outcomes, manifest = verify(BASE)
    r = json.loads((BASE/'analysis/scores.json').read_text())
    assert r['freeze_sha256'] == sha(BASE/'frozen_manifest.json')
    assert r['scorer_sha256'] == sha(ROOT/'src/analysis/audit_domestic_extension.py')
    assert r['metric_implementation_sha256'] == sha(ROOT/'src/analysis/score_revision_v3.py')
    assert r['valid_sessions'] == manifest['valid_sessions'] == 15
    assert r['valid_predictions'] == manifest['valid_predictions'] == 990
    def metric(p, g, repeat=1):
        return next(x for x in r['metrics'] if x['provider']==p and x['group']==f'{g}_r{repeat}')
    def pair(p, repeat=1):
        return next(x for x in r['paired'] if x['provider']==p and x['repeat']==repeat)
    q, d = pair('qwen'), pair('deepseek', 2)
    text = {}
    text['DOMESTIC_ABSTRACT'] = (
        f"An additional DeepSeek/Qwen study yields {r['valid_sessions']}/18 valid responses: "
        f"Qwen's primary corruption loss is {q['accuracy_difference_pp']:.1f} percentage points, "
        "whereas DeepSeek's primary corrupted response is invalid."
    )
    stage_table = latex_table(
        'Evidence stages and response coverage. Labels are repeated classifications, not independent observations. '
        'The aborted API configuration trial is disclosed separately and is not included in these totals.',
        'tab:stages', 'lrrr', ['Stage', 'Requests', 'Valid batches', 'Valid/requested labels'],
        [['Original study', '27', '26', '1,740/1,800'],
         ['Representation/context follow-up', '84', '84', '2,604/2,604'],
         ['Cross-provider API addition', '18', '15', '990/1,200'],
         ['Total (130 unique source units)', '129', '125', '5,334/5,604']])
    text['STAGE_TABLE'] = stage_table
    text['DOMESTIC_METHODS'] = r'''\subsection{Exploratory cross-provider API addition}
After the original results and representation/context follow-up, we evaluate two additional service aliases, \texttt{deepseek-v4-pro} and \texttt{qwen3.8-max}, through their provider APIs. This addition repeats the original three task/condition groups and three fresh requests per group, for 18 requests. The original observations, full batch text, order, labels and instruction wording are unchanged. The API system role replaces the original CLI developer-instruction role; no tools are supplied. Repeat 1 remains the primary descriptive comparison, designated in the addition's plan before its requests. No representation/context follow-up conditions are added for these services.

The addition uses a separately versioned, budget-corrected plan. An earlier configuration used a 16,384-token setting whose scope differed between providers; four completed batches were invalid and two active requests were stopped, leaving twelve unattempted jobs. Those records are retained, were not classification-scored, and are excluded from the revised addition. All 18 revised requests were newly scheduled, rather than selectively replacing failures. Both revised payloads cap combined reasoning and final output at 65,536 tokens. DeepSeek uses its high reasoning setting; Qwen additionally caps thinking at 16,384 tokens. These provider-defined controls do not equalize compute or decoding behavior. The correction followed observed execution failures and is not an original-study protocol choice.

Requests follow fixed shuffled submission orders with at most two concurrent sessions per provider and no automatic retries or semantic repair. Complete raw streams, submitted payloads, provider-returned aliases, finish reasons, usage and derived outputs are retained. A manifest freezes every terminal outcome before scoring. The same whole-response validation and paired source-unit analysis apply; an invalid primary is never replaced by a later repeat. The first three stages in Table~\ref{tab:stages} reuse the original 130 source units; the separately described fourth stage adds new sources. None establishes an immutable model-weight snapshot.

'''
    primary_rows, repeat_rows = [], []
    for p in NAMES:
        for g, label in [('cmapss_clean','C-MAPSS'), ('tep_clean','TEP clean'), ('tep_misnamed','TEP corrupted')]:
            x = metric(p,g)
            primary_rows.append([NAMES[p],label,percent(x['accuracy']),interval(x.get('accuracy_wilson95')),percent(x['macro_f1'])])
            rr=[metric(p,g,i) for i in (1,2,3)]
            repeat_rows.append([NAMES[p],label,*[percent(x['accuracy']) for x in rr],f"{sum(x['status']=='valid' for x in rr)}/3"])
    primary_table = latex_table(
        'Cross-provider addition, repeat 1. Accuracy and macro F1 are percentages; accuracy intervals are 95\\% Wilson summaries conditional on the observed grouped response. '
        'N/A denotes a failed whole batch. The same RF/SVM references in Table~\ref{tab:primary} apply.',
        'tab:domestic_primary','llrrr',['Service','Task/condition','Acc.','95\\% interval','Macro F1'],primary_rows)
    repeat_table = latex_table(
        'All revised API sessions: accuracy (\\%) and valid-batch counts. Repeats use the same sources and do not increase sample size. '
        'No failed response is scored, repaired, or replaced; one available value cannot establish repeatability.',
        'tab:domestic_repeat','llrrrr',['Service','Task/condition','R1','R2','R3','Valid'],repeat_rows)
    text['DOMESTIC_RESULTS'] = (
        r'\subsection{Cross-provider coverage and mapping sensitivity}'+'\n'
        'The revised addition produces 15/18 valid responses and 990/1,200 scorable classifications. '
        'DeepSeek yields 7/9 valid batches and Qwen 8/9. All three invalid batches occur under TEP descriptor corruption: '
        'DeepSeek repeats 1 and 3 exhaust the 65,536-token completion budget with a length finish; '
        'Qwen repeat 2 finishes normally but adds prose before the requested JSON array. '
        'The latter is a response-contract failure, and its apparent labels are not extracted or scored. '
        'Coverage and accuracy therefore describe different outcomes.\n\n'+primary_table+'\n'
        f"DeepSeek's primary clean accuracies are {percent(metric('deepseek','cmapss_clean')['accuracy'])}\\% on C-MAPSS and "
        f"{percent(metric('deepseek','tep_clean')['accuracy'])}\\% on TEP; the corresponding Qwen values are "
        f"{percent(metric('qwen','cmapss_clean')['accuracy'])}\\% and {percent(metric('qwen','tep_clean')['accuracy'])}\\%. "
        f"For Qwen, the primary clean-minus-corrupted TEP accuracy difference is {q['accuracy_difference_pp']:.1f} pp "
        f"(paired bootstrap 95\\% interval [{q['accuracy_difference_pp_bootstrap95'][0]:.1f}, {q['accuracy_difference_pp_bootstrap95'][1]:.1f}]); "
        f"{round(q['prediction_disagreement']*70)}/70 labels change. DeepSeek's primary paired contrast remains N/A.\n\n"
        +repeat_table+'\n'
        "DeepSeek's clean TEP accuracy ranges from 34.3\\% to 74.3\\% across the three valid repeats, a 40.0-pp range. "
        f"Its only valid corrupted batch is repeat 2; the matched accuracy difference is {d['accuracy_difference_pp']:.1f} pp "
        f"(95\\% interval [{d['accuracy_difference_pp_bootstrap95'][0]:.1f}, {d['accuracy_difference_pp_bootstrap95'][1]:.1f}]). "
        f"Qwen's second available pair, repeat 3, differs by {pair('qwen',3)['accuracy_difference_pp']:.1f} pp. "
        "These secondary contrasts retain their repeat identities. Full macro-F1 intervals, class-level confusion matrices, paired differences and case records accompany the tables."
    )
    text['DOMESTIC_DISCUSSION'] = r'''\subsection{What the cross-provider addition establishes}
The API addition broadens the observed service families and preserves the same numerical cases and descriptor intervention. Its valid pairs show lower accuracy under descriptor corruption, while its incomplete and nonconforming responses expose an additional interface failure. Reporting only valid classification scores would omit that failure. Because missingness occurs under corruption, available paired scores need not represent the outcomes that would have been obtained from every scheduled request; N/A is neither zero accuracy nor evidence of robustness.

This extension does not isolate provider identity as a causal factor. The original services and the API additions differ in transport, instruction role, reasoning controls and potentially hidden service settings. The budget was revised after a configuration failure, and the addition was planned after seeing the original study. It therefore supports exploratory comparisons of recorded service configurations, rather than an equal-compute model ranking. The representation/context controls remain limited to Sol and Astra; their findings cannot be transferred to DeepSeek or Qwen from the present addition alone.'''
    text['DOMESTIC_CONCLUSION'] = (
        'The separately frozen API addition extends the mapping-sensitivity evidence to other service families while showing '
        'that response availability must be reported alongside conditional classification scores.'
    )
    text['DOMESTIC_AUDIT'] = r'''The cross-provider API addition has its own plan, frozen output manifest and separately archived aborted configuration trial. Its native response streams and terminal failures are included in the review package. A historical Claude Agent-tool collection, including a manually regenerated batch read in chunks in an existing conversation, is excluded from the controlled comparisons. The available records do not establish isolated fresh inference, authenticated complete-input delivery, or a pre-score output freeze. Removing an integrity flag or changing a response hash cannot establish those properties. This exclusion concerns provenance and protocol comparability, not a claim that every submitted label is wrong; the artifacts and audit are preserved separately. The following appendix distinguishes the corrected historical summary from the later tool-enabled conversational exploration and reports their unresolved provenance limits.'''
    return text

if __name__=='__main__':
    (ROOT/'paper/domestic_extension_text.json').write_text(json.dumps(build(),indent=2)+'\n')
    print('Generated cross-provider inserts from verified frozen outputs.')
