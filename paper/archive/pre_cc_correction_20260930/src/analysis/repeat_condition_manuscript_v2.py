"""Offline manuscript rendering from the separately versioned repeat audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_repeat_fragments():
    audit = json.loads((ROOT/'results/repeat_condition_v2/audit.json').read_text())
    if audit['analysis_code_sha256'] != digest(ROOT/'src/analysis/audit_repeat_condition_v2.py'):
        raise ValueError('Repeat analysis code differs from its recorded version')
    for study in audit['studies']:
        for rel, expected in study['input_hashes'].items():
            if digest(ROOT/rel) != expected:
                raise ValueError('Repeat analysis source changed: '+rel)
    timing=json.loads((ROOT/'results/call_timing_v2/timing.json').read_text())
    if timing['analysis_code_sha256'] != digest(ROOT/'src/analysis/audit_call_timing_v2.py'):
        raise ValueError('Timing analysis code differs from its recorded version')
    for rel,expected in timing['input_source_hashes'].items():
        if digest(ROOT/rel)!=expected:
            raise ValueError('Timing audit source changed: '+rel)
    bback=timing['condition_timing']['gpt-5.6-sol/b_back']
    bfront=timing['condition_timing']['gpt-5.6-sol/b_front']
    if not bback['start_max_utc'] < bfront['start_min_utc']:
        raise ValueError('The reported chronological separation no longer holds')
    rows = [r for r in audit['contrasts'] if r['study']=='tep_three_repeats']
    if len(rows)!=20:
        raise ValueError('Expected all 20 model/contrast combinations')
    def get(model, contrast):
        return next(r for r in rows if r['model']==model and r['comparison']==contrast)
    def count_range(r, prefix):
        lo,hi = (r[prefix+'_flip_rate_'+k] for k in ['min','max'])
        if lo is None or hi is None:
            return 'N/A'
        a,b = (round(v*r['n_cases']) for v in [lo,hi])
        return str(a) if a==b else f'{a}--{b}'
    labels = {'clean__rotation':'Clean / rotation', 'clean__random_1':'Clean / random 1',
              'clean__random_2':'Clean / random 2', 'clean__random_3':'Clean / random 3',
              'clean__same_unit_swap':'Clean / same-unit swap', 'clean__pair_shuffle':'Clean / pair reorder',
              'a_front__a_back':'A: front / back', 'b_front__b_back':'B: front / back',
              'a_front__b_front':'Front: A / B', 'a_back__b_back':'Back: A / B'}
    table = [r'\begin{table}[htbp]\centering\small',
             r'\caption{Post hoc call-pair audit on new TEP sources. Counts are out of $n$ cases, not independent call pairs. Within L/R gives the range across three identical-input pairs on each side; across gives the range across all nine between-condition pairs. D: both sides agree internally over their three calls but give different labels. V: at least one side varies. The remaining $n-D-V$ cases agree in all six calls. These observed categories do not remove service-time confounding.}',
             r'\label{tab:repeat_separation}',
             r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
             r'Service & Left / right input & $n$ & Within L / R & Across & D & V \\\midrule']
    for model,short in [('gpt-5.6-sol','Sol'),('gpt-6-astra','Astra')]:
        for key,label in labels.items():
            r=get(model,key)
            table.append(f"{short} & {label} & {r['n_cases']} & {count_range(r,'left_within')} / {count_range(r,'right_within')} & {count_range(r,'cross')} & {r['observed_repeat_stable_different']} & {r['any_side_unstable']}"+r' \\')
        if model=='gpt-5.6-sol':table.append(r'\midrule')
    table += [r'\bottomrule\end{tabular}\end{table}']
    table='\n'.join(table)
    rotation=get('gpt-5.6-sol','clean__rotation')
    bpos=get('gpt-5.6-sol','b_front__b_back')
    aback=get('gpt-6-astra','a_back__b_back')
    methods=r'''\subsection{Post hoc comparison of within-input and between-input variation}
The primary same-numbered-repeat comparisons are retained, but a second, explicitly post hoc audit uses every recorded pairing. For each of the six clean-versus-mapping contrasts and four fixed-size context contrasts, each service supplies three calls per condition. We enumerate the three within-condition pairs on each side and all nine between-condition pairs, matching observations by ID. These nine comparisons reuse six calls and the same cases; they are not nine independent replications. We report the range of changed-label counts without subtracting a variability baseline, testing significance, or treating overlapping ranges as evidence of equivalence.

For case $i$ in condition $c$, let $\mathcal L_i^c=\{\hat y_{i1}^c,\hat y_{i2}^c,\hat y_{i3}^c\}$. A case has an \emph{observed repeat-stable difference} if both condition-specific sets contain one label but those labels differ. The other complete cases either agree in all six calls or vary within at least one condition. Incomplete planned repeats form a separate unavailable category. This classification uses predicted labels, not correctness, and does not select cases for a higher accuracy estimate. It is an audit summary rather than a novel statistical estimator or validated acceptance rule.

The source-disjoint study's repeat numbers identify separate scheduled jobs, not matched temporal blocks. A separate audit checks client-side request timestamps. All pairings contextualize finite-call repeatability but cannot remove service drift, unexposed generation settings or hidden model updates. Three agreeing labels do not establish population stability, a nonzero stable-difference count is not a causal-effect lower bound, and zero such cases does not establish absence of an input effect. The hydraulic audit separately uses its two complete execution blocks, keeps explicit duplicate conditions separate and retains the original missing R3--R5 cells in the full results. Two- and three-repeat categories are not compared as equivalent tests.
'''
    results=(r'\subsection{Which differences persist across the observed calls?}'+'\n'+table+'\n\n'+
        f"For Sol, clean versus cyclic descriptor rotation yields {rotation['observed_repeat_stable_different']}/{rotation['n_cases']} observed repeat-stable differences, "
        f"{rotation['observed_repeat_stable_same']} cases agreeing in all six calls and {rotation['any_side_unstable']} cases varying within at least one condition. "
        f"Across-condition counts range from {count_range(rotation,'cross')} of 70, while identical clean requests differ on {count_range(rotation,'left_within')} "
        f"and identical rotated requests on {count_range(rotation,'right_within')}. "
        "The other five mapping contrasts have no observed repeat-stable differences. In particular, the source-disjoint correct-pair reorder has 67 cases agreeing across all calls and three with within-condition variation. "
        "Thus the cyclic-mapping finding has a specified finite-repeat component, but it does not establish a uniform mapping effect across permutations.\n\n"+
        f"None of Sol's four fixed-size context contrasts has an observed repeat-stable difference. In the B-front/B-back comparison, "
        f"{bpos['observed_repeat_stable_same']} of 21 cases agree in all six calls and {bpos['any_side_unstable']} vary within at least one condition; "
        f"all nine cross-condition comparisons range from {count_range(bpos,'cross')} changed labels. "
        "This identifies the cases behind the large primary contrast without attributing their changes solely to context. "
        f"Astra agrees across all tested mapping calls, while the A-back/B-back comparison has {aback['observed_repeat_stable_different']}/21 observed repeat-stable difference; its other three context contrasts have none. "
        "These asymmetries support reporting each intervention and service separately.\n\n"+
        "The timestamp audit also exposes a specific attribution limit: all three Sol B-back jobs have recorded client starts before any B-front job. "
        "Same-numbered repeats therefore do not provide time-matched position contrasts. The observed stable-difference category does not repair that schedule. "
        "This finding motivated the separately frozen sequential follow-up reported below. Estimating an effect over service periods would require more time blocks and repeated controls across those periods; an order-balanced schedule alone cannot guarantee physical-time balance.\n")
    abstract=(f"A post hoc audit of all recorded call pairs finds {rotation['observed_repeat_stable_different']}/70 Sol cases with a consistent label within each of three clean and three rotated calls but different labels between conditions. "
              "None of its four fixed-size context contrasts has a case meeting that finite-repeat criterion; their interpretation remains limited by within-input variation and unmatched call timing.")
    hydro=[r for r in audit['contrasts'] if r['study']=='hydraulic_two_complete_blocks']
    htable=[r'\begin{table}[htbp]\centering\small',
            r'\caption{Separate hydraulic audit using only complete blocks R1 and R2: each side has two calls. D and V have the definitions in Table~\ref{tab:repeat_separation}; stability across two calls is not comparable to a three-call test. No third-block result replaces a missing condition.}',
            r'\label{tab:hydraulic_repeat_partition}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',
            r'Left / right input & $n$ & D & Same throughout & V \\\midrule']
    hnames={'clean__pair_reverse':'Clean / pair reversal','clean__descriptor_rotation':'Clean / descriptor rotation',
            'clean__clean_repeat':'Clean / explicit duplicate','a_front__a_front_repeat':'A-front / explicit duplicate'}
    for r in hydro:
        label=hnames.get(r['comparison'],labels.get(r['comparison'],r['comparison']))
        htable.append(f"{label} & {r['n_cases']} & {r['observed_repeat_stable_different']} & {r['observed_repeat_stable_same']} & {r['any_side_unstable']}"+r' \\')
    htable += [r'\bottomrule\end{tabular}\end{table}']
    supplement=(r'\section{Post hoc condition, repetition and timing audit}'+'\n'+methods+'\n'+results+'\n'+
                r'\subsection{Interrupted measured follow-up: separate two-block description}'+'\n'+
                'All eight specified hydraulic contrasts have zero observed repeat-stable differences in the two complete blocks. This does not establish no input effect: each contrast includes 12--18 cases varying within at least one side. Explicit duplicate conditions are not merged with across-block calls. The original 29 valid, two failed and 32 unattempted request counts remain unchanged.\n\n'+
                '\n'.join(htable)+'\n\n'+
                'The offline exports retain every within-condition pair, every cross-condition pair, all per-case prediction sequences and the unavailable category. They also retain client-side timestamps for all audited TEP calls. Request timestamps do not reveal server-side inference intervals. The analysis reads no answer labels and adds no model calls or source units.\n'+r'\FloatBarrier'+'\n')
    # The shared table appears once in the main manuscript and once in the supplement.
    return {'REPEAT_SEPARATION_METHODS':methods,'REPEAT_SEPARATION_RESULTS':results,
            'REPEAT_SEPARATION_ABSTRACT':abstract},supplement
