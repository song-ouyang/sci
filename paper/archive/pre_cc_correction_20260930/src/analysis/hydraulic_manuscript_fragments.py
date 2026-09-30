"""Render manuscript fragments from audited scores, never from model calls."""
import json
import statistics
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/hydraulic_validation_v1'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_fragments():
    s=json.loads((BASE/'analysis/scores.json').read_text())
    p=json.loads((BASE/'protocol.json').read_text())
    if s['protocol_sha256']!=digest(BASE/'protocol.json') or s['frozen_manifest_sha256']!=digest(BASE/'predictions/frozen_manifest.json'):
        raise ValueError('Measured score provenance mismatch')
    t=s['totals']
    labels={'clean':'Fixed focal','clean_repeat':'Fixed focal duplicate','pair_reverse':'Intact-pair reversal',
            'descriptor_rotation':'Descriptor rotation','a_front':'Focal before A','a_front_repeat':'Focal before A duplicate',
            'a_back':'Focal after A','b_front':'Focal before B','b_back':'Focal after B'}
    def row(condition,repeat=1):
        return next(x for x in s['focal_metrics'] if x['condition']==condition and x['repeat']==repeat)
    def fraction(r):
        return 'N/A' if r['correct'] is None else f"{r['correct']}/{r['n']}"
    def span(name):
        values=[x['flips'] for x in s['paired'] if x['comparison']==name and x['flips'] is not None]
        if not values:return 'N/A'
        return str(min(values)) if min(values)==max(values) else f'{min(values)}--{max(values)}'
    table=[r'\begin{table}[htbp]\centering\small',
           r'\caption{Measured hydraulic follow-up: correct labels among the same 18 focal cycles. R1--R5 are scheduled execution blocks, not independent devices. All planned repetitions are shown; N/A denotes an invalid or unattempted request.}',
           r'\label{tab:hydraulic}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
           r'Input condition & R1 & R2 & R3 & R4 & R5 \\\midrule']
    for c,label in labels.items():
        table.append(label+' & '+' & '.join(fraction(row(c,r)) for r in range(1,6))+r' \\')
    table += [r'\bottomrule\end{tabular}\end{table}']
    table='\n'.join(table)
    singleton=s['singleton']
    tokens=t['tokens_observed']
    elapsed=[x['elapsed_seconds'] for x in s['resource_by_request'] if x['elapsed_seconds'] is not None]
    timing='No complete duration data were available.' if not elapsed else f"Observed per-request wall times range from {min(elapsed):.1f} to {max(elapsed):.1f} seconds (median {statistics.median(elapsed):.1f}); parallel durations are not summed as total elapsed study time."
    text=(r'\subsection{Measured input-policy comparison on one hydraulic rig}'+'\n'+
          f"The measured schedule yields {t['requests_complete']}/{t['requests_scheduled']} valid requests and "
          f"{t['slots_valid']:,}/{t['slots_scheduled']:,} valid classification slots, separately from the four simulation stages. "
          "Two requests time out after network disconnections; the stopping rule leaves 32 unattempted. "
          "Only R1 and R2 are complete; R3 is partial and R4--R5 are unattempted. "
          "Each failed job has one invocation, within which native transport reconnections are recorded. "
          f"Primary fixed-focal, intact-reorder and descriptor-rotation results are {fraction(row('clean'))}, "
          f"{fraction(row('pair_reverse'))} and {fraction(row('descriptor_rotation'))}, respectively "
          r'(Table~\ref{tab:hydraulic}). '+'\n\n'+table+'\n\n'+
          "In R1 and R2, byte-identical focal duplicates change 17/18 and 0/18 labels, respectively, "
          f"and byte-identical 36-record A-panel duplicates change {span('duplicate_panel_a')}. "
          f"The same-size position contrasts change {span('position_a')} labels with background A and {span('position_b')} with B; "
          f"background replacement changes {span('background_front')} with focal records first and {span('background_back')} with them last. "
          "Position-B comparisons have three valid pairs; the other contrasts have two. "
          "These counts retain service variation rather than removing it by subtraction. "
          "The R2 descriptor-rotation request classifies all 18 focal cases correctly despite the faulty mapping; this does not demonstrate correct use of descriptions. "
          f"Singletons supply {singleton['valid']}/18 valid decisions with {singleton['correct']} correct; their calls occur at different times and each case is evaluated only once. "
          "The available singleton subset has one no-leakage, five weak-leakage and three severe-leakage cases, so its accuracy is not directly comparable with the complete balanced focal panel.\n\n"+
          "On all 54 measured records, the gateway accepts clean input and intact-pair reversal, yielding identical canonical records. "
          "It rejects all 54 descriptor rotations and all 54 injected same-unit descriptor swaps, with no clean-input rejection. "
          "All 54 internally consistent hidden value swaps remain accepted. These finite injection checks require trustworthy acquisition IDs; "
          "they neither detect every mapping failure nor classify an equipment fault. Rejected corrupt inputs have zero automated diagnostic coverage, "
          "so error prevention is accompanied by explicit withholding of a decision.\n\n"+
          f"Known usage totals {tokens:,} input-plus-output tokens, a lower bound because "
          f"{t['usage_unknown_requests']} attempted requests lack complete token usage. "+timing+
          " These are resource measurements under a subscription-authenticated interface; no monetary price or zero-compute-cost claim is inferred. "
          "The new data were not used to choose a best-performing policy, and all planned conditions are retained. "
          "The interrupted schedule does not complete the planned five-block policy comparison and supports no validated mitigation gain. "
          "It assesses specified summaries and one service on a single rig, without a trained domain classifier comparator or evidence of cross-device generalization.")
    abstract=("A separately frozen single-rig follow-up stops after network failures, with 29 of 63 planned requests valid. "
              "An identical-input pair changes 17 of 18 focal labels; the incomplete study establishes no mitigation benefit.")
    # Keep supplementary F1 and resource detail separate from accuracy counts.
    f1table=[r'\begin{table}[htbp]\centering\small',r'\caption{Hydraulic focal macro F1 by execution block; all three task labels enter each macro average. Singleton macro F1 is reported only after pooling its available one-case calls.}',
             r'\label{tab:hydraulic_f1}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'Input condition & R1 & R2 & R3 & R4 & R5 \\\midrule']
    for c,label in labels.items():
        values=[row(c,r)['macro_f1'] for r in range(1,6)]
        f1table.append(label+' & '+' & '.join('N/A' if v is None else f'{v:.3f}' for v in values)+r' \\')
    f1table += [r'\bottomrule\end{tabular}\end{table}']
    resource=[r'\begin{table}[htbp]\centering\small',r'\caption{Hydraulic resource and availability accounting by input condition. Tokens sum reported input and output tokens; unavailable usage is not zero. Latency is per attempted request; no monetary conversion is made.}',
              r'\label{tab:hydraulic_resources}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r'Condition & Valid/planned & Tokens & Unknown & Median s \\\midrule']
    for c,label in {**labels,'singleton':'Singletons (one per case)'}.items():
        rows=[x for x in s['resource_by_request'] if x['condition']==c]
        known=[x['usage']['input_tokens']+x['usage']['output_tokens'] for x in rows if isinstance(x['usage'],dict) and all(type(x['usage'].get(k)) is int for k in ['input_tokens','output_tokens'])]
        durations=[x['elapsed_seconds'] for x in rows if x['elapsed_seconds'] is not None]
        valid=sum(x['status']=='complete' for x in rows)
        unknown=sum(x['status']!='not_attempted' for x in rows)-len(known)
        latency='N/A' if not durations else f'{statistics.median(durations):.1f}'
        resource.append(f'{label} & {valid}/{len(rows)} & {sum(known):,} & {unknown} & {latency}'+r' \\')
    resource += [r'\bottomrule\end{tabular}\end{table}']
    s_f1='N/A' if singleton['macro_f1'] is None else f"{singleton['macro_f1']:.3f}"
    supp=(r'\section{Measured follow-up and repeated-input reanalysis}'+'\n'+
          r'This appendix was added after the archived four-stage report above. Its hydraulic results are a separate exploratory study; they do not retrospectively amend the original protocols or turn measured cycles into independent devices. The focused manuscript provides the new sampling and execution methods. The local package retains every condition, timestamp, response, failure and per-case score.'+'\n\n'+
          text.replace(r'\subsection{Measured input-policy comparison on one hydraulic rig}',r'\subsection{Hydraulic outcomes and input contract}')+'\n\n'+
          '\n'.join(f1table)+'\n\n'+f"The aggregate singleton macro F1 is {s_f1}, conditional on {singleton['valid']} valid outputs. Its denominator remains 18 planned calls for availability. "+
          "The supplementary CSV exports include focal metrics stratified by cooler state; each such stratum has only six focal cycles. These are descriptive subgroup records, not independent replication.\n\n"+
          '\n'.join(resource)+'\n\n')
    return {'HYDRAULIC_RESULTS':text,'HYDRAULIC_ABSTRACT':abstract},supp
