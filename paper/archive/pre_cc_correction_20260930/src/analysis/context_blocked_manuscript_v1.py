"""Non-frozen TeX renderer for an already frozen/scored context-blocked study.

Importing this module does not read study files. Call make_context_fragments()
only after the terminal-output freeze and scoring have finished. This renderer
never opens answer keys or individual running prediction artifacts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/context_blocked_v1'
CATEGORIES = ('observed_two_repeat_stable_different', 'observed_two_repeat_stable_same',
              'any_condition_unstable', 'incomplete_block')


def _digest(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or symlinked frozen/scored input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    _digest(path)
    return json.loads(path.read_text())


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _read_scored_study(base):
    """Read only completed summaries, protocol and output manifest, not answers."""
    protocol = _load(base / 'protocol.json')
    manifest = _load(base / 'predictions/frozen_manifest.json')
    scores = _load(base / 'analysis/scores.json')
    ph = _digest(base / 'protocol.json')
    fh = _digest(base / 'predictions/frozen_manifest.json')
    _need(ph == (base / 'protocol.sha256').read_text().strip(), 'Protocol SHA-256 mismatch')
    _need(manifest['protocol_sha256'] == ph == scores['protocol_sha256'], 'Score/freeze/protocol mismatch')
    _need(scores['frozen_manifest_sha256'] == fh, 'Scores refer to a different output freeze')
    _need(scores['answers_sha256'] == protocol['answers_sha256'], 'Scored answer reference changed')
    _need(protocol['calls_scheduled'] == 16 and protocol['slots_scheduled'] == 672,
          'This renderer requires the fixed 16-request, 672-slot design')
    _need(protocol['focal_cases'] == protocol['shared_background_cases'] == 21
          and protocol['new_source_units'] == 0, 'Unexpected source or panel size')
    _need(protocol['block_sequences'] == ['ABBA', 'ABBA', 'BAAB', 'BAAB'], 'Unexpected realized block order')
    totals = scores['totals']
    _need(totals['planned_sessions'] == 16 and totals['planned_prediction_slots'] == 672,
          'Scheduled denominators disagree')
    _need(totals['scheduled_focal_slots'] == 336, 'Expected 16 x 21 scheduled focal slots')
    for field in ('planned_sessions', 'attempted_sessions', 'complete_sessions',
                  'not_attempted_sessions', 'planned_prediction_slots', 'valid_prediction_slots'):
        _need(totals[field] == manifest[field], f'Freeze/score total mismatch: {field}')
    requests, focal = scores['request_metrics'], scores['focal_metrics']
    _need(len(requests) == len(focal) == 16, 'All 16 scheduled request rows must be retained')
    _need(len({r['group'] for r in requests}) == len({r['group'] for r in focal}) == 16,
          'Duplicate request rows')
    _need(sum(r['valid_cases'] for r in requests) == totals['valid_prediction_slots'], 'Valid full-slot mismatch')
    _need(sum(r['valid_cases'] for r in focal) == totals['valid_focal_slots'], 'Valid focal-slot mismatch')
    _need(len(scores['block_summaries']) == 4 and len(scores['paired']) == 24,
          'All four blocks and six planned pairs per block must be retained')
    for block in scores['block_summaries']:
        _need(sum(block[k] for k in CATEGORIES) == 21, 'Block stability categories must partition 21 focal IDs')
        _need(block['scheduled_requests'] == 4, 'Block request denominator changed')
    return protocol, manifest, scores


def _fraction(row):
    return 'N/A' if row['correct'] is None else f"{row['correct']}/21"


def _flip(row):
    return 'N/A' if row['flips'] is None else f"{row['flips']}/21"


def _range(values, *, latex=True):
    if not values:
        return 'N/A'
    a, b = min(values), max(values)
    return str(a) if a == b else f"{a}{'--' if latex else '–'}{b}"


def _number(value):
    return 'N/A' if value is None else f'{value:.1f}'


def _result_tables(scores):
    outcomes = [r'\begin{table}[htbp]\centering\small',
                r'\caption{Same 21 focal TEP records with the original background panel B. F denotes focal-first (sequence A), R focal-last (sequence B). F1/F2 and R1/R2 show correct/21 for chronological calls within each block. Other cells show label changes/21; cross-order cells give the observed range and available pairs out of four. N/A means unavailable.}',
                r'\label{tab:context_blocked_outcomes}',
                r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
                r'Block & Order & F1; F2 & R1; R2 & Within F & Within R & Cross F/R \\\midrule']
    stable = [r'\begin{table}[htbp]\centering\small',
              r'\caption{Finite two-call categories within each scheduled block. ``Stable different'' requires both A calls to agree, both B calls to agree, and their labels to differ; ``stable same'' also requires agreement across conditions. ``Unstable'' means at least one condition changes. Any missing call leaves the block incomplete for this classification. Each row partitions the same 21 reused cases; counts are not independent-source additions or causal lower bounds.}',
              r'\label{tab:context_blocked_stability}',
              r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
              r'Block & Valid/4 & Stable different & Stable same & Unstable & Incomplete \\\midrule']
    for block in scores['block_summaries']:
        number = block['block']
        calls = sorted([r for r in scores['focal_metrics'] if r['block'] == number], key=lambda r: r['position'])
        a = [r for r in calls if r['condition'] == 'A']
        b = [r for r in calls if r['condition'] == 'B']
        _need(len(a) == len(b) == 2, 'Missing planned A or B call')
        pairs = [r for r in scores['paired'] if r['block'] == number]
        within_a = next(r for r in pairs if r['comparison'] == 'within_A')
        within_b = next(r for r in pairs if r['comparison'] == 'within_B')
        cross = [r['flips'] for r in pairs if r['comparison'] == 'cross_A_B' and r['flips'] is not None]
        cross_text = 'N/A (0/4)' if not cross else f"{_range(cross)}/21 ({len(cross)}/4)"
        outcomes.append(f"{number} & {block['sequence']} & {'; '.join(_fraction(r) for r in a)} & "
                        f"{'; '.join(_fraction(r) for r in b)} & {_flip(within_a)} & {_flip(within_b)} & {cross_text}" + r' \\')
        stable.append(f"{number} & {block['valid_requests']}/4 & {block[CATEGORIES[0]]} & {block[CATEGORIES[1]]} & "
                      f"{block[CATEGORIES[2]]} & {block[CATEGORIES[3]]}" + r' \\')
    outcomes += [r'\bottomrule\end{tabular}\end{table}']
    stable += [r'\bottomrule\end{tabular}\end{table}']
    return '\n'.join(outcomes), '\n'.join(stable)


def _timing_table(scores):
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\caption{Client-side timing for the blocked follow-up. A/B are schedule symbols for focal-first/focal-last with the same original background panel B. Differences are mean A minus mean B, in seconds, reported only when both calls per order were attempted. Starts are wrapper dispatch timestamps and ends are metadata-update timestamps. Midpoints summarize client envelopes, not server inference times. Request-level timestamps and durations remain in the CSV exports.}',
             r'\label{tab:context_blocked_timing}',
             r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
             r'Block & Order & A tried & B tried & Start diff. & End diff. & Midpoint diff. \\\midrule']
    for block in scores['block_summaries']:
        t = block['timing']
        if t is None:
            values = ['0', '0', 'N/A', 'N/A', 'N/A']
        else:
            values = [str(t['A']['attempted_calls']), str(t['B']['attempted_calls'])]
            values += [_number(t[f'a_minus_b_mean_{measure}_seconds']) for measure in ('start', 'end', 'midpoint')]
        lines.append(f"{block['block']} & {block['sequence']} & " + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    return '\n'.join(lines)


def make_context_fragments(base=BASE):
    """Return ({METHODS, RESULTS, ABSTRACT markers}, Appendix E TeX string).

    Invoke only after freeze and scoring. All outcomes come from the separately
    versioned score file; the original simulation results remain untouched.
    """
    base = Path(base)
    protocol, manifest, scores = _read_scored_study(base)
    totals = scores['totals']
    complete, attempted, unattempted = (totals[k] for k in ('complete_sessions', 'attempted_sessions', 'not_attempted_sessions'))
    failed = attempted - complete
    _need(complete + failed + unattempted == 16, 'Request outcome partition mismatch')
    counts = Counter(r['status'] for r in scores['request_metrics'])
    _need(counts['complete'] == complete and counts['not_attempted'] == unattempted, 'Per-request availability mismatch')
    within = [r['flips'] for r in scores['paired'] if r['comparison'] in ('within_A', 'within_B') and r['flips'] is not None]
    cross = [r['flips'] for r in scores['paired'] if r['comparison'] == 'cross_A_B' and r['flips'] is not None]
    outcome_table, stability_table = _result_tables(scores)
    stable_counts = ', '.join(str(r[CATEGORIES[0]]) if r['incomplete_block'] == 0 else 'N/A'
                              for r in sorted(scores['block_summaries'], key=lambda r: r['block']))
    methods = (r'\subsection{Prospective ordering control on reused TEP contexts}' + '\n' +
        "After inspecting earlier results and request timing, we specified a separate exploratory replication using exact frozen Sol b-front/b-back payloads. Both retain the original background panel B. Sequence A places 21 focal records before the same 21 background records (F); sequence B places them after that background (R). These symbols denote order, not different background panels. Values, descriptors and task instructions remain unchanged; no new sources are added.\n\n"
        "Four consecutive blocks contain ABBA twice and BAAB twice. One seeded shuffle (20260928) fixes the realized order as ABBA, ABBA, BAAB, BAAB. Sixteen fresh, sequential subscription-authenticated CLI calls use Sol with medium reasoning and the existing isolated interface. The plan contains 672 full-record slots and 336 focal slots. Failure or resource stops preserve unattempted requests without replacements; terminal outputs are frozen before scoring.\n\n"
        "Within each block, we retain one F/F pair, one R/R pair, all four F/R pairs and a case-level partition into observed two-call stable-different, stable-same, unstable or incomplete outcomes. Positional balance does not guarantee equal elapsed time or eliminate service drift. Appendix E records timeouts, resource decisions, complete per-block outcomes and client timing.")
    results_intro = (
        f"Of 16 planned requests, {attempted} were attempted: {complete} returned valid outputs and {failed} failed execution or validation; {unattempted} remained unattempted. "
        f"Coverage was {totals['valid_prediction_slots']}/672 full-record and {totals['valid_focal_slots']}/336 focal slots; {totals['complete_blocks']} of four blocks had all outputs available. "
        r'Table~\ref{tab:context_blocked_outcomes} reports both calls under each order and available within- and cross-order disagreements. ' +
        "The 21 focal cases are reused throughout, so request and slot counts do not enlarge the source sample.")
    range_sentence = (f"Within-order pairs changed {_range(within)} labels and cross-order pairs changed {_range(cross)} labels per 21-case comparison. "
                      if within and cross else "The incomplete outputs do not provide both within-order and cross-order comparisons. ")
    results_followup = (range_sentence +
        f"Observed stable-different counts for blocks 1--4 were {stable_counts}, respectively; N/A denotes an incomplete block. "
        "This category requires both calls in each condition to agree while the two condition labels differ. Incomplete blocks are not filled using other calls. "
        "These descriptive results assess the new schedule without replacing earlier responses or identifying a net causal effect. Full category counts, unavailable comparisons and client-time offsets appear in Appendix E.")
    results = (r'\subsection{Replication with balanced call positions}' + '\n' + results_intro + '\n\n' +
               outcome_table + '\n\n' + results_followup)
    if within and cross:
        abstract = (f"In a separate result-informed ABBA/BAAB replication, {complete}/16 requests were valid; "
                    f"21 reused cases showed {_range(within)} within-condition and {_range(cross)} cross-condition label changes per pair, "
                    "without establishing a causal context effect.")
    else:
        abstract = (f"A separate, result-informed ABBA/BAAB replication on the same 21 cases returned {complete}/16 valid requests; "
                    "incomplete paired outputs preclude a blockwise comparison.")
    _need(len(abstract.split()) <= 40, 'Abstract addition exceeds the requested 40-word bound')
    estimates = protocol['budget_reference']
    appendix_followup = results_followup.replace(
        'Full category counts, unavailable comparisons and client-time offsets appear in Appendix E.',
        'The tables and exports retain every scheduled block and unavailable comparison.')
    appendix = (r'\section{Separately frozen ABBA/BAAB context replication}' + '\n' +
        r'\label{app:context_blocked}' + '\n' +
        "This Appendix E concerns a separate prospective call-order protocol selected after the earlier results and retrospective timing audit. "
        "It is an exploratory replication on already evaluated cases, not original preregistration, new-source confirmation or a replacement for the earlier outputs. "
        "Both payloads retain the original background panel B: sequence A is focal-first (F), and sequence B is focal-last (R). The A/B symbols therefore denote order, not different background identities. "
        "The two payloads exactly reuse the historical b-front/b-back requests, and the isolated CLI helper retains the same developer instructions. "
        "The fixed sequence is ABBA, ABBA, BAAB, BAAB after one shuffle with seed 20260928. All calls are sequential, use Sol with medium reasoning, and share the isolated, tool-disabled subscription CLI interface. "
        "Within-group record order, descriptions and numerical strings remain unchanged. Any failed or invalid request, unknown usage or resource stop prevents new dispatch; none is repaired or replaced.\n\n" +
        r'\subsection{Complete blockwise outcomes}' + '\n' + results_intro + '\n\n' +
        outcome_table + '\n\n' + stability_table + '\n\n' + appendix_followup + '\n\n' +
        r'\subsection{Resource decision before freezing and execution accounting}' + '\n' +
        "Before protocol freezing or new inference, the initially discussed 400,000-token dispatch threshold was revised to 1,200,000 after examining the old same-payload usage logs. "
        f"Those logs implied approximately {round(estimates['estimated_16_call_observed_tokens']):,} reported input-plus-output tokens for the unchanged 16-call plan. "
        "This revision added no calls, cases or conditions and was not based on new outcomes. Cached tokens are already part of reported input tokens and are not subtracted. "
        "The limit stops future dispatch; it is not a hard per-request generation cap. The total wall-time limit is 1,800 seconds, and a call is dispatched only if its full 300-second timeout fits within the remaining allowance.\n\n"
        f"Available usage records report {totals['observed_total_tokens']:,} input-plus-output tokens and {totals['unknown_usage_sessions']} attempted requests with unknown usage. "
        "Unknown usage is not treated as zero, so observed totals need not be a complete resource total. "
        f"Availability remains {complete}/{attempted} among attempted requests and {complete}/16 over the fixed plan; {unattempted} unattempted requests are not model-generated formatting failures. "
        f"All scheduled denominators remain 672 full-record and 336 focal slots. Failed partial arrays supply no classifications.\n\n" +
        r'\subsection{Client timing and interpretation}' + '\n' + _timing_table(scores) + '\n\n' +
        "An ABBA or BAAB block places each condition at the same average ordinal position, but asymmetric request duration can shift its average client time. "
        "The displayed offsets therefore accompany, rather than validate away, temporal confounding. Starts are client dispatch timestamps; ends are metadata updates, and their midpoint is a request-envelope summary. "
        "They do not locate server-side inference or reveal a model snapshot change.\n\n"
        "Four A/B pairings per block reuse the same four calls, and every block reuses the same 21 focal records. Neither pair counts nor 336 repeated focal classifications enlarge the source sample. "
        "Stable-different, stable-same, unstable and incomplete counts are descriptive partitions of these focal IDs within a block. "
        "Incomplete blocks are never completed with a call from another block; no p-values, noise-corrected effects or causal lower bounds are computed. "
        "The protocol, input hashes, timestamps, frozen raw outputs, request-level accuracy, individual paired labels and timing summaries remain in the separate context-blocked reproduction package.")
    # Avoid a spurious fraction if transport prevented every planned attempt.
    if attempted == 0:
        appendix = appendix.replace(f'Availability remains {complete}/{attempted} among attempted requests',
                                    'Availability among attempted requests is undefined because no request was attempted')
    return {'CONTEXT_BLOCKED_METHODS': methods, 'CONTEXT_BLOCKED_RESULTS': results,
            'CONTEXT_BLOCKED_ABSTRACT': abstract}, appendix
