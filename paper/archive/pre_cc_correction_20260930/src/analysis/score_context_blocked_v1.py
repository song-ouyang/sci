"""Offline scoring of the separately frozen ABBA/BAAB exploratory replication."""
from collections import Counter
from datetime import datetime, timezone
import csv
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.context_blocked_v1 import BASE, audit_outputs, digest, read, summary, token_count


def metric(ids, predictions, truth):
    if predictions is None:
        return {'scheduled_cases': len(ids), 'valid_cases': 0, 'correct': None, 'accuracy': None, 'usable_correct': 0}
    correct = sum(predictions[i] == truth[i]['label'] for i in ids)
    return {'scheduled_cases': len(ids), 'valid_cases': len(ids), 'correct': correct,
            'accuracy': correct / len(ids), 'usable_correct': correct}


def pair(name, left_name, right_name, ids, predictions):
    left, right = predictions[left_name], predictions[right_name]
    if left is None or right is None:
        return {'comparison': name, 'left': left_name, 'right': right_name, 'scheduled_cases': len(ids),
                'paired_cases': 0, 'flips': None, 'flip_rate': None}
    flips = sum(left[i] != right[i] for i in ids)
    return {'comparison': name, 'left': left_name, 'right': right_name, 'scheduled_cases': len(ids),
            'paired_cases': len(ids), 'flips': flips, 'flip_rate': flips / len(ids)}


def classify_id(case_id, a_names, b_names, predictions):
    a = [None if predictions[name] is None else predictions[name][case_id] for name in a_names]
    b = [None if predictions[name] is None else predictions[name][case_id] for name in b_names]
    if None in a or None in b:
        category = 'incomplete_block'
    elif len(set(a)) > 1 or len(set(b)) > 1:
        category = 'any_condition_unstable'
    elif a[0] != b[0]:
        category = 'observed_two_repeat_stable_different'
    else:
        category = 'observed_two_repeat_stable_same'
    return {'id': case_id, 'a_predictions': a, 'b_predictions': b, 'category': category}


def timing(rows, origin):
    offsets = {}
    for condition in ['A', 'B']:
        condition_rows = [r for r in rows if r['condition'] == condition and r['status'] != 'not_attempted']
        starts = [(datetime.fromisoformat(r['started_at']) - origin).total_seconds() for r in condition_rows]
        ends = [(datetime.fromisoformat(r['completed_at']) - origin).total_seconds() for r in condition_rows]
        offsets[condition] = {'attempted_calls': len(condition_rows),
                              'mean_start_seconds': statistics.mean(starts) if starts else None,
                              'mean_end_seconds': statistics.mean(ends) if ends else None,
                              'mean_midpoint_seconds': statistics.mean([(a + b) / 2 for a, b in zip(starts, ends)]) if starts else None}
    complete_schedule = all(offsets[c]['attempted_calls'] == 2 for c in ['A', 'B'])
    for measure in ['start', 'end', 'midpoint']:
        offsets[f'a_minus_b_mean_{measure}_seconds'] = (offsets['A'][f'mean_{measure}_seconds'] - offsets['B'][f'mean_{measure}_seconds']) if complete_schedule else None
    offsets['offsets_comparable_over_two_calls_per_condition'] = complete_schedule
    return offsets


def score(base=BASE):
    base = Path(base)
    # Original terminal artifacts and every freeze hash are checked before
    # opening this study's answer key. No scorer repairs any model response.
    protocol, groups, audited, _ = audit_outputs(base, require_frozen=True)
    if digest(base / 'answers/records.json') != protocol['answers_sha256']:
        raise ValueError('Answer key hash changed')
    truth = {r['id']: r for r in read(base / 'answers/records.json')}
    panels = read(base / 'inference/panels.json')
    focal = panels['focal']
    schedule = read(base / 'schedule.json')['blocks']
    indexed = {r['group']: r for r in audited}
    predictions = {r['group']: None if r['predictions'] is None else {x['id']: x['predicted'] for x in r['predictions']} for r in audited}
    request_rows, focal_rows, pairs, case_rows, block_rows = [], [], [], [], []
    attempted = [r for r in audited if r['status'] != 'not_attempted']
    origin = min((datetime.fromisoformat(r['started_at']) for r in attempted), default=None)
    for block_index, names in enumerate(schedule, 1):
        rows = [indexed[name] for name in names]
        a_names = [n for n in names if groups[n]['condition'] == 'A']
        b_names = [n for n in names if groups[n]['condition'] == 'B']
        for position, name in enumerate(names, 1):
            r = indexed[name]
            common = {'block': block_index, 'position': position, 'group': name,
                      'condition': groups[name]['condition'], 'status': r['status'],
                      'started_at': r.get('started_at'), 'completed_at': r.get('completed_at'),
                      'elapsed_seconds': r.get('elapsed_seconds'), 'observed_tokens': token_count(r['usage'])}
            ids = [x['id'] for x in groups[name]['samples']]
            request_rows.append(dict(common, **metric(ids, predictions[name], truth)))
            focal_rows.append(dict(common, **metric(focal, predictions[name], truth)))
        pairs.append(dict(block=block_index, **pair('within_A', *a_names, focal, predictions)))
        pairs.append(dict(block=block_index, **pair('within_B', *b_names, focal, predictions)))
        pairs.extend(dict(block=block_index, **pair('cross_A_B', a, b, focal, predictions)) for a, b in itertools.product(a_names, b_names))
        cases = [dict(block=block_index, **classify_id(i, a_names, b_names, predictions)) for i in focal]
        case_rows.extend(cases)
        counts = Counter(r['category'] for r in cases)
        block_rows.append({'block': block_index, 'sequence': protocol['block_sequences'][block_index - 1],
                           'valid_requests': sum(r['status'] == 'complete' for r in rows), 'scheduled_requests': 4,
                           'focal_cases': 21, **{k: counts[k] for k in ['observed_two_repeat_stable_different',
                           'observed_two_repeat_stable_same', 'any_condition_unstable', 'incomplete_block']},
                           'timing': timing(rows, origin) if origin is not None else None})
    totals = summary(audited)
    totals.update(usable_correct_slots=sum(r['usable_correct'] for r in request_rows),
                  valid_focal_slots=sum(r['valid_cases'] for r in focal_rows),
                  scheduled_focal_slots=sum(r['scheduled_cases'] for r in focal_rows),
                  complete_blocks=sum(r['valid_requests'] == 4 for r in block_rows))
    result = {'schema': 'context-blocked-v1-scores', 'scored_at': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': digest(base / 'protocol.json'),
              'frozen_manifest_sha256': digest(base / 'predictions/frozen_manifest.json'),
              'answers_sha256': protocol['answers_sha256'], 'totals': totals, 'request_metrics': request_rows,
              'focal_metrics': focal_rows, 'block_summaries': block_rows, 'paired': pairs, 'case_stability': case_rows,
              'limits': ['Exploratory replication selected after known outcomes; no new source units.',
                         'Ordinal-position balance does not equal clock-time balance or eliminate service drift.',
                         'Starts are client wrapper dispatch timestamps; ends are client metadata updates. Midpoints summarize this client-side request envelope, not server inference timestamps.',
                         'Within-block pairs reuse the same 21 focal cases and calls; no independent-pair inference.',
                         'Two-repeat stable differences are finite observations, not causal lower bounds; zero does not prove no effect.',
                         'Failed and unattempted requests retain full planned denominators; incomplete blocks are not filled from another block.']}
    output = base / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'scores.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, rows in [('requests', request_rows), ('focal_metrics', focal_rows), ('paired', pairs), ('case_stability', case_rows), ('block_summaries', block_rows)]:
        with (output / f'{filename}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    lines = ['# ABBA/BAAB exploratory context replication', '', json.dumps(totals, indent=2), '',
             '| Block | Sequence | Valid/4 | Stable different | Stable same | Any unstable | Incomplete |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in block_rows:
        lines.append(f"| {r['block']} | {r['sequence']} | {r['valid_requests']}/4 | {r['observed_two_repeat_stable_different']} | {r['observed_two_repeat_stable_same']} | {r['any_condition_unstable']} | {r['incomplete_block']} |")
    lines += ['', *result['limits'], '']
    (output / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps(totals))
    return result


if __name__ == '__main__':
    score()
