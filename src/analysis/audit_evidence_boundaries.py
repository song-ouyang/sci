"""Independently recompute finite-repeat evidence from existing frozen responses.

No inference, downloads, refits or changes to frozen inputs. This standard-library
implementation does not import the original runners, scorers or repeat audit.
Answer labels are opened only after the frozen requests/responses pass integrity
checks. It is a computational verification, not independent human peer review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['Normal', 'Fault_1', 'Fault_2', 'Fault_4', 'Fault_6', 'Fault_7', 'Fault_13']
MAPS = ['rotation', 'random_1', 'random_2', 'random_3', 'same_unit_swap', 'pair_shuffle']
CONTEXT = [('a_front', 'a_back'), ('b_front', 'b_back'), ('a_front', 'b_front'), ('a_back', 'b_back')]
CATEGORIES = ['stable_different', 'stable_same', 'any_side_unstable', 'incomplete']


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def digest(self, path):
        require(path.is_file() and not path.is_symlink(), f'Missing/symlink input: {path}')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        value = h.hexdigest()
        self.hashes[str(path.relative_to(ROOT))] = value
        return value

    def read(self, path):
        self.digest(path)
        return json.loads(path.read_text())

    def check(self, path, expected):
        require(self.digest(path) == expected, f'Hash mismatch: {path}')


def safe_path(base, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, f'Unsafe path: {relative}')
    return base / p


def load_frozen(inputs, study):
    base = ROOT / 'results' / study
    p = inputs.read(base / 'protocol.json')
    ph = inputs.digest(base / 'protocol.json')
    inputs.digest(base / 'protocol.sha256')
    require((base / 'protocol.sha256').read_text().strip() == ph, 'Protocol SHA mismatch')
    manifest = inputs.read(base / 'predictions/frozen_manifest.json')
    require(manifest['protocol_sha256'] == ph, 'Manifest/protocol mismatch')
    present = {str(f.relative_to(base / 'predictions')) for f in (base / 'predictions').rglob('*')
               if f.is_file() and f.name != 'frozen_manifest.json'}
    require(present == set(manifest['files']), 'Frozen file inventory changed')
    for relative, expected in manifest['files'].items():
        inputs.check(safe_path(base / 'predictions', relative), expected)
    for relative, expected in p['code_sha256'].items():
        inputs.check(safe_path(ROOT, relative), expected)
    inputs.check(base / 'inference/batches.json', p['input_sha256'])
    groups = inputs.read(base / 'inference/batches.json')['groups']
    prompt_hashes = p.get('prompt_sha256', p.get('prompts_sha256'))
    require(set(prompt_hashes) == set(groups), 'Prompt inventory differs')
    for name, expected in prompt_hashes.items():
        inputs.check(base / 'inference/prompts' / (name + '.txt'), expected)
        group = groups[name]
        payload = (base / 'inference/prompts' / (name + '.txt')).read_text()
        require(json.loads(payload.split('\n\n', 1)[1]) == {
            'valid_labels': group['valid_labels'], 'samples': group['samples']}, 'Prompt/group mismatch')
        require(all(set(s) == {'id', 'prompt'} for s in group['samples']), 'Private fields in request')
    schedule_name = 'session_schedule.json' if study == 'independent_validation' else 'schedule.json'
    inputs.check(base / schedule_name, p['schedule_sha256'])
    schedule = inputs.read(base / schedule_name)
    jobs = schedule if isinstance(schedule, list) else [dict(model=p['model'], group=n)
               for block in schedule['blocks'] for n in block]
    require(len(jobs) == len({(j['model'], j['group']) for j in jobs}), 'Duplicate scheduled request')
    predictions, requests, execution = {}, {}, []
    session_ids = set()
    for job in jobs:
        model, name = job['model'], job['group']
        g = groups[name]
        d = base / 'predictions' / model
        meta = inputs.read(d / (name + '_metadata.json'))
        require(meta['batch'] == name and meta['model_requested'] == model, 'Request identity mismatch')
        require(meta['protocol_sha256'] == ph and meta['input_sha256'] == p['input_sha256'], 'Metadata provenance mismatch')
        require(meta['condition'] == g['condition'] and meta['repeat'] == g['repeat'], 'Condition/repeat mismatch')
        require(meta['tools_observed'] == 0 and meta['answer_keys_read'] is False, 'Recorded inference isolation failure')
        require(meta['status'] == 'complete', 'This audit requires the complete frozen TEP studies; missing outputs must not be replaced')
        req_path = d / (name + '_request.json')
        req = inputs.read(req_path)
        require(req['model'] == model and req['reasoning_effort'] == p['reasoning_effort'], 'Request settings mismatch')
        require(req['user_message'] == (base / 'inference/prompts' / (name + '.txt')).read_text(), 'Request/prompt mismatch')
        require(len(meta['attempts']) == 1 and meta['attempts'][0]['attempt'] == 1, 'Unexpected retries')
        attempt = meta['attempts'][0]
        require(attempt['returncode'] == 0 and not attempt['timed_out'], 'Incomplete execution')
        raw_path = d / attempt['response_file']
        raw = inputs.read(raw_path)  # All frozen responses in these two studies are strict JSON.
        results_path = d / (name + '_results.json')
        inputs.check(results_path, meta['prediction_sha256'])
        parsed = inputs.read(results_path)
        expected_ids = [r['id'] for r in g['samples']]
        require(len(set(expected_ids)) == len(expected_ids), 'Duplicate input ID')
        require(len(raw) == len(expected_ids) == len(parsed), 'Incomplete response')
        require(all(set(r) == {'id', 'predicted'} for r in raw), 'Unexpected raw response fields')
        require(len({r['id'] for r in raw}) == len(raw), 'Duplicate returned ID')
        raw_by_id = {r['id']: r['predicted'] for r in raw}
        require(set(raw_by_id) == set(expected_ids), 'Response ID coverage mismatch')
        require(set(raw_by_id.values()) <= set(g['valid_labels']), 'Unpermitted label')
        require([r['id'] for r in parsed] == expected_ids, 'Parsed output order changed')
        require(raw_by_id == {r['id']: r['predicted'] for r in parsed}, 'Raw/parsed disagreement')
        events_path = d / attempt['events_file']
        inputs.digest(events_path)
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        starts = [e['thread_id'] for e in events if e['type'] == 'thread.started']
        finishes = [e for e in events if e['type'] == 'turn.completed']
        messages = [e['item']['text'] for e in events if e['type'] == 'item.completed' and e['item']['type'] == 'agent_message']
        require(len(starts) == len(finishes) == len(messages) == 1, 'Unexpected session lifecycle')
        require(starts[0] not in session_ids, 'Session reused')
        session_ids.add(starts[0])
        require(events[-1]['type'] == 'turn.completed' and json.loads(messages[0]) == raw, 'Raw/event disagreement')
        require(finishes[0]['usage'] == attempt['usage'], 'Usage provenance mismatch')
        predictions[model, name] = raw_by_id
        requests[model, name] = inputs.digest(req_path)
        execution.append({'model': model, 'group': name, 'n': len(raw), 'status': meta['status'],
                          'raw_response_path': str(raw_path.relative_to(ROOT)),
                          'raw_response_sha256': inputs.digest(raw_path)})
    inputs.check(base / 'answers/records.json', p['answers_sha256'])
    answers = inputs.read(base / 'answers/records.json')
    require(len({r['id'] for r in answers}) == len(answers), 'Duplicate answer ID')
    return {'protocol': p, 'groups': groups, 'schedule': schedule, 'predictions': predictions,
            'requests': requests, 'truth': {r['id']: r for r in answers}, 'execution': execution,
            'manifest_files_verified': len(manifest['files']), 'base': base}


def classify(left, right):
    if None in left or None in right:
        return 'incomplete'
    if len(set(left)) > 1 or len(set(right)) > 1:
        return 'any_side_unstable'
    return 'stable_same' if left[0] == right[0] else 'stable_different'


def audit_transformations(d):
    """Check actual field operations without calling the preparation transform."""
    groups = d['groups']
    realized_orders = {c: set() for c in MAPS}
    counts = Counter()
    for repeat in [1, 2, 3]:
        clean = groups[f'clean_r{repeat}']['samples']
        clean_ids = [s['id'] for s in clean]
        for condition in MAPS:
            altered = groups[f'{condition}_r{repeat}']['samples']
            require([s['id'] for s in altered] == clean_ids, 'Mapping changed example identity/order')
            for left, right in zip(clean, altered):
                intro_a, body_a = left['prompt'].split('\n\n', 1)
                intro_b, body_b = right['prompt'].split('\n\n', 1)
                original = [line.rsplit(': ', 1) for line in body_a.splitlines()]
                changed = [line.rsplit(': ', 1) for line in body_b.splitlines()]
                require(intro_a == intro_b and len(original) == len(changed) == 52, 'Transformation changed task/field count')
                headers = [r[0] for r in original]
                require(len(set(headers)) == 52 and set(headers) == {r[0] for r in changed}, 'Header inventory changed')
                order = tuple(headers.index(r[0]) for r in changed)
                realized_orders[condition].add(order)
                if condition == 'pair_shuffle':
                    require(dict(original) == dict(changed), 'Paired shuffle broke physical mapping')
                else:
                    require([r[1] for r in original] == [r[1] for r in changed], 'Mismatch altered numeric strings/positions')
                    if condition == 'rotation':
                        require(order == tuple(list(range(1, 52)) + [0]), 'Rotation not the specified one-field cyclic shift')
                    elif condition.startswith('random_'):
                        require(all(i != j for i, j in enumerate(order)), 'Random mapping has fixed positions')
                    else:
                        changed_positions = [i for i, j in enumerate(order) if i != j]
                        require(changed_positions == [7, 11] and order[7] == 11 and order[11] == 7, 'Same-unit swap changed unexpected fields')
                        require(original[7][0].endswith('(%)') and original[11][0].endswith('(%)'), 'Swap units differ')
                counts[condition] += 1
        for a, b in CONTEXT:
            left = {s['id']: s['prompt'] for s in groups[f'{a}_r{repeat}']['samples']}
            right = {s['id']: s['prompt'] for s in groups[f'{b}_r{repeat}']['samples']}
            require(len(left) == len(right) == 42, 'Context request size changed')
            require(all(left[i] == right[i] for i in d['protocol']['focal_ids']), 'Context changed focal content')
    require(all(len(orders) == 1 for orders in realized_orders.values()), 'Condition mapping realization varies by case/repeat')
    return {'cases_checked_per_condition': dict(counts),
            'realized_zero_based_header_orders': {c: list(next(iter(v))) for c, v in realized_orders.items()},
            'focal_content_unchanged_in_all_12_context_contrasts': True,
            'scope': 'Three repetitions reuse each specified mapping realization; they are not three new permutations.'}


def contrast(data, model, left, right, ids, repeats=(1, 2, 3)):
    p = data['predictions']
    rows = []
    for i in ids:
        lv = [p[model, f'{left}_r{r}'][i] for r in repeats]
        rv = [p[model, f'{right}_r{r}'][i] for r in repeats]
        truth = data['truth'][i]
        rows.append({'id': i, 'source_group': truth['group'], 'true_label': truth['label'],
                     'left_predictions': lv, 'right_predictions': rv, 'category': classify(lv, rv)})
    counts = Counter(r['category'] for r in rows)
    within = [sum(p[model, f'{c}_r{a}'][i] != p[model, f'{c}_r{b}'][i] for i in ids)
              for c in [left, right] for a, b in combinations(repeats, 2)]
    cross = [sum(p[model, f'{left}_r{a}'][i] != p[model, f'{right}_r{b}'][i] for i in ids)
             for a, b in product(repeats, repeats)]
    for c in [left, right]:
        require(len({data['requests'][model, f'{c}_r{r}'] for r in repeats}) == 1, 'Repeat requests differ')
    return {'model': model, 'left': left, 'right': right, 'n_cases': len(ids),
            'n_source_units': len({r['source_group'] for r in rows}), 'repeats_per_condition': len(repeats),
            'counts': {c: counts[c] for c in CATEGORIES}, 'within_pair_flips': within,
            'cross_pair_flips': cross, 'stable_difference_ids': [r['id'] for r in rows if r['category'] == 'stable_different'],
            'cases': rows}


def audit_tep(inputs, d):
    p, truth = d['protocol'], d['truth']
    old_path = ROOT / 'results/revision_v3/data/datasets.json'
    inputs.check(old_path, p['original_dataset_sha256'])
    old = inputs.read(old_path)['datasets']['tep']
    prior_runs = {r['provenance']['simulation_run'] for split in ['train', 'test'] for r in old[split]}
    # Reconstruct the original conservative evenly-spaced run exclusion using
    # integer arithmetic, independently of numpy/the preparation routine.
    historical_runs = {((k * 249999) // (n - 1)) // 500 + 1 for n in [50, 100, 200] for k in range(n)}
    inputs.check(d['base'] / 'data_audit.json', p['data_audit_sha256'])
    old_audit = inputs.read(d['base'] / 'data_audit.json')
    require(prior_runs == set(old_audit['controlled_excluded_run_ids']), 'Original run inventory disagreement')
    require(historical_runs == set(old_audit['historical_excluded_run_ids']), 'Historical exclusion disagreement')
    tep_rows = [r for r in truth.values() if r['dataset'] == 'tep']
    main = [truth[i] for i in p['main_ids']]
    all_runs = {r['provenance']['simulation_run'] for r in tep_rows}
    main_runs = {r['provenance']['simulation_run'] for r in main}
    require(not all_runs & (prior_runs | historical_runs), 'Source overlap with excluded runs')
    require(all_runs == set(old_audit['new_tep_run_ids']), 'New run inventory disagreement')
    require(len(main_runs) == len(main) == 70, 'Main source units not one per case')
    require(Counter(r['label'] for r in main) == Counter({label: 10 for label in LABELS}), 'Main class counts changed')
    comparisons = [contrast(d, m, 'clean', c, p['main_ids']) for m in p['models'] for c in MAPS]
    comparisons += [contrast(d, m, a, b, p['focal_ids']) for m in p['models'] for a, b in CONTEXT]
    previous = inputs.read(ROOT / 'results/repeat_condition_v2/audit.json')
    previous_rows = {(r['model'], r['left_condition'], r['right_condition']): r for r in previous['contrasts'] if r['study'] == 'tep_three_repeats'}
    category_names = ['observed_repeat_stable_different', 'observed_repeat_stable_same', 'any_side_unstable', 'incomplete_planned_repeats']
    for c in comparisons:
        ref = previous_rows[c['model'], c['left'], c['right']]
        require(all(c['counts'][a] == ref[b] for a, b in zip(CATEGORIES, category_names)), 'Prior repeat counts disagree')
        require(c['stable_difference_ids'] == ref['stable_difference_ids'], 'Prior stable IDs disagree')
    rotation = next(c for c in comparisons if c['model'] == 'gpt-5.6-sol' and c['right'] == 'rotation')
    stable = [r for r in rotation['cases'] if r['category'] == 'stable_different']
    class_rows = []
    for label in LABELS:
        rows = [r for r in rotation['cases'] if r['true_label'] == label]
        selected = [r for r in rows if r['category'] == 'stable_different']
        shifts = Counter((r['left_predictions'][0], r['right_predictions'][0]) for r in selected)
        class_rows.append({'true_label': label, 'denominator': len(rows),
                           'stable_different': len(selected),
                           'stable_same': sum(r['category'] == 'stable_same' for r in rows),
                           'any_side_unstable': sum(r['category'] == 'any_side_unstable' for r in rows),
                           'incomplete': sum(r['category'] == 'incomplete' for r in rows),
                           'stable_transitions': [{'clean_label': a, 'rotation_label': b, 'count': n} for (a, b), n in sorted(shifts.items())]})
    direction = Counter('correct_to_wrong' if r['left_predictions'][0] == r['true_label'] else
                        'wrong_to_correct' if r['right_predictions'][0] == r['true_label'] else
                        'wrong_to_other_wrong' for r in stable)
    return {'source_inventory': {'original_controlled_excluded_runs': len(prior_runs),
                'historical_excluded_runs': len(historical_runs), 'new_tep_source_units_whole_study': len(all_runs),
                'new_tep_observations_whole_study': len(tep_rows), 'mapping_panel_source_units': len(main_runs),
                'mapping_panel_observations': len(main), 'context_focal_source_units': len(p['focal_ids']),
                'mapping_repeats_per_model_condition': 3, 'whole_study_calls': len(d['execution']),
                'whole_study_prediction_slots': sum(r['n'] for r in d['execution']),
                'overlap_with_controlled_or_historical_excluded_runs': [], 'mapping_panel_run_ids': sorted(main_runs),
                'scope': '70 mapping runs are a subset of the 112 TEP follow-up runs; repeats do not add sources. 84 calls/4854 slots include FD002.'},
            'sol_rotation': {'n': 70, 'repeats_per_condition': 3, 'counts': rotation['counts'],
                'class_rows': class_rows, 'stable_cases': stable,
                'stable_transition_directions': {k: direction[k] for k in ['correct_to_wrong', 'wrong_to_correct', 'wrong_to_other_wrong']}},
            'comparisons': comparisons, 'crosschecked_prior_repeat_audit': True,
            'input_transformations': audit_transformations(d)}


def audit_original_scores(inputs, d):
    score = inputs.read(d['base'] / 'analysis/scores.json')
    metrics = score['metrics'] if 'metrics' in score else score['request_metrics']
    for row in metrics:
        model = row.get('model', d['protocol'].get('model'))
        prediction = d['predictions'][model, row['group']]
        correct = sum(y == d['truth'][i]['label'] for i, y in prediction.items())
        require(correct == row['correct'], 'Original request correct count disagreement')
        require(correct / len(prediction) == row['accuracy'], 'Original request accuracy disagreement')
    if 'metrics' in score:
        for r in score['paired']:
            if r['kind'] not in ['mapping', 'context_size', 'background', 'position']:
                continue
            ids = d['protocol']['main_ids'] if r['kind'] == 'mapping' else d['protocol']['focal_ids']
            a = d['predictions'][r['model'], f"{r['left']}_r{r['repeat']}"]
            b = d['predictions'][r['model'], f"{r['right']}_r{r['repeat']}"]
            require(sum(a[i] != b[i] for i in ids) == r['flips'], 'Original paired flip disagreement')
    # Existing per-case export is an independently compared representation.
    cases_path = d['base'] / 'analysis/cases.csv'
    n_export = 0
    if cases_path.exists():
        inputs.digest(cases_path)
        with cases_path.open(newline='') as stream:
            for r in csv.DictReader(stream):
                predicted = d['predictions'][r['model'], f"{r['condition']}_r{r['repeat']}"][r['id']]
                require(predicted == r['predicted'] and d['truth'][r['id']]['label'] == r['true_label'], 'Per-case export disagreement')
                n_export += 1
    return {'request_metrics_verified': len(metrics), 'case_export_rows_verified': n_export}


def audit_context(inputs, d, old):
    p = d['protocol']
    inputs.check(d['base'] / 'inference/panels.json', p['panels_sha256'])
    panels = inputs.read(d['base'] / 'inference/panels.json')
    focal = panels['focal']
    for condition, source in p['source_payloads'].items():
        inputs.check(ROOT / source['source_prompt'], source['prompt_sha256'])
        inputs.check(ROOT / source['source_request'], source['request_sha256'])
        for name, group in d['groups'].items():
            if group['condition'] == condition:
                require(d['requests'][p['model'], name] == source['request_sha256'], 'Historical payload not preserved')
    for key, rel in [('source_protocol_sha256', 'protocol.json'), ('source_output_manifest_sha256', 'predictions/frozen_manifest.json'), ('source_answers_sha256', 'answers/records.json')]:
        inputs.check(old['base'] / rel, p[key])
    require(set(focal) == set(old['protocol']['focal_ids']), 'Replication focal IDs changed')
    require({i: d['truth'][i]['label'] for i in focal} == {i: old['truth'][i]['label'] for i in focal}, 'Replication truth differs')
    pred, model = d['predictions'], p['model']
    pairs, blocks, cases = [], [], []
    for index, names in enumerate(d['schedule']['blocks'], 1):
        names_a = [n for n in names if d['groups'][n]['condition'] == 'A']
        names_b = [n for n in names if d['groups'][n]['condition'] == 'B']
        require(len(names_a) == len(names_b) == 2, 'Incomplete balanced block')
        for kind, call_pairs in [('within_A', list(combinations(names_a, 2))), ('within_B', list(combinations(names_b, 2))), ('cross_A_B', list(product(names_a, names_b)))]:
            for a, b in call_pairs:
                pairs.append({'block': index, 'comparison': kind, 'left': a, 'right': b, 'n': len(focal),
                              'flips': sum(pred[model, a][i] != pred[model, b][i] for i in focal)})
        block_cases = [{'block': index, 'id': i, 'source_group': d['truth'][i]['source_group'],
                         'true_label': d['truth'][i]['label'], 'a_predictions': [pred[model, n][i] for n in names_a],
                         'b_predictions': [pred[model, n][i] for n in names_b],
                         'category': classify([pred[model, n][i] for n in names_a], [pred[model, n][i] for n in names_b])}
                        for i in focal]
        cases.extend(block_cases)
        counts = Counter(r['category'] for r in block_cases)
        blocks.append({'block': index, 'sequence': ''.join(d['groups'][n]['condition'] for n in names),
                       'focal_cases': len(focal), 'counts': {c: counts[c] for c in CATEGORIES}})
    original = inputs.read(d['base'] / 'analysis/scores.json')
    require([{k: r[k] for k in ['block', 'comparison', 'left', 'right', 'flips']} for r in pairs] ==
            [{k: r[k] for k in ['block', 'comparison', 'left', 'right', 'flips']} for r in original['paired']], 'Context pair disagreement')
    for new, ref in zip(blocks, original['block_summaries']):
        require(new['counts']['stable_different'] == ref['observed_two_repeat_stable_different'] and
                new['counts']['stable_same'] == ref['observed_two_repeat_stable_same'] and
                new['counts']['any_side_unstable'] == ref['any_condition_unstable'], 'Context block stability disagreement')
    cross = [r['flips'] for r in pairs if r['comparison'] == 'cross_A_B']
    within = [r['flips'] for r in pairs if r['comparison'] != 'cross_A_B']
    return {'source_inventory': {'focal_source_units': len({d['truth'][i]['source_group'] for i in focal}),
                'all_request_source_units': len({r['source_group'] for r in d['truth'].values()}),
                'new_source_units': 0, 'calls': len(d['execution']), 'blocks': len(blocks),
                'repeats_per_condition_per_block': 2, 'focal_prediction_slots': len(focal) * len(d['execution']),
                'all_prediction_slots': sum(r['n'] for r in d['execution'])},
            'cross_position_flip_range': [min(cross), max(cross)], 'within_condition_flip_range': [min(within), max(within)],
            'denominator_per_pair': len(focal), 'blocks': blocks, 'pairs': pairs, 'cases': cases,
            'original_scores_crosscheck': audit_original_scores(inputs, d),
            'interpretation': 'Pairs and blocks reuse 21 focal sources. No net-effect subtraction, independent n expansion, equivalence inference or proof of absence is justified.'}


def verify_rotation_from_export(inputs, expected):
    """A second calculation uses relational aggregation of the existing export.

    It does not call classify/contrast or use their intermediate case arrays.
    This verifies arithmetic by another implementation, not new empirical data.
    """
    path = ROOT / 'results/independent_validation/analysis/cases.csv'
    inputs.digest(path)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE predictions (id TEXT, model TEXT, condition TEXT, repeat INTEGER, truth TEXT, predicted TEXT)')
        db.executemany('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)',
                       [(r['id'], r['model'], r['condition'], int(r['repeat']), r['true_label'], r['predicted']) for r in rows])
        selected = db.execute('''
            SELECT id, MIN(truth),
                   MIN(CASE WHEN condition='clean' THEN predicted END),
                   MIN(CASE WHEN condition='rotation' THEN predicted END)
            FROM predictions
            WHERE model='gpt-5.6-sol' AND condition IN ('clean', 'rotation')
            GROUP BY id
            HAVING COUNT(*)=6 AND COUNT(DISTINCT condition || ':' || repeat)=6
               AND COUNT(DISTINCT truth)=1
               AND COUNT(DISTINCT CASE WHEN condition='clean' THEN predicted END)=1
               AND COUNT(DISTINCT CASE WHEN condition='rotation' THEN predicted END)=1
               AND MIN(CASE WHEN condition='clean' THEN predicted END)
                    <> MIN(CASE WHEN condition='rotation' THEN predicted END)
            ORDER BY id
        ''').fetchall()
    calculated = sorted((r['id'], r['true_label'], r['left_predictions'][0], r['right_predictions'][0]) for r in expected)
    require(selected == calculated, 'Independent SQL/export calculation disagrees with raw-response audit')
    return {'method': 'SQLite relational aggregation of the original 4854-row per-case CSV, separate from raw-response array classification.',
            'verified_stable_different_count': len(selected), 'verified_ids': [r[0] for r in selected],
            'all_correct_to_wrong': all(truth == clean and truth != rotated for _, truth, clean, rotated in selected),
            'new_empirical_evidence': False, 'passed': True}


def run(output):
    inputs = Inputs()
    independent = load_frozen(inputs, 'independent_validation')
    context = load_frozen(inputs, 'context_blocked_v1')
    tep = audit_tep(inputs, independent)
    tep['original_scores_crosscheck'] = audit_original_scores(inputs, independent)
    blocked = audit_context(inputs, context, independent)
    verification = verify_rotation_from_export(inputs, tep['sol_rotation']['stable_cases'])
    result = {'schema': 'evidence-boundary-independent-audit-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'new_model_calls': 0, 'network_requests': 0, 'new_fits': 0, 'answer_labels_read': True,
              'analysis_type': 'Post hoc descriptive computational audit; not preregistered, independent team replication, or human peer review.',
              'method': 'Standard-library SHA-256, direct strict-JSON raw/event parsing, ID-aligned label comparison, independently reconstructed source exclusions; no imports from original scorers or repeat-audit script.',
              'tep': tep, 'context_blocked': blocked,
              'independent_calculation_verification': verification,
              'verification_scope': 'Raw means the saved service response and event artifacts. Original RData source rows were not re-extracted in this audit; source separation is checked against the frozen answer provenance and original run inventories. Server-side weights, timestamps, and hidden mechanisms cannot be verified.',
              'evidence_ladder': [
                  {'level': 'integrity', 'supported': 'Frozen code, protocol, request, response, event and parsed artifacts agree; original per-call scores and case exports are reproduced.'},
                  {'level': 'finite_repeat_observation', 'supported': 'The specified Sol cyclic mismatch has repeat-stable changes on 20 of 70 source-disjoint mapping cases, all correct-to-wrong; the complete class denominators and every repeat label are retained.'},
                  {'level': 'scope_and_replication', 'supported': 'The concentration is not shared by all mapping realizations/models; blocked context follow-up does not reproduce a stable large positional effect.'},
                  {'level': 'engineering_validation', 'supported': False, 'reason': 'These observations do not establish deployable diagnostic reliability, a universally effective check, or transferable factory benefit.'}],
              'risks': [
                  'The three-repeat stable category is descriptive and post hoc; shared calls and panels prevent treating all case/repeat combinations as independent.',
                  'Specified cyclic label/value mismatch changes physical input meaning; its failure cannot establish general mapping sensitivity or internal semantic reasoning.',
                  'Original TEP repeat indices are not clock-time matched. ABBA/BAAB balances ordinal position but does not eliminate unequal durations or service drift.',
                  'Zero stable differences and overlapping flip ranges establish neither equivalence nor absence of a context effect.',
                  'Source-disjoint simulation runs and finite measured rigs do not establish field performance or stable engineering gains; weak bearing baselines leave representation/training adequacy unresolved.'],
              'manifest_file_counts': {name: d['manifest_files_verified'] for name, d in [('independent_validation', independent), ('context_blocked_v1', context)]},
              'source_artifacts': independent['execution'] + context['execution'],
              'analysis_code_sha256': inputs.digest(Path(__file__)), 'input_sha256': inputs.hashes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'rotation_counts': tep['sol_rotation']['counts'],
                      'rotation_directions': tep['sol_rotation']['stable_transition_directions'],
                      'context_cross': blocked['cross_position_flip_range'], 'context_within': blocked['within_condition_flip_range'],
                      'manifest_files': result['manifest_file_counts'], 'original_scores': tep['original_scores_crosscheck']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/evidence_boundary_audit_2026-09-30.json')
    run(parser.parse_args().output)
