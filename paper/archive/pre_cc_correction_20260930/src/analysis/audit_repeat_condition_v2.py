"""Offline, post hoc descriptive comparison of repeat and condition differences.

Reads frozen requests/predictions, never answer labels. It performs no inference,
network access, significance testing, noise subtraction, or protocol changes.
All repeat pairings reuse the same cases and calls; they are not independent n.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_revision_v3 import parse_raw

OUT = ROOT / 'results/repeat_condition_v2'
MAPPING = ('rotation', 'random_1', 'random_2', 'random_3', 'same_unit_swap', 'pair_shuffle')
CONTEXT = (('a_front', 'a_back'), ('b_front', 'b_back'),
           ('a_front', 'b_front'), ('a_back', 'b_back'))
CATEGORIES = ('observed_repeat_stable_different', 'observed_repeat_stable_same',
              'any_side_unstable', 'incomplete_planned_repeats')


def need(ok, text):
    if not ok:
        raise ValueError(text)


def digest(path):
    need(path.is_file() and not path.is_symlink(), f'Missing or symlinked input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    digest(path)
    return json.loads(path.read_text())


def resolve_within(base, relative):
    path = Path(relative)
    need(not path.is_absolute() and '..' not in path.parts, f'Unsafe manifest path: {relative}')
    return base / path


def verify_study(base):
    """Validate original frozen code, protocol, payloads and every frozen output."""
    protocol = read(base / 'protocol.json')
    ph = digest(base / 'protocol.json')
    need(ph == (base / 'protocol.sha256').read_text().strip(), 'Protocol hash mismatch')
    manifest = read(base / 'predictions/frozen_manifest.json')
    need(manifest['protocol_sha256'] == ph, 'Frozen manifest/protocol mismatch')
    checked = {}
    for relative, expected in manifest['files'].items():
        path = resolve_within(base / 'predictions', relative)
        actual = digest(path)
        need(actual == expected, f'Frozen output changed: {relative}')
        checked[str(path.relative_to(ROOT))] = actual
    for relative, expected in protocol['code_sha256'].items():
        path = resolve_within(ROOT, relative)
        actual = digest(path)
        need(actual == expected, f'Frozen code changed: {relative}')
        checked[str(path.relative_to(ROOT))] = actual
    batches = base / 'inference/batches.json'
    need(digest(batches) == protocol['input_sha256'], 'Frozen batches changed')
    groups = read(batches)['groups']
    prompt_hashes = protocol.get('prompt_sha256', protocol.get('prompts_sha256'))
    need(set(prompt_hashes) == set(groups), 'Prompt inventory mismatch')
    for name, expected in prompt_hashes.items():
        path = base / 'inference/prompts' / f'{name}.txt'
        need(digest(path) == expected, f'Frozen prompt changed: {name}')
        checked[str(path.relative_to(ROOT))] = expected
    for path in (base / 'protocol.json', base / 'protocol.sha256',
                 base / 'predictions/frozen_manifest.json', batches):
        checked[str(path.relative_to(ROOT))] = digest(path)
    return protocol, groups, manifest, checked


def load_condition(base, protocol, groups, manifest, model, condition, repeats, ids):
    """Missing/failed requests remain unavailable; later repeats never replace them."""
    predictions, request_bytes, statuses, full_sizes, samples = {}, {}, {}, {}, {}
    for repeat in repeats:
        name = f'{condition}_r{repeat}'
        need(name in groups, f'Missing planned group: {name}')
        group = groups[name]
        ordered_ids = [r['id'] for r in group['samples']]
        need(len(ordered_ids) == len(set(ordered_ids)), 'Duplicate scheduled IDs')
        need(set(ids) <= set(ordered_ids), f'Comparison IDs absent from {name}')
        samples[repeat] = {r['id']: r['prompt'] for r in group['samples'] if r['id'] in ids}
        full_sizes[repeat] = len(ordered_ids)
        prefix = f'{model}/{name}'
        meta_rel = prefix + '_metadata.json'
        need(meta_rel in manifest['files'], f'Metadata not frozen: {meta_rel}')
        metadata = read(base / 'predictions' / meta_rel)
        need(metadata['model_requested'] == model and metadata['batch'] == name, 'Metadata identity mismatch')
        need(metadata['input_sha256'] == protocol['input_sha256'], 'Metadata input hash mismatch')
        need(metadata['protocol_sha256'] == digest(base / 'protocol.json'), 'Metadata protocol mismatch')
        statuses[repeat] = metadata['status']
        need(statuses[repeat] in {'complete', 'failed', 'not_attempted', 'invalid_response',
                                 'invalid_tool_use', 'invalid_event_log'}, f'Nonterminal status: {name}')
        req_rel = prefix + '_request.json'
        req_path = base / 'predictions' / req_rel
        if req_path.exists():
            need(req_rel in manifest['files'], 'Request not frozen')
            request = read(req_path)
            need(request['model'] == model, 'Requested model mismatch')
            need(request['reasoning_effort'] == protocol['reasoning_effort'], 'Reasoning setting mismatch')
            need(request['user_message'] == (base / 'inference/prompts' / f'{name}.txt').read_text(),
                 'Actual request differs from frozen prompt')
            request_bytes[repeat] = req_path.read_bytes()
        result_rel = prefix + '_results.json'
        result_path = base / 'predictions' / result_rel
        if statuses[repeat] != 'complete':
            need(not result_path.exists(), 'Invalid request must not supply scored results')
            predictions[repeat] = None
            continue
        need(repeat in request_bytes, 'Complete request has no request artifact')
        need(result_rel in manifest['files'], 'Prediction result not frozen')
        result = read(result_path)
        need(digest(result_path) == metadata['prediction_sha256'], 'Metadata/result hash mismatch')
        attempts = metadata['attempts']
        need(len(attempts) == 1 and attempts[0]['attempt'] == 1, 'Unexpected repeat attempt')
        raw_rel = f"{model}/{attempts[0]['response_file']}"
        need(raw_rel in manifest['files'], 'Raw response not frozen')
        parsed = parse_raw((base / 'predictions' / raw_rel).read_text(), group)
        need(parsed == result, 'Raw response and parsed predictions disagree')
        need([r['id'] for r in result] == ordered_ids, 'Prediction order/coverage mismatch')
        need(all(r['predicted'] in group['valid_labels'] for r in result), 'Unpermitted prediction label')
        full = {r['id']: r['predicted'] for r in result}
        predictions[repeat] = {i: full[i] for i in ids}
    payloads = [(base / 'inference/prompts' / f'{condition}_r{r}.txt').read_bytes() for r in repeats]
    need(all(payload == payloads[0] for payload in payloads), f'Within-condition prompts differ: {condition}')
    need(len(set(request_bytes.values())) <= 1, f'Within-condition requests differ: {condition}')
    return {'predictions': predictions, 'statuses': statuses, 'full_sizes': full_sizes,
            'samples': samples, 'request_hashes': {str(r): hashlib.sha256(x).hexdigest() for r, x in request_bytes.items()},
            'identical_recorded_requests': len(request_bytes) >= 2,
            'recorded_request_count': len(request_bytes)}


def compare_pair(left, right, ids):
    if left is None or right is None:
        return {'available': False, 'n_scheduled_cases': len(ids), 'n_paired_cases': 0,
                'flips': None, 'flip_rate': None}
    need(set(left) == set(ids) == set(right), 'Paired ID mismatch')
    flips = sum(left[i] != right[i] for i in ids)
    return {'available': True, 'n_scheduled_cases': len(ids), 'n_paired_cases': len(ids),
            'flips': flips, 'flip_rate': flips / len(ids)}


def classify_case(left, right, repeats, case_id):
    lv = [None if left[r] is None else left[r][case_id] for r in repeats]
    rv = [None if right[r] is None else right[r][case_id] for r in repeats]
    complete = None not in lv and None not in rv
    ls = len(set(lv)) == 1 if None not in lv else None
    rs = len(set(rv)) == 1 if None not in rv else None
    if not complete:
        category = CATEGORIES[3]
    elif not ls or not rs:
        category = CATEGORIES[2]
    else:
        category = CATEGORIES[1] if lv[0] == rv[0] else CATEGORIES[0]
    return {'id': case_id, 'left_predictions': lv, 'right_predictions': rv,
            'all_planned_repeats_available': complete, 'left_stable': ls, 'right_stable': rs,
            'category': category}


def minmax(rows):
    values = [r['flip_rate'] for r in rows if r['available']]
    return (min(values), max(values)) if values else (None, None)


def analyze_contrast(study, model, name, kind, left_name, right_name, ids, repeats, left, right):
    common = {'study': study, 'model': model, 'comparison': name, 'kind': kind,
              'left_condition': left_name, 'right_condition': right_name}
    within = []
    for side, condition, data in [('left', left_name, left), ('right', right_name, right)]:
        for a, b in itertools.combinations(repeats, 2):
            within.append(dict(common, side=side, condition=condition, left_repeat=a, right_repeat=b,
                               pairing_scope=('between_execution_blocks' if study.startswith('hydraulic')
                                              else 'between_calls_same_condition_not_time_matched'),
                               **compare_pair(data['predictions'][a], data['predictions'][b], ids)))
    cross = [dict(common, left_repeat=a, right_repeat=b, same_repeat_index=(a == b),
                  pairing_scope=(('within_execution_block' if a == b else 'between_execution_blocks')
                                 if study.startswith('hydraulic') else
                                 ('same_repeat_index_not_temporally_matched' if a == b else 'different_repeat_indices')),
                  **compare_pair(left['predictions'][a], right['predictions'][b], ids))
             for a, b in itertools.product(repeats, repeats)]
    cases = [dict(common, **classify_case(left['predictions'], right['predictions'], repeats, i)) for i in ids]
    counts = Counter(r['category'] for r in cases)
    summary = dict(common, n_cases=len(ids), repeats_per_condition=len(repeats),
                   repeats=','.join(map(str, repeats)),
                   left_valid_repeats=sum(x is not None for x in left['predictions'].values()),
                   right_valid_repeats=sum(x is not None for x in right['predictions'].values()),
                   within_pairs_planned_per_condition=len(repeats) * (len(repeats) - 1) // 2,
                   cross_pairs_planned=len(cross), cross_pairs_available=sum(r['available'] for r in cross),
                   **{category: counts[category] for category in CATEGORIES})
    for side in ('left', 'right'):
        summary[f'{side}_within_flip_rate_min'], summary[f'{side}_within_flip_rate_max'] = minmax([r for r in within if r['side'] == side])
    summary['cross_flip_rate_min'], summary['cross_flip_rate_max'] = minmax(cross)
    summary['stable_difference_ids'] = [r['id'] for r in cases if r['category'] == CATEGORIES[0]]
    return summary, within, cross, cases


def process_study(base, study, include_hydraulic=False):
    protocol, groups, manifest, hashes = verify_study(base)
    if include_hydraulic:
        repeats, models = [1, 2], [protocol['model']]
        panels = read(base / 'inference/panels.json')
        ids_by_kind = {'mapping': panels['focal'], 'context': panels['focal'], 'duplicate': panels['focal']}
        contrasts = [('mapping', 'clean', c) for c in ('pair_reverse', 'descriptor_rotation')]
        contrasts += [('context', a, b) for a, b in CONTEXT]
        contrasts += [('duplicate', 'clean', 'clean_repeat'), ('duplicate', 'a_front', 'a_front_repeat')]
        # Selected after the protocol-directed stop, before this post hoc
        # analysis: retain both complete blocks together, never fill from R3.
        for r in repeats:
            for model in models:
                for condition in {c for _, a, b in contrasts for c in (a, b)}:
                    meta = read(base / 'predictions' / model / f'{condition}_r{r}_metadata.json')
                    need(meta['status'] == 'complete', 'Hydraulic analysis requires complete designated blocks 1 and 2')
    else:
        repeats, models = [1, 2, 3], protocol['models']
        ids_by_kind = {'mapping': protocol['main_ids'], 'context': protocol['focal_ids']}
        need(len(ids_by_kind['mapping']) == 70 and len(ids_by_kind['context']) == 21, 'Unexpected TEP comparison panels')
        contrasts = [('mapping', 'clean', c) for c in MAPPING] + [('context', a, b) for a, b in CONTEXT]
    summaries, within, cross, cases, request_checks = [], [], [], [], []
    loaded = {}
    for model in models:
        for kind, a, b in contrasts:
            ids = ids_by_kind[kind]
            need(len(ids) == len(set(ids)) and ids, 'Invalid analysis panel IDs')
            for condition in (a, b):
                key = (model, condition, tuple(ids))
                if key not in loaded:
                    loaded[key] = load_condition(base, protocol, groups, manifest, model, condition, repeats, ids)
                    request_checks.append({'study': study, 'model': model, 'condition': condition,
                                           'n_comparison_cases': len(ids), 'repeats': repeats,
                                           'recorded_request_count': loaded[key]['recorded_request_count'],
                                           'identical_recorded_requests': loaded[key]['identical_recorded_requests'],
                                           'request_hashes': loaded[key]['request_hashes']})
            left, right = loaded[model, a, tuple(ids)], loaded[model, b, tuple(ids)]
            if kind == 'context':
                expected_size = 36 if include_hydraulic else 42
                need(set(left['full_sizes'].values()) == {expected_size} == set(right['full_sizes'].values()),
                     'Fixed-size context contrast changed request size')
                for r in repeats:
                    need(left['samples'][r] == right['samples'][r], 'Context contrast changes focal record content')
            result = analyze_contrast(study, model, f'{a}__{b}', kind, a, b, ids, repeats, left, right)
            summaries.append(result[0]); within.extend(result[1]); cross.extend(result[2]); cases.extend(result[3])
    return {'contrasts': summaries, 'within_condition_pairs': within, 'cross_condition_pairs': cross,
            'case_stability': cases, 'request_checks': request_checks, 'input_hashes': hashes,
            'frozen_output_files_checked': len(manifest['files']),
            'original_plan_status': {k: v for k, v in manifest.items() if k != 'files'}}


def write_csv(path, rows):
    def scalar(value):
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: scalar(v) for k, v in row.items()} for row in rows)


def brief(result):
    lines = ['# 重复调用与条件差异：事后描述性审计', '',
             '这是既有冻结输出的离线再分析，不是新实验、预注册指标或独立人类同行评审。未读取逐例真实标签。', '',
             '“观察到的重复稳定差异”仅表示：指定的全部重复中，两条件各自标签完全一致且两侧不同。它不是总体稳定性、显著性或因果证据。', '',
             'TEP的R编号不是匹配的执行时间块；条件与调用时间仍可能混淆。这些稳定计数不隔离输入效应，也不是因果效应的下界。没有观察到稳定差异不等于没有条件效应。', '',
             '每个条件内部列出全部重复对；条件之间列出所有交叉重复对。配对复用同一组案例和调用，不能当作独立样本、用于扩大 n，或相减形成净效应。', '',
             '| 数据 / 模型 | 条件对 | 案例 | 重复/侧 | 稳定不同 | 稳定相同 | 任一不稳定 | 重复缺失 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in result['contrasts']:
        lines.append(f"| {row['study']} / {row['model']} | {row['left_condition']} → {row['right_condition']} | {row['n_cases']} | {row['repeats_per_condition']} | {row[CATEGORIES[0]]} | {row[CATEGORIES[1]]} | {row[CATEGORIES[2]]} | {row[CATEGORIES[3]]} |")
    lines += ['', '液压结果若纳入，仅单列两个完整执行块（R1/R2）。R3部分完成、R4/R5未执行仍属于原试验缺失；这里不以晚轮补位。clean与clean_repeat是两个已规定的相同输入条件；其同块调用对与跨块调用对分别由pairing_scope字段标记，不能将全交叉配对范围误读为同块duplicate波动。不同数据集及两轮/三轮稳定计数不可横向排名。', '',
              '输出：contrasts.csv 给出条件级计数和观察范围；within_condition_pairs.csv 与 cross_condition_pairs.csv 保留每一对；case_stability.csv 保留各ID的全部标签及分类；audit.json 记录输入哈希和每个条件的字节一致性核验。条件内配对在不同对照中可重复出现，comparison列标识所属对照。', '',
              '结论边界：大量一次调用差异可以伴随条件内不稳定；少数指定条件也可能在所有已观察重复中各自保持不同标签。前者限制单次效果解释，后者提供可继续验证的具体案例。二者均不能排除抽样、服务时间或有限重复带来的偶然性。', '',
              '## 本次观察', '']
    tep = [r for r in result['contrasts'] if r['study'] == 'tep_three_repeats']
    rotation = next((r for r in tep if r['model'] == 'gpt-5.6-sol' and r['comparison'] == 'clean__rotation'), None)
    astra = next((r for r in tep if r['model'] == 'gpt-6-astra' and r['comparison'] == 'a_back__b_back'), None)
    if rotation:
        lines.append(f"Sol的clean/rotation在70例中有{rotation[CATEGORIES[0]]}例观察到重复稳定不同、{rotation[CATEGORIES[1]]}例稳定相同、{rotation[CATEGORIES[2]]}例至少一侧不稳定。其余五个映射对照与四个固定规模上下文对照均未观察到两侧各三轮稳定且不同的案例；这不表示不存在条件差异。")
    if astra:
        lines.append(f"Astra的六个映射对照均无观察到的稳定不同；A-back/B-back在21例中有{astra[CATEGORIES[0]]}例稳定不同、{astra[CATEGORIES[1]]}例稳定相同。另三个上下文对照各有一例至少一侧不稳定。")
    lines += ['', '## 离线重建', '', '```sh',
              '.venv/bin/python src/analysis/audit_repeat_condition_v2.py --include-hydraulic',
              '.venv/bin/python -m unittest tests.test_repeat_condition_v2 -v', '```', '']
    return '\n'.join(lines)


def run(output=OUT, include_hydraulic=False):
    studies = [process_study(ROOT / 'results/independent_validation', 'tep_three_repeats')]
    if include_hydraulic:
        studies.append(process_study(ROOT / 'results/hydraulic_validation_v1', 'hydraulic_two_complete_blocks', True))
    result = {'schema': 'repeat-condition-posthoc-v2', 'created_at': datetime.now(timezone.utc).isoformat(),
              'new_model_calls': 0, 'answer_labels_read': False, 'post_hoc': True,
              'interpretation': 'Descriptive finite-repeat categories, no novel estimator, significance test, causal subtraction, or independent-sample inflation. TEP repeat indices are not matched temporal blocks; condition/time confounding remains. Stable differences are not causal lower bounds; zero stable differences do not establish absence of condition effects.',
              'analysis_code_sha256': digest(Path(__file__)),
              'studies': [{k: v for k, v in s.items() if k not in ('contrasts', 'within_condition_pairs', 'cross_condition_pairs', 'case_stability')} for s in studies]}
    for field in ('contrasts', 'within_condition_pairs', 'cross_condition_pairs', 'case_stability'):
        result[field] = [row for study in studies for row in study[field]]
    output.mkdir(parents=True, exist_ok=True)
    for field in ('contrasts', 'within_condition_pairs', 'cross_condition_pairs', 'case_stability'):
        write_csv(output / f'{field}.csv', result[field])
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'README.md').write_text(brief(result))
    print(json.dumps({'contrasts': len(result['contrasts']), 'within_pairs': len(result['within_condition_pairs']),
                      'cross_pairs': len(result['cross_condition_pairs']), 'case_rows': len(result['case_stability']),
                      'new_model_calls': 0}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-hydraulic', action='store_true')
    args = parser.parse_args()
    run(include_hydraulic=args.include_hydraulic)
