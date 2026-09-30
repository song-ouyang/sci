"""Audit saved conversation-mode artifacts; not a certification of inference.

Read a post-score snapshot, preserve failed outcomes, and recompute descriptive
statistics from full precision. Never call models or alter the submitted records.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analysis'))
from audit_revision_v3_claude_extension import strict_json, validate_array
from score_revision_v3 import metric_values
from recheck_claude_descriptive import independent_metrics

BASE = ROOT / 'results/claude_conversation_exploratory_v1'
OUT = BASE / 'audit_2026-09-28'
CONDITIONS = ('cmapss_clean', 'tep_clean', 'tep_misnamed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(success_path, failure_paths, group):
    if success_path.exists() and failure_paths:
        raise ValueError('Both success and failure artifacts exist; resolve provenance explicitly')
    if success_path.exists():
        rows = strict_json(success_path.read_text())
        try:
            validate_array(rows, group)
        except ValueError as exc:
            return 'failed_invalid_array', str(exc), None
        return 'array_complete', None, rows
    if failure_paths:
        if len(failure_paths) != 1:
            raise ValueError('Multiple failure records require explicit attempt accounting')
        failed = strict_json(failure_paths[0].read_text())
        if isinstance(failed, list):
            return 'failed_incomplete', f'{len(failed)}/{len(group["samples"])} stored records; entire batch N/A', None
        return 'failed_off_task', failed.get('reason', 'recorded_failure'), None
    return 'unrecorded', 'No success or failure artifact', None


def summarize(values):
    return {'mean_pct': statistics.mean(values) if values else None,
            'sample_sd_pp': statistics.stdev(values) if len(values) > 1 else None,
            'n_available': len(values), 'n_planned': 3, 'repeat_values_pct': values}


def paired_summary(rows):
    complete = [row for row in rows if row['status'] == 'paired_arrays_available']
    if not complete:
        return {'paired_repeats': [], 'accuracy_delta_misnamed_minus_clean_pp': None}
    return {'paired_repeats': [r['repeat'] for r in complete],
            'clean_accuracy_mean_pct': statistics.mean(r['clean_accuracy_pct'] for r in complete),
            'misnamed_accuracy_mean_pct': statistics.mean(r['misnamed_accuracy_pct'] for r in complete),
            'accuracy_delta_misnamed_minus_clean_pp': statistics.mean(r['accuracy_delta_misnamed_minus_clean_pp'] for r in complete),
            'macro_f1_delta_misnamed_minus_clean_pp': statistics.mean(r['macro_f1_delta_misnamed_minus_clean_pp'] for r in complete),
            'flip_counts': [r['flips'] for r in complete],
            'paired_cases_each_repeat': [r['n_paired'] for r in complete]}


def main():
    snapshot = OUT / 'snapshot'
    manifest = strict_json((OUT / 'snapshot_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(snapshot / name) != expected:
            raise ValueError('Post-score snapshot changed: ' + name)
    protocol = strict_json((snapshot / 'protocol.json').read_text())
    registry = strict_json((snapshot / 'session_registry.json').read_text())['sessions']
    supplied = strict_json((snapshot / 'scores.json').read_text())['sessions']
    groups_path = ROOT / 'results/revision_v3/inference/batches.json'
    groups = strict_json(groups_path.read_text())['groups']
    prompt_checks = []
    for condition in CONDITIONS:
        p = ROOT / f'results/revision_v3/inference/prompts/{condition}_r1.txt'
        raw = p.read_bytes()
        prompt_checks.append({'condition': condition, 'bytes': len(raw),
                              'declared_hash_unspecified_algorithm': protocol['prompt_hashes'][condition],
                              'current_sha256': sha(p), 'current_md5': hashlib.md5(raw).hexdigest(),
                              'declared_matches_sha256': protocol['prompt_hashes'][condition] == sha(p),
                              'declared_matches_md5': protocol['prompt_hashes'][condition] == hashlib.md5(raw).hexdigest()})
    data_path = ROOT / 'results/revision_v3/data/datasets.json'
    parent = strict_json((ROOT / 'results/revision_v3/protocol.json').read_text())
    if sha(data_path) != parent['dataset_sha256']:
        raise ValueError('Original answer data hash mismatch')
    datasets = strict_json(data_path.read_text())['datasets']
    sessions, aggregates, pairs, cases = {}, [], [], []
    predictions = {}
    for model in ('sonnet', 'opus'):
        for condition in CONDITIONS:
            available = []
            for repeat in (1, 2, 3):
                name = f'{condition}_r{repeat}'
                key = f'{model}/{name}'
                folder = snapshot / 'predictions' / model
                state, error, rows = outcome(folder / f'{name}_predictions.json', list(folder.glob(f'{name}_FAILED_*.json')), groups[name])
                row = {'model_alias': model, 'condition': condition, 'repeat': repeat,
                       'status': state, 'failure_reason': error, 'eligible_for_controlled_primary': False,
                       'inference_provenance_verified': False, 'expected_n': groups[name]['n_samples'],
                       'submitted_score_status': supplied[key]['status'],
                       'registry_status': registry[f'{model}/{condition}/r{repeat}']['status'],
                       'accuracy': None, 'macro_f1': None, 'n_scored': 0}
                if rows is not None:
                    dataset = condition.split('_')[0]
                    truth_map = {r['id']: r for r in datasets[dataset]['test']}
                    if set(truth_map) != {r['id'] for r in rows}:
                        raise ValueError('ID-set mismatch with evaluation data')
                    truth = [truth_map[r['id']]['label'] for r in rows]
                    preds = [r['predicted'] for r in rows]
                    values = metric_values(truth, preds, groups[name]['valid_labels'], dataset)
                    independent = independent_metrics(truth, preds, groups[name]['valid_labels'])
                    assert abs(values['accuracy'] - independent[0]) < 1e-12
                    assert abs(values['macro_f1'] - independent[1]) < 1e-12
                    row.update(values, n_scored=len(rows))
                    row['submitted_rounded_metrics_match'] = all(round(values[k] * 100, 1) == supplied[key][k] for k in ('accuracy', 'macro_f1'))
                    available.append(row)
                    predictions[key] = {r['id']: r['predicted'] for r in rows}
                    cases.extend({'model_alias': model, 'condition': condition, 'repeat': repeat, 'id': p['id'],
                                  'truth': t, 'predicted': p['predicted'], 'correct': int(t == p['predicted']),
                                  'eligible_for_controlled_primary': False} for p, t in zip(rows, truth))
                elif state == 'failed_invalid_array':
                    # A separate post-hoc arithmetic sensitivity check may align
                    # a complete unique ID set, but never repairs saved output or
                    # changes the exact-order failure to a protocol success.
                    original = strict_json((folder / f'{name}_predictions.json').read_text())
                    if isinstance(original, list) and all(isinstance(p, dict) and set(p) == {'id', 'predicted'} and isinstance(p['id'], str) for p in original):
                        by_id = {p['id']: p for p in original}
                        ids = [p['id'] for p in groups[name]['samples']]
                        if len(by_id) == len(original) == len(ids) and set(by_id) == set(ids):
                            aligned = [by_id[i] for i in ids]
                            validate_array(aligned, groups[name])
                            dataset = condition.split('_')[0]
                            truth_map = {p['id']: p['label'] for p in datasets[dataset]['test']}
                            truth = [truth_map[i] for i in ids]
                            pred = [by_id[i]['predicted'] for i in ids]
                            sensitivity = metric_values(truth, pred, groups[name]['valid_labels'], dataset)
                            ia, iff = independent_metrics(truth, pred, groups[name]['valid_labels'])
                            assert abs(ia - sensitivity['accuracy']) < 1e-12 and abs(iff - sensitivity['macro_f1']) < 1e-12
                            row['posthoc_id_aligned_arithmetic_only'] = sensitivity
                            row['order_mismatches'] = [{'position_one_based': n + 1, 'expected_id': wanted, 'returned_id': actual['id']} for n, (wanted, actual) in enumerate(zip(ids, original)) if wanted != actual['id']]
                            row['submitted_rounded_metrics_match'] = all(round(sensitivity[k] * 100, 1) == supplied[key][k] for k in ('accuracy', 'macro_f1'))
                sessions[key] = row
            aggregates.append({'model_alias': model, 'condition': condition,
                               'available_repeats': [r['repeat'] for r in available],
                               'accuracy': summarize([r['accuracy'] * 100 for r in available]),
                               'macro_f1': summarize([r['macro_f1'] * 100 for r in available])})
        for repeat in (1, 2, 3):
            a, b = f'{model}/tep_clean_r{repeat}', f'{model}/tep_misnamed_r{repeat}'
            row = {'model_alias': model, 'repeat': repeat, 'status': 'pair_unavailable'}
            if a in predictions and b in predictions:
                left, right = predictions[a], predictions[b]
                if left.keys() != right.keys():
                    raise ValueError('Pair IDs do not match')
                row.update(status='paired_arrays_available', n_paired=len(left),
                           flips=sum(left[i] != right[i] for i in left),
                           clean_accuracy_pct=100 * sessions[a]['accuracy'],
                           misnamed_accuracy_pct=100 * sessions[b]['accuracy'],
                           accuracy_delta_misnamed_minus_clean_pp=100 * (sessions[b]['accuracy'] - sessions[a]['accuracy']),
                           macro_f1_delta_misnamed_minus_clean_pp=100 * (sessions[b]['macro_f1'] - sessions[a]['macro_f1']))
            pairs.append(row)
    sensitivity_aggregates, sensitivity_pairs = [], []
    for model in ('sonnet', 'opus'):
        for condition in CONDITIONS:
            available = [(r['repeat'], r if r['accuracy'] is not None else r.get('posthoc_id_aligned_arithmetic_only')) for r in sessions.values() if r['model_alias'] == model and r['condition'] == condition]
            available = [(repeat, v) for repeat, v in available if v is not None]
            sensitivity_aggregates.append({'model_alias': model, 'condition': condition,
                                          'available_repeats': [repeat for repeat, _ in available],
                                          'accuracy': summarize([v['accuracy'] * 100 for _, v in available]),
                                          'macro_f1': summarize([v['macro_f1'] * 100 for _, v in available])})
        for repeat in (1, 2, 3):
            left, right = [sessions[f'{model}/{cond}_r{repeat}'] for cond in ('tep_clean', 'tep_misnamed')]
            a = left if left['accuracy'] is not None else left.get('posthoc_id_aligned_arithmetic_only')
            b = right if right['accuracy'] is not None else right.get('posthoc_id_aligned_arithmetic_only')
            if a is None or b is None:
                continue
            maps = [{p['id']: p['predicted'] for p in strict_json((snapshot / f'predictions/{model}/{cond}_r{repeat}_predictions.json').read_text())} for cond in ('tep_clean', 'tep_misnamed')]
            assert maps[0].keys() == maps[1].keys()
            sensitivity_pairs.append({'model_alias': model, 'repeat': repeat, 'status': 'paired_arrays_available',
                                      'n_paired': len(maps[0]), 'flips': sum(maps[0][i] != maps[1][i] for i in maps[0]),
                                      'clean_accuracy_pct': a['accuracy'] * 100, 'misnamed_accuracy_pct': b['accuracy'] * 100,
                                      'accuracy_delta_misnamed_minus_clean_pp': (b['accuracy'] - a['accuracy']) * 100,
                                      'macro_f1_delta_misnamed_minus_clean_pp': (b['macro_f1'] - a['macro_f1']) * 100})
    report = {'type': 'post_score_descriptive_audit_of_tool_enabled_conversations',
              'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'eligible_for_controlled_primary': False, 'native_identity_verified': False,
              'pre_score_freeze_verified': False, 'raw_response_chain_verified': False,
              'planned_sessions': 18, 'complete_arrays': len(predictions),
              'failed_artifacts': sum(r['status'].startswith('failed_') for r in sessions.values()),
              'scored_labels': len(cases), 'planned_labels': 1200, 'distinct_sources': 130,
              'snapshot_manifest_sha256': sha(OUT / 'snapshot_manifest.json'),
              'dataset_sha256': sha(data_path), 'batches_sha256': sha(groups_path),
              'auditor_sha256': sha(Path(__file__)), 'prompt_checks': prompt_checks,
              'sessions': sessions, 'aggregates': aggregates, 'pairs': pairs,
              'paired_summaries': {m: paired_summary([r for r in pairs if r['model_alias'] == m]) for m in ('sonnet', 'opus')},
              'posthoc_id_alignment_sensitivity': {
                  'note': 'Allows the one complete but misordered array for retrospective arithmetic only. This does not alter original files or waive the exact-order prompt requirement.',
                  'aggregates': sensitivity_aggregates,
                  'paired_summaries': {m: paired_summary([r for r in sensitivity_pairs if r['model_alias'] == m]) for m in ('sonnet', 'opus')}},
              'note': 'Array completeness and arithmetic agreement do not validate tool-enabled inference. All failed arrays remain unscored. No logs or prospective freeze reconstructed.'}
    (OUT / 'corrected_descriptive_scores.json').write_text(json.dumps(report, indent=2) + '\n')
    with (OUT / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)
    print(json.dumps({k: report[k] for k in ('complete_arrays', 'failed_artifacts', 'scored_labels', 'prompt_checks', 'aggregates', 'paired_summaries')}, indent=2))


if __name__ == '__main__':
    main()
