"""Post-report arithmetic only, NEVER a controlled Claude scoring/eligibility gate.

Recomputes all legacy arrays, including provenance failures, solely to check
submitted summaries. Preserves the controlled scorer's N/A decisions. Does not
call models, repair records, create a prospective freeze, or infer model identity.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analysis'))
from audit_revision_v3_claude_extension import strict_json, validate_array
from score_revision_v3 import metric_values

BASE = ROOT / 'results/claude_descriptive_recheck_2026-09-28'
MODELS = ('claude-sonnet-4-6', 'claude-opus-5-5', 'claude-fable-5-1')
CONDITIONS = ('cmapss_clean', 'tep_clean', 'tep_misnamed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics_pct(values):
    if len(values) != 3:
        raise ValueError('This descriptive collection requires exactly three stored arrays')
    return {'mean_pct': statistics.mean(values), 'population_sd_pp_ddof0': statistics.pstdev(values),
            'sample_sd_pp_ddof1': statistics.stdev(values), 'repeat_values_pct': values}


def independent_metrics(truth, predicted, labels):
    if not truth or len(truth) != len(predicted):
        raise ValueError('Unequal or empty paired records')
    correct = sum(a == b for a, b in zip(truth, predicted))
    f1 = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return correct / len(truth), sum(f1) / len(labels)


def main():
    manifest = strict_json((BASE / 'snapshot_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(BASE / name) != expected:
            raise ValueError(f'Post-report snapshot changed: {name}')
    snap = BASE / 'artifact_snapshot'
    groups = strict_json((snap / 'parent/inference/batches.json').read_text())['groups']
    audit = strict_json((BASE / 'audit.json').read_text())
    statuses = {(r['model'], r['group']): r for r in audit['sessions']}
    # The legacy predictions have already been reported/scored. Reading the
    # established answer key here is explicitly retrospective, not a new trial.
    data_path = ROOT / 'results/revision_v3/data/datasets.json'
    data_hash = sha(data_path)
    protocol = strict_json((snap / 'parent/protocol.json').read_text())
    if protocol['dataset_sha256'] != data_hash:
        raise ValueError('Original evaluation dataset hash mismatch')
    datasets = strict_json(data_path.read_text())['datasets']
    supplied = strict_json((BASE / 'supplied_summary.json').read_text())['models']
    cases, sessions, aggregates, contrasts = [], [], [], []
    saved = {}
    for model in MODELS:
        for condition in CONDITIONS:
            subset = []
            dataset = condition.split('_')[0]
            evaluation = datasets[dataset]['test']
            truth_by_id = {row['id']: row for row in evaluation}
            if len(truth_by_id) != len(evaluation):
                raise ValueError('Duplicate answer ID')
            for repeat in (1, 2, 3):
                name = f'{condition}_r{repeat}'
                group = groups[name]
                preds = strict_json((snap / f'extension/predictions/{model}/{name}_results.json').read_text())
                validate_array(preds, group)
                if {p['id'] for p in preds} != truth_by_id.keys():
                    raise ValueError('Prediction/answer ID set mismatch')
                truth = [truth_by_id[p['id']]['label'] for p in preds]
                labels = [p['predicted'] for p in preds]
                values = metric_values(truth, labels, group['valid_labels'], dataset)
                a, f = independent_metrics(truth, labels, group['valid_labels'])
                if abs(a - values['accuracy']) > 1e-12 or abs(f - values['macro_f1']) > 1e-12:
                    raise ValueError('Independent arithmetic mismatch')
                row = {'model_label_unverified': model, 'condition': condition, 'repeat': repeat,
                       'eligible_for_primary_comparison': False,
                       'artifact_chain_consistent': statuses[model, name]['artifact_chain_consistent'], **values}
                sessions.append(row); subset.append(row)
                saved[model, condition, repeat] = preds
                cases.extend({'model_label_unverified': model, 'condition': condition, 'repeat': repeat,
                              'id': p['id'], 'source_unit': truth_by_id[p['id']]['group'],
                              'truth': t, 'predicted': p['predicted'], 'correct': int(t == p['predicted']),
                              'eligible_for_primary_comparison': False} for p, t in zip(preds, truth))
            acc = statistics_pct([r['accuracy'] * 100 for r in subset])
            f1 = statistics_pct([r['macro_f1'] * 100 for r in subset])
            reported = supplied[model]['cmapss' if dataset == 'cmapss' else condition]
            match = all(abs(reported[k] - value) < 1e-8 for k, value in {
                'acc_mean': acc['mean_pct'], 'acc_std': acc['population_sd_pp_ddof0'],
                'f1_mean': f1['mean_pct'], 'f1_std': f1['population_sd_pp_ddof0']}.items())
            aggregates.append({'model_label_unverified': model, 'condition': condition,
                               'accuracy': acc, 'macro_f1': f1, 'supplied_matches_ddof0': match,
                               'eligible_for_primary_comparison': False})
        for repeat in (1, 2, 3):
            clean = {p['id']: p['predicted'] for p in saved[model, 'tep_clean', repeat]}
            misnamed = {p['id']: p['predicted'] for p in saved[model, 'tep_misnamed', repeat]}
            if clean.keys() != misnamed.keys():
                raise ValueError('Paired IDs differ')
            a = next(r for r in sessions if r['model_label_unverified'] == model and r['condition'] == 'tep_clean' and r['repeat'] == repeat)
            b = next(r for r in sessions if r['model_label_unverified'] == model and r['condition'] == 'tep_misnamed' and r['repeat'] == repeat)
            flips = sum(clean[i] != misnamed[i] for i in clean)
            contrasts.append({'model_label_unverified': model, 'repeat': repeat, 'paired_n': len(clean),
                              'flips': flips, 'flip_pct': flips / len(clean) * 100,
                              'accuracy_clean_minus_misnamed_pp': 100 * (a['accuracy'] - b['accuracy']),
                              'macro_f1_clean_minus_misnamed_pp': 100 * (a['macro_f1'] - b['macro_f1']),
                              'eligible_for_primary_comparison': False})
    output = {'type': 'retrospective_arithmetic_check_NOT_validated_experiment_results',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'eligible_for_primary_comparison': False, 'answer_keys_opened_for_descriptive_check': True,
              'dataset_sha256': data_hash, 'script_sha256': sha(Path(__file__)),
              'source_snapshot_manifest_sha256': sha(BASE / 'snapshot_manifest.json'),
              'counts': {'stored_arrays': len(sessions), 'stored_labels': len(cases), 'distinct_sources': 130, 'new_model_calls': 0},
              'all_supplied_metrics_match_population_sd': all(r['supplied_matches_ddof0'] for r in aggregates),
              'independent_accuracy_and_macro_f1_recount_passed': True,
              'interval_note': 'Wilson intervals are arithmetic descriptions conditional on legacy arrays. They do not establish isolation, model identity, independence of grouped predictions, or experiment validity. No inferential model-ranking tests performed.',
              'sessions': sessions, 'aggregates': aggregates, 'paired_contrasts': contrasts}
    (BASE / 'descriptive_metrics.json').write_text(json.dumps(output, indent=2) + '\n')
    for name, rows in [('cases.csv', cases), ('paired_contrasts.csv', contrasts)]:
        with (BASE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'counts': output['counts'], 'all_supplied_metrics_match_population_sd': output['all_supplied_metrics_match_population_sd'], 'aggregates': aggregates, 'paired_contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
