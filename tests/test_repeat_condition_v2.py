"""Meaningful synthetic invariants for frozen-output, repeat/condition audits."""
import json
from pathlib import Path
import tempfile
import unittest

from src.analysis import audit_repeat_condition_v2 as audit


def condition(predictions):
    return {'predictions': predictions}


class RepeatConditionTests(unittest.TestCase):
    def test_stable_same_different_and_either_side_unstable_are_exhaustive(self):
        ids = ['different', 'same', 'left_changes', 'right_changes']
        left = {r: dict(zip(ids, ['A', 'A', 'A' if r < 3 else 'B', 'A'])) for r in [1, 2, 3]}
        right = {r: dict(zip(ids, ['B', 'A', 'B', 'B' if r < 3 else 'A'])) for r in [1, 2, 3]}
        rows = [audit.classify_case(left, right, [1, 2, 3], i) for i in ids]
        self.assertEqual([r['category'] for r in rows],
                         [audit.CATEGORIES[0], audit.CATEGORIES[1], audit.CATEGORIES[2], audit.CATEGORIES[2]])
        self.assertEqual(rows[0]['left_predictions'], ['A'] * 3)
        self.assertEqual(rows[0]['right_predictions'], ['B'] * 3)

    def test_missing_planned_repeat_is_not_replaced_or_called_stable(self):
        left = {1: {'x': 'A'}, 2: {'x': 'A'}, 3: {'x': 'A'}}
        right = {1: {'x': 'B'}, 2: {'x': 'B'}, 3: None}
        row = audit.classify_case(left, right, [1, 2, 3], 'x')
        self.assertEqual(row['category'], 'incomplete_planned_repeats')
        self.assertTrue(row['left_stable'])
        self.assertIsNone(row['right_stable'])
        result = audit.analyze_contrast('synthetic', 'model', 'a__b', 'mapping', 'a', 'b',
                                        ['x'], [1, 2, 3], condition(left), condition(right))
        self.assertEqual(result[0]['cross_pairs_planned'], 9)
        self.assertEqual(result[0]['cross_pairs_available'], 6)
        self.assertEqual(result[0]['n_cases'], 1)
        self.assertEqual(sum(not r['available'] for r in result[2]), 3)
        self.assertTrue(all(r['n_paired_cases'] == 0 and r['flips'] is None
                            for r in result[2] if not r['available']))

    def test_all_cross_repeats_are_enumerated_without_inflating_n(self):
        predictions = {1: {'x': 'A'}, 2: {'x': 'B'}, 3: {'x': 'C'}}
        result = audit.analyze_contrast('synthetic', 'model', 'a__b', 'context', 'a', 'b',
                                        ['x'], [1, 2, 3], condition(predictions), condition(predictions))
        self.assertEqual(len(result[1]), 6)
        self.assertEqual(len(result[2]), 9)
        self.assertEqual(len(result[3]), 1)
        self.assertEqual(sum(r['flips'] for r in result[2]), 6)
        self.assertTrue(all(r['flips'] == 0 for r in result[2] if r['same_repeat_index']))
        self.assertEqual(result[0]['any_side_unstable'], 1)
        self.assertEqual(result[0]['observed_repeat_stable_different'], 0)

    def test_pairing_is_by_id_not_dictionary_order(self):
        self.assertEqual(audit.compare_pair({'a': 'X', 'b': 'Y'}, {'b': 'Y', 'a': 'X'}, ['a', 'b'])['flips'], 0)
        with self.assertRaisesRegex(ValueError, 'ID mismatch'):
            audit.compare_pair({'a': 'X'}, {'c': 'X'}, ['a'])

    def fixture(self, different_request_bytes=False):
        temporary = tempfile.TemporaryDirectory(prefix='repeat-condition-test-', dir=audit.ROOT / 'tmp')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        groups = {}
        prompt_hashes = {}
        def write(path, value, compact=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, separators=(',', ':') if compact else None))
        for r in [1, 2, 3]:
            name = f'clean_r{r}'
            groups[name] = {'dataset': 'tep', 'condition': 'clean', 'repeat': r,
                            'samples': [{'id': 'abc12345', 'prompt': 'p=1'}], 'valid_labels': ['A', 'B']}
            path = base / 'inference/prompts' / f'{name}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Identical frozen prompt.')
            prompt_hashes[name] = audit.digest(path)
        write(base / 'inference/batches.json', {'groups': groups})
        protocol = {'reasoning_effort': 'medium', 'input_sha256': audit.digest(base / 'inference/batches.json'),
                    'code_sha256': {}, 'prompt_sha256': prompt_hashes}
        write(base / 'protocol.json', protocol)
        ph = audit.digest(base / 'protocol.json')
        (base / 'protocol.sha256').write_text(ph)
        for r in [1, 2, 3]:
            name = f'clean_r{r}'
            folder = base / 'predictions/model'
            request = {'model': 'model', 'reasoning_effort': 'medium', 'developer_instructions': 'fixture',
                       'user_message': 'Identical frozen prompt.'}
            write(folder / f'{name}_request.json', request, compact=different_request_bytes and r == 3)
            result = [{'id': 'abc12345', 'predicted': 'A'}]
            write(folder / f'{name}_results.json', result)
            write(folder / f'{name}_attempt1_response.txt', result)
            write(folder / f'{name}_metadata.json', {'model_requested': 'model', 'batch': name,
                  'input_sha256': protocol['input_sha256'], 'protocol_sha256': ph, 'status': 'complete',
                  'prediction_sha256': audit.digest(folder / f'{name}_results.json'),
                  'attempts': [{'attempt': 1, 'response_file': f'{name}_attempt1_response.txt'}]})
        manifest = {'protocol_sha256': ph, 'files': {str(p.relative_to(base / 'predictions')): audit.digest(p)
                     for p in (base / 'predictions').rglob('*') if p.is_file()}}
        write(base / 'predictions/frozen_manifest.json', manifest)
        return base

    def test_frozen_hash_changes_are_rejected_before_analysis(self):
        base = self.fixture()
        audit.verify_study(base)
        path = base / 'predictions/model/clean_r1_results.json'
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Frozen output changed'):
            audit.verify_study(base)

    def test_same_condition_checks_exact_recorded_request_bytes(self):
        base = self.fixture(different_request_bytes=True)
        p, groups, manifest, _ = audit.verify_study(base)
        with self.assertRaisesRegex(ValueError, 'Within-condition requests differ'):
            audit.load_condition(base, p, groups, manifest, 'model', 'clean', [1, 2, 3], ['abc12345'])

    def test_raw_predictions_and_saved_results_must_agree(self):
        base = self.fixture()
        raw = base / 'predictions/model/clean_r1_attempt1_response.txt'
        raw.write_text(json.dumps([{'id': 'abc12345', 'predicted': 'B'}]))
        manifest_path = base / 'predictions/frozen_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['model/clean_r1_attempt1_response.txt'] = audit.digest(raw)
        manifest_path.write_text(json.dumps(manifest))
        p, groups, manifest, _ = audit.verify_study(base)
        with self.assertRaisesRegex(ValueError, 'Raw response and parsed predictions disagree'):
            audit.load_condition(base, p, groups, manifest, 'model', 'clean', [1, 2, 3], ['abc12345'])


if __name__ == '__main__':
    unittest.main()
