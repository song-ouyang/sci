"""Synthetic-data design/scoring invariants; no downloads or model calls."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.prepare_hydraulic_data import profile_audit
from src.analysis import score_hydraulic_v1 as scoring
from src.experiments import prepare_hydraulic_v1 as design
from src.experiments import run_hydraulic_v1 as runner
from src.experiments.interface_contract import canonicalize


def synthetic_profile():
    rows = []
    for cooler in (3, 20, 100):
        for pump in (0, 1, 2):
            for valve in (73, 80, 90, 100):
                for accumulator in (90, 100, 115, 130):
                    for position in range(7):
                        rows.append([cooler, valve, pump, accumulator, int(position == 0)])
    rows += [rows[-1][:4] + [1] for _ in range(2205 - len(rows))]
    return np.asarray(rows, dtype=int)


class HydraulicDesignTests(unittest.TestCase):
    def setUp(self):
        self.profile = synthetic_profile()
        self.blocks = profile_audit(self.profile)

    def record(self):
        return {'id': 'abc123ef', 'readings': [dict(r, value=f'{i + 0.125:.6f}')
                for i, r in enumerate(design.registry())]}

    def prepared(self):
        tmp = tempfile.TemporaryDirectory(prefix='hydraulic-design-test-', dir=design.ROOT / 'tmp')
        self.addCleanup(tmp.cleanup)
        parent = Path(tmp.name)
        data, base = parent / 'data', parent / 'study'
        data.mkdir()
        (data / 'profile.json').write_text(json.dumps({'rows': self.profile.tolist()}))
        (data / 'blocks.json').write_text(json.dumps(self.blocks))
        (data / 'metadata.json').write_text(json.dumps({'fixture': 'synthetic_no_measurements'}))
        x = np.arange(2205 * 28, dtype=float).reshape(2205, 28) / 7
        np.savez(data / 'measurement_summary.npz', X=x, channel_names=np.asarray(design.CHANNELS))
        with patch.object(design, 'DATA', data), patch.object(design, 'BASE', base), \
             patch.object(design.subprocess, 'check_output', return_value='codex-test'), \
             patch('sys.stdout', new=io.StringIO()):
            design.prepare()
        return base, x

    def test_selection_balance_stability_and_block_permutation(self):
        blocks = self.blocks['joint_condition_blocks']
        panels, _ = design.select_blocks(blocks, self.profile)
        shuffled = copy.deepcopy(blocks)
        random.Random(19).shuffle(shuffled)
        reordered, _ = design.select_blocks(shuffled, self.profile)
        self.assertEqual(panels, reordered)
        selected = [r for rows in panels.values() for r in rows]
        self.assertEqual(len({r['block_id'] for r in selected}), 54)
        self.assertEqual(len({r['cycle_1based'] for r in selected}), 54)
        self.assertEqual(len({r['id'] for r in selected}), 54)
        for rows in panels.values():
            self.assertEqual(len(rows), 18)
            self.assertEqual(Counter((r['profile'][2], r['profile'][0]) for r in rows),
                             {(pump, cooler): 2 for pump in (0, 1, 2) for cooler in (3, 20, 100)})
            self.assertEqual(Counter(r['true_label'] for r in rows), dict.fromkeys(design.LABELS, 6))
            self.assertEqual(len({(r['profile'][1], r['profile'][3]) for r in rows}), 2)
            for row in rows:
                self.assertEqual(row['profile'][4], 0)
                self.assertGreaterEqual(row['cycle_1based'], row['block_start'])
                self.assertLessEqual(row['cycle_1based'], row['block_end'])

    def test_insufficient_shared_nuisance_settings_rejected(self):
        blocks = [b for b in self.blocks['joint_condition_blocks'] if b['condition_values'][1] == 100]
        with self.assertRaisesRegex(ValueError, 'Fewer than six'):
            design.select_blocks(blocks, self.profile)

    def test_correct_pairs_reorder_without_value_or_identity_change(self):
        clean = self.record()
        reordered = design.perturb(clean, 'pair_reverse')
        self.assertNotEqual(reordered, clean)
        self.assertEqual({(r['channel_id'], r['descriptor'], r['value']) for r in reordered['readings']},
                         {(r['channel_id'], r['descriptor'], r['value']) for r in clean['readings']})
        self.assertEqual(canonicalize(reordered, design.registry()), clean)
        self.assertEqual(clean, self.record())

    def test_visible_corruptions_preserve_numeric_positions_and_are_rejected(self):
        clean = self.record()
        for mode in ('descriptor_rotation', 'same_unit_swap'):
            changed = design.perturb(clean, mode)
            self.assertEqual([r['value'] for r in changed['readings']], [r['value'] for r in clean['readings']])
            self.assertEqual([r['channel_id'] for r in changed['readings']], [r['channel_id'] for r in clean['readings']])
            self.assertNotEqual(design.prompt_sample(clean), design.prompt_sample(changed))
            with self.assertRaisesRegex(ValueError, 'Descriptor differs'):
                canonicalize(changed, design.registry())
        rotated = design.perturb(clean, 'descriptor_rotation')
        self.assertTrue(all(a['descriptor'] != b['descriptor'] for a, b in zip(clean['readings'], rotated['readings'])))

    def test_internally_consistent_value_swap_is_an_explicit_blind_spot(self):
        clean = self.record()
        changed = design.perturb(clean, 'hidden_value_swap')
        accepted = canonicalize(changed, design.registry())
        self.assertNotEqual(accepted, clean)
        self.assertEqual([r['descriptor'] for r in changed['readings']], [r['descriptor'] for r in clean['readings']])
        self.assertEqual(sorted(r['value'] for r in changed['readings']), sorted(r['value'] for r in clean['readings']))

    def test_prepared_schedule_payload_identity_and_no_private_fields(self):
        base, x = self.prepared()
        p, groups, blocks = runner.validate(base)
        self.assertEqual(len(groups), 63)
        self.assertEqual(sum(len(g['samples']) for g in groups.values()), 1278)
        self.assertEqual([len(block) for block in blocks], [13, 13, 13, 12, 12])
        self.assertEqual(p['calls_scheduled'], 63)
        self.assertEqual(p['slots_scheduled'], 1278)
        answers = json.loads((base / 'answers/records.json').read_text())['panels']
        truth = {r['id']: r for panel in answers.values() for r in panel}
        singles = [g['samples'][0]['id'] for g in groups.values() if g['condition'] == 'singleton']
        self.assertEqual(Counter(singles), Counter(r['id'] for r in answers['focal']))
        for g in groups.values():
            for sample in g['samples']:
                self.assertEqual(set(sample), {'id', 'prompt'})
                self.assertNotIn('channel_id', sample['prompt'])
                self.assertNotIn('true_label', sample['prompt'])
                self.assertNotIn('cycle_1based', sample['prompt'])
                self.assertNotIn('stable_flag', sample['prompt'])
                self.assertEqual(len(sample['prompt'].split('; ')), 28)
                if g['condition'] == 'clean':
                    expected = [f'{v:.6f}' for v in x[truth[sample['id']]['cycle_1based'] - 1]]
                    self.assertEqual([field.rsplit('=', 1)[1] for field in sample['prompt'].split('; ')], expected)
        for repeat in range(1, 6):
            def payload(condition):
                return (base / f'inference/prompts/{condition}_r{repeat}.txt').read_bytes()
            self.assertEqual(payload('clean'), payload('clean_repeat'))
            self.assertEqual(payload('a_front'), payload('a_front_repeat'))
            for front, back in (('a_front', 'a_back'), ('b_front', 'b_back')):
                first, last = groups[f'{front}_r{repeat}']['samples'], groups[f'{back}_r{repeat}']['samples']
                self.assertEqual(len(first), 36)
                self.assertEqual(last, first[18:] + first[:18])

    def test_prepare_cannot_overwrite_an_existing_study(self):
        base, _ = self.prepared()
        with patch.object(design, 'BASE', base), self.assertRaisesRegex(ValueError, 'overwrite'):
            design.prepare()

    def test_invalid_metric_preserves_scheduled_denominator(self):
        truth = {'a': {'true_label': 'No_leakage'}, 'b': {'true_label': 'Weak_leakage'}}
        result = scoring.metric(truth, None, ['a', 'b'])
        self.assertEqual(result, {'n': 2, 'valid': 0, 'correct': None, 'accuracy': None,
                                  'macro_f1': None, 'usable_correct': 0})
        pair = scoring.pair('test', 'left', 'right', ['a', 'b'], truth,
                            {'left': {'a': 'No_leakage', 'b': 'Weak_leakage'}, 'right': None})
        self.assertEqual(pair['n_scheduled'], 2)
        self.assertEqual(pair['n'], 0)
        self.assertIsNone(pair['flips'])

    def test_scorer_fails_before_answer_access_when_output_freeze_is_absent(self):
        touched = []
        original = Path.read_text
        def track(path, *args, **kwargs):
            touched.append(str(path))
            return original(path, *args, **kwargs)
        with patch.object(scoring, 'audit_outputs', side_effect=ValueError('Missing frozen output')) as audit, \
             patch.object(Path, 'read_text', track), self.assertRaisesRegex(ValueError, 'Missing frozen'):
            scoring.score(Path('/nonexistent/synthetic-study'))
        audit.assert_called_once_with(Path('/nonexistent/synthetic-study'), require_frozen=True)
        self.assertFalse(any('answers' in path for path in touched))

    def test_scoring_all_missing_retains_planned_slots_and_unknown_usage(self):
        base, _ = self.prepared()
        p, groups, blocks = runner.validate(base)
        records = [{'group': name, 'condition': g['condition'], 'repeat': g['repeat'],
                    'status': 'not_attempted', 'n': len(g['samples']), 'usage': None,
                    'elapsed_seconds': None, 'predictions': None} for name, g in groups.items()]
        # An attempted execution with unavailable token accounting must remain
        # reportable. It is not a real inference output or scientific result.
        records[0].update(status='failed', usage={})
        by_name = {r['group']: r for r in records}
        (base / 'predictions').mkdir()
        (base / 'predictions/frozen_manifest.json').write_text('{}')
        with patch.object(scoring, 'audit_outputs', return_value=(p, groups, records, {})), \
             patch.object(scoring, 'group_audit', side_effect=lambda b, p, name, group: by_name[name]), \
             patch('sys.stdout', new=io.StringIO()):
            result = scoring.score(base)
        self.assertEqual(result['totals']['requests_scheduled'], 63)
        self.assertEqual(result['totals']['slots_scheduled'], 1278)
        self.assertEqual(result['totals']['slots_valid'], 0)
        self.assertEqual(result['totals']['usable_correct_slots'], 0)
        self.assertEqual(result['totals']['tokens_observed'], 0)
        self.assertEqual(result['totals']['usage_unknown_requests'], 1)
        self.assertTrue(all(p['n'] == 0 and p['flips'] is None for p in result['paired']))
        self.assertIsNone(result['singleton']['accuracy'])


if __name__ == '__main__':
    unittest.main()
