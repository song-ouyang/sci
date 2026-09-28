import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'src/analysis/audit_conversation_exploratory.py'
spec = importlib.util.spec_from_file_location('conversation_audit', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ConversationAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.success = self.base / 'predictions.json'
        self.failure = self.base / 'FAILED.json'
        self.group = {'samples': [{'id': '00000001'}, {'id': '00000002'}], 'valid_labels': ['A', 'B']}

    def test_off_task_is_failure_not_unrun(self):
        self.failure.write_text(json.dumps({'reason': 'off_task'}))
        state, reason, rows = m.outcome(self.success, [self.failure], self.group)
        self.assertEqual(state, 'failed_off_task'); self.assertIsNone(rows)

    def test_partial_array_never_scored(self):
        self.failure.write_text(json.dumps([{'id': '00000001', 'predicted': 'A'}]))
        state, reason, rows = m.outcome(self.success, [self.failure], self.group)
        self.assertEqual(state, 'failed_incomplete'); self.assertIsNone(rows)

    def test_order_violation_remains_failure(self):
        self.success.write_text(json.dumps([{'id': '00000002', 'predicted': 'A'}, {'id': '00000001', 'predicted': 'B'}]))
        self.assertEqual(m.outcome(self.success, [], self.group)[0], 'failed_invalid_array')

    def test_conflicting_outcomes_rejected(self):
        self.success.write_text('[]'); self.failure.write_text('{}')
        with self.assertRaises(ValueError): m.outcome(self.success, [self.failure], self.group)

    def test_only_complete_pairs_contribute(self):
        rows = [{'status': 'pair_unavailable', 'repeat': 2, 'clean_accuracy_pct': 99}]
        for repeat, clean, mis in [(1, 60, 10), (3, 50, 10)]:
            rows.append({'status': 'paired_arrays_available', 'repeat': repeat, 'clean_accuracy_pct': clean,
                         'misnamed_accuracy_pct': mis, 'accuracy_delta_misnamed_minus_clean_pp': mis-clean,
                         'macro_f1_delta_misnamed_minus_clean_pp': -40, 'flips': 5, 'n_paired': 10})
        summary = m.paired_summary(rows)
        self.assertEqual(summary['paired_repeats'], [1, 3])
        self.assertEqual(summary['clean_accuracy_mean_pct'], 55)
        self.assertEqual(summary['accuracy_delta_misnamed_minus_clean_pp'], -45)

    def test_singleton_sd_is_missing_not_zero(self):
        self.assertIsNone(m.summarize([100])['sample_sd_pp'])
        self.assertEqual(m.summarize([100, 100])['sample_sd_pp'], 0)


if __name__ == '__main__': unittest.main()
