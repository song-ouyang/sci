"""Regression checks for the actual failure modes found in the CC audit."""
import unittest
import numpy as np

from src.analysis.statistics_v2 import abba_design_masks, call_randomization, matched_block_ci, pigeonhole_ci
from src.analysis.error_structure_v3 import chronological, group_summary, hydraulic_calls
from src.analysis.decision_policies_v3 import evaluate, prospective_calls, windows


class StatisticalCorrectionTests(unittest.TestCase):
    def test_dispatch_order_ignores_numeric_names_and_timezone_lexical_order(self):
        calls = [
            {"name": "r5", "order": 10, "started_at": "2026-09-29T01:00:00+00:00"},
            {"name": "r1", "order": 2, "started_at": "2026-09-29T08:59:00+08:00"},
            {"name": "r1_duplicate", "order": 3, "started_at": "2026-09-29T00:58:00+00:00"},
        ]
        self.assertEqual([c["name"] for c in chronological(calls)], ["r1_duplicate", "r1", "r5"])
        with self.assertRaises(ValueError):
            chronological([{"name": "missing_time", "order": 1}])
        with self.assertRaises(ValueError):
            chronological([{"name": "no_timezone", "started_at": "2026-09-29T01:00:00"}])

    def test_actual_hydraulic_duplicate_precedes_nominal_original(self):
        observed = windows(hydraulic_calls())
        front = observed[("Hydraulic (5 blocks)", "a_front")]
        self.assertEqual(front[0]["name"], "a_front_repeat_r1")
        totals = {policy: {k: sum(evaluate(w)[policy][k] for w in observed.values())
                           for k in ("correct", "error", "abstain", "calls")}
                  for policy in ("single", "canary")}
        self.assertEqual(totals["single"], {"correct": 49, "error": 23, "abstain": 0, "calls": 4})
        self.assertEqual(totals["canary"], {"correct": 66, "error": 6, "abstain": 0, "calls": 7})

    def test_abba_allocation_space_matches_two_of_each_sequence(self):
        masks = list(abba_design_masks())
        self.assertEqual(len(masks), 6)
        self.assertEqual(len({tuple(m) for m in masks}), 6)
        for mask in masks:
            blocks = mask.reshape(4, 4)
            self.assertEqual(sum(tuple(b) == (True, False, False, True) for b in blocks), 2)
            self.assertTrue(all(tuple(b) in [(True, False, False, True), (False, True, True, False)] for b in blocks))
        counts = [21, 19, 21, 21, 21, 21, 21, 19, 21, 21, 21, 21, 19, 21, 21, 15]
        correctness = np.array([[i < c for c in counts] for i in range(21)], dtype=int)
        # Deliberately construct labels with known counts; p_acc only depends on
        # the fixed call totals, not this arbitrary assignment of errors to cases.
        p_acc, _, n, p_min = call_randomization(correctness, correctness, masks[0], masks=masks)
        self.assertEqual(n, 6)
        self.assertAlmostEqual(p_acc, 4 / 6)
        self.assertAlmostEqual(p_min, 2 / 6)

    def test_matched_block_common_shock_does_not_create_a_difference(self):
        left = np.tile([0., 1., 0., 1.], (9, 1))
        right = left.copy()
        self.assertEqual(matched_block_ci(left, right, [(1, 1)] * 4, np.random.default_rng(4), 1000), (0., 0.))
        lo, hi = pigeonhole_ci(left, right, np.random.default_rng(4), 1000)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_always_wrong_varying_labels_can_be_rejected_by_unanimity(self):
        calls = [{"study": "toy", "service": "toy", "group": "toy",
                  "records": [("same", "a", "b", True), ("vary", "a", label, True)]}
                 for label in ["b", "c", "d"]]
        row = group_summary(calls)[0]
        self.assertEqual(row["stable_error"], 2)
        self.assertEqual(row["stable_error_same_label"], 1)
        self.assertEqual(row["stable_error_varying_labels"], 1)
        self.assertEqual(row["unanimity_coverage"], .5)
        self.assertEqual(row["plurality_wrong"], 1)
        self.assertEqual(row["plurality_withheld_ties"], 1)
        self.assertEqual(row["plurality_correct"], 0)

    def test_prospective_denominators_and_frozen_tie_first_rule_remain(self):
        ws = list(windows(prospective_calls()).values())
        self.assertEqual(len(ws), 9)
        self.assertTrue(all(len(w) == 3 for w in ws))
        for policy, expected in {
            "single": (105, 57, 0, 9), "canary": (123, 3, 36, 17),
            "vote": (97, 59, 6, 27), "vote_tie_first": (102, 60, 0, 27)
        }.items():
            actual = tuple(sum(evaluate(w)[policy][k] for w in ws) for k in ("correct", "error", "abstain", "calls"))
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
