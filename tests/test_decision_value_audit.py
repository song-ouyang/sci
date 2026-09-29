"""Synthetic checks for the new offline audit; no source-result modifications."""
import unittest
from src.analysis.audit_decision_value import intersection, policy_actions, summarize, stratified_cluster_totals, POLICIES


class DecisionValueTests(unittest.TestCase):
    def test_gate_agreement_preserves_wrong_base(self):
        actions = policy_actions("wrong", "wrong", "true", "true")
        self.assertEqual(actions["Same"]["action"], "accepted_error")
        self.assertEqual(actions["View"]["action"], "rejected")
        self.assertEqual(actions["View"]["output"], "")

    def test_partition_and_loss_endpoints(self):
        rows = [{"action": a, "base_correct": correct} for a, correct in
                [("accepted_correct", True), ("accepted_error", False), ("rejected", True), ("rejected", False)]]
        s = summarize(rows)
        self.assertEqual((s["N"], s["accepted_error"], s["rejected"]), (4, 1, 2))
        self.assertEqual((s["coverage"], s["correct_yield"], s["selective_error"]), (.5, .25, .5))
        self.assertEqual(s["loss_intercept"] + s["loss_slope"], .75)

    def test_empty_acceptance_risk_is_undefined(self):
        s = summarize([{"action": "rejected", "base_correct": False}])
        self.assertIsNone(s["selective_error"])
        self.assertEqual((s["loss_intercept"], s["loss_slope"]), (0, 1))

    def test_intersection_sign_and_outside_domain(self):
        a = {"N": 54, "accepted_error": 4, "rejected": 17}
        b = {"N": 54, "accepted_error": 5, "rejected": 10}
        x = intersection(a, b)
        self.assertEqual(x["c_exact"], "1/7")
        self.assertTrue(x["left_lower_below"])
        self.assertFalse(x["left_lower_above"])
        x = intersection({"N": 72, "accepted_error": 58, "rejected": 6},
                         {"N": 72, "accepted_error": 21, "rejected": 42})
        self.assertEqual(x["c_exact"], "37/36")
        self.assertFalse(x["in_unit_interval"])

    def test_normalized_intersection_handles_different_denominators(self):
        x = intersection({"N": 2, "accepted_error": 1, "rejected": 0},
                         {"N": 4, "accepted_error": 0, "rejected": 4})
        self.assertEqual(x["c_exact"], "1/2")

    def test_parallel_and_coincident(self):
        a = {"N": 2, "accepted_error": 1, "rejected": 1}
        self.assertEqual(intersection(a, a)["kind"], "coincident")
        self.assertEqual(intersection(a, {**a, "accepted_error": 0})["kind"], "parallel")

    def test_cluster_resampling_preserves_rounds_and_pairing(self):
        rows = []
        for unit, label, count in [("u1", "a", 8), ("u2", "b", 8)]:
            for _ in range(count):
                for p in POLICIES:
                    rows.append({"source_unit": unit, "truth": label, "policy": p,
                                 "action": "accepted_error" if p != "Reject_all" else "rejected", "base_correct": False})
        a, strata = stratified_cluster_totals(rows, replicates=17, seed=1)
        self.assertEqual(strata, {"a": 1, "b": 1})
        self.assertTrue((a[:, :, 0] == 16).all())
        self.assertTrue((a[:, 1] == a[:, 2]).all())
        self.assertTrue((a[:, 3, 2] == 16).all())


if __name__ == "__main__":
    unittest.main()
