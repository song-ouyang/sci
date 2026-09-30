"""Fixed gate/retry/vote edge cases; all synthetic and offline."""
import unittest
from fractions import Fraction
from src.analysis.canary_controls_v2 import collapse_pass, known_answer_pass, make_windows, decide

def call(slot, label, known=False, valid=True):
    return {"group": str(slot), "arm": "C", "block": 1, "slot": slot,
            "focal": ["f"], "truth": {"f": "A"}, "predictions": {"f": label} if valid else None,
            "valid": valid, "known_pass": known}

class CanaryControlsTests(unittest.TestCase):
    def test_collapse_boundary_and_renaming(self):
        self.assertTrue(collapse_pass(["A"]*24+["B"]*3))
        self.assertFalse(collapse_pass(["A"]*25+["B"]*2))
        self.assertEqual(collapse_pass(["X"]*25+["Y"]*2), collapse_pass(["A"]*25+["B"]*2))
        self.assertTrue(collapse_pass(["A"]*26+["B"], Fraction(1)))
        self.assertFalse(collapse_pass(["A"]*27, Fraction(1)))

    def test_known_answer_eight_passes_seven_fails(self):
        truth = {str(i): "A" for i in range(9)}
        pred = {**truth, "0": "B"}
        self.assertTrue(known_answer_pass(pred, truth, list(truth)))
        pred["1"] = "B"
        self.assertFalse(known_answer_pass(pred, truth, list(truth)))
        self.assertFalse(known_answer_pass(None, truth, list(truth)))

    def test_numeric_schedule_order(self):
        seq = make_windows([call(10,"A"),call(2,"B"),call(1,"C")])[("C",1)]
        self.assertEqual([r["slot"] for r in seq], [1,2,10])

    def test_first_passing_not_best_outcome(self):
        seq = [call(1,"B"),call(2,"C",True),call(3,"A",True)]
        out, required, selected = decide(seq,"Known_answer")
        self.assertEqual((out["f"],len(required),selected),("C",2,"2"))

    def test_all_fail_retains_full_retry_budget(self):
        out, required, selected = decide([call(i,"A") for i in range(1,4)],"Known_answer")
        self.assertEqual((out["f"],len(required),selected),(None,3,""))

    def test_vote_majority_and_tie_modes(self):
        seq = [call(1,"B"),call(2,"A"),call(3,"A")]
        self.assertEqual(decide(seq,"Vote_first_tie")[0]["f"],"A")
        seq[-1] = call(3,"C")
        self.assertEqual(decide(seq,"Vote_first_tie")[0]["f"],"B")
        self.assertIsNone(decide(seq,"Vote_abstain_tie")[0]["f"])

    def test_invalid_response_cannot_pass_or_vote(self):
        seq = [call(1,"A",True,False),call(2,"A",True),call(3,"A",True)]
        self.assertIsNone(decide(seq,"Single")[0]["f"])
        self.assertIsNone(decide(seq,"Vote_first_tie")[0]["f"])
        self.assertEqual(decide(seq,"Known_answer")[2],"2")

if __name__ == "__main__":
    unittest.main()
