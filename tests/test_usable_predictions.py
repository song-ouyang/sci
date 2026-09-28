"""Protect denominators and whole-batch exclusion in post hoc delivery reporting."""
from copy import deepcopy
import unittest
from src.analysis.summarize_usable_predictions import summarize


def outcomes():
    return [dict(model="fixture", dataset=d, condition=c, repeat=r, status="valid",
                 expected_n=n, n_scored=n, correct=n//2, accuracy=0.5,
                 confusion_matrix={"A": {"A": n//2, "B": n//2}})
            for d,c,n in [("cmapss","clean",60),("tep","clean",70),("tep","misnamed",70)]
            for r in (1,2,3)]


class UsablePredictionTests(unittest.TestCase):
    def test_failed_batch_retains_scheduled_denominator_without_imputing_accuracy(self):
        rows=outcomes()
        for x in rows[-3:-1]:
            x.update(status="invalid_response",n_scored=0,correct=None,accuracy=None,confusion_matrix=None)
        x=summarize(rows,"fixture")[-1]
        assert (x["usable_correct"],x["valid_slots"],x["scheduled_slots"]) == (35,70,210)
        assert x["conditional_accuracy"] == .5
        assert x["usable_correct_fraction"] == 1/6
        assert x["unique_source_units"] == 70


    def test_all_failed_is_zero_delivery_but_no_classification_estimate(self):
        rows=outcomes()
        for x in rows:
            x.update(status="invalid_response",n_scored=0,correct=None,accuracy=None,confusion_matrix=None)
        assert all(x["conditional_accuracy"] is None and x["usable_correct_fraction"]==0
                   for x in summarize(rows,"fixture"))


    def test_duplicate_repeat_cannot_replace_missing_outcome(self):
        rows=outcomes();rows[-1]=deepcopy(rows[-2])
        with self.assertRaisesRegex(ValueError,"scheduled outcome"):
            summarize(rows,"fixture")


    def test_partial_failure_cannot_contribute_labels(self):
        rows=outcomes();rows[-1]["status"]="invalid_response"
        with self.assertRaisesRegex(ValueError,"partial classification"):
            summarize(rows,"fixture")


    def test_corrupt_summary_is_rejected_using_confusion_counts(self):
        rows=outcomes();rows[-1]["correct"] += 1
        with self.assertRaisesRegex(ValueError,"Confusion"):
            summarize(rows,"fixture")


if __name__ == "__main__":
    unittest.main()
