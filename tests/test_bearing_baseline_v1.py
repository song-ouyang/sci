"""Synthetic offline invariants; no actual study data, models, or LLM calls."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis import bearing_baseline_v1 as b


def documents():
    dictionary = [{"name": f"feature_{i}", "unit": "synthetic"} for i in range(28)]
    result = []
    serial = 0
    for role, bearings, repetitions in (("development", 2, 5), ("test", 3, 1)):
        rows = []
        for label_index, label in enumerate(b.LABELS):
            for group in range(bearings):
                bearing = f"{role}_{label_index}_{group}"
                for regime in b.REGIMES:
                    for repetition in range(1, repetitions + 1):
                        serial += 1
                        rows.append({"id": f"{serial:08x}", "label": label, "bearing": bearing, "regime": regime,
                            "repetition": repetition, "member": f"{bearing}/{regime}_{bearing}_{repetition}.mat",
                            "values": [f"{serial + i / 100:.6f}" for i in range(28)]})
        result.append({"schema": f"bearing-baseline-{role}-v1", "dictionary": copy.deepcopy(dictionary), "records": rows})
    return result


def prepared(folder):
    base = Path(folder)
    data, out = base / "data", base / "result"
    dev, test = documents()
    for name, doc in (("development.json", dev), ("test.json", test)):
        b.write(data / name, doc)
    b.write(data / "plan.json", {"code_sha256": {}})
    (data / "plan.sha256").write_text(b.digest(data / "plan.json") + "\n")
    b.write(data / "audit.json", {"plan_sha256": b.digest(data / "plan.json"), "code_sha256": {},
        "output_sha256": {name: b.digest(data / name) for name in ("development.json", "test.json")}})
    original, llm = base / "original.json", base / "llm.csv"
    b.write(original, {"dictionary": dev["dictionary"]})
    with llm.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "panel", "round", "condition", "repeat", "id", "focal", "truth", "predicted", "correct"])
        writer.writeheader()
        for call in range(12):
            for r in test["records"]:
                writer.writerow({"group": f"call{call}", "panel": "p01", "round": call // 6 + 1, "condition": "FR"[call % 2], "repeat": call % 3 + 1,
                    "id": r["id"], "focal": True, "truth": r["label"], "predicted": "healthy", "correct": r["label"] == "healthy"})
    b.prepare_protocol(data, out, original, llm)
    return data, out


class ConstantFake:
    """Records fitting inputs but performs no statistical training."""
    def __init__(self, seen=None):
        self.seen = seen

    def fit(self, x, y):
        if self.seen is not None:
            self.seen.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return self

    def predict(self, x):
        return np.repeat("healthy", len(x))


class BaselineTests(unittest.TestCase):
    def test_schema_dimensions_and_candidate_order(self):
        dev, test = documents()
        dr, tr = b.validate_split(dev, test, dev["dictionary"])
        self.assertEqual((len(dr), len(tr)), (120, 36))
        self.assertEqual(b.SVM_GRID, [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.1)])
        self.assertEqual(b.RF_GRID, [{"max_features": f, "min_samples_leaf": n} for f in ("sqrt", 1.0) for n in (1, 3)])

    def test_physical_source_overlap_rejected(self):
        dev, test = documents()
        original = test["records"][0]["bearing"]
        target = dev["records"][0]["bearing"]
        for r in test["records"]:
            if r["bearing"] == original:
                r["bearing"] = target
                r["member"] = f'{target}/{r["regime"]}_{target}_{r["repetition"]}.mat'
        with self.assertRaisesRegex(ValueError, "leakage"):
            b.validate_split(dev, test, dev["dictionary"])

    def test_duplicate_member_and_wrong_column_order_rejected(self):
        dev, test = documents()
        test["dictionary"].reverse()
        with self.assertRaisesRegex(ValueError, "dictionary/order"):
            b.validate_split(dev, test, dev["dictionary"])
        dev, test = documents()
        test["records"][1] = {**test["records"][0], "id": "ffffffff"}
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            b.validate_split(dev, test, dev["dictionary"])

    def test_eight_fold_specimen_and_source_isolation(self):
        dev, _ = documents()
        rows = dev["records"]
        folds = b.grouped_folds(rows)
        self.assertEqual(len(folds), 8)
        self.assertEqual(len({tuple(f["validation_bearings"]) for f in folds}), 8)
        for f in folds:
            self.assertEqual((len(f["train"]), len(f["validation"])), (60, 60))
            self.assertFalse(set(f["train_bearings"]) & set(f["validation_bearings"]))
            self.assertFalse({rows[i]["member"] for i in f["train"]} & {rows[i]["member"] for i in f["validation"]})
            self.assertEqual({rows[i]["label"] for i in f["train"]}, set(b.LABELS))

    def test_loro_excludes_regime_from_every_tuning_training_fold(self):
        dev, test = documents()
        for _, regime, train, tst in b.tasks(dev["records"], test["records"])[1:]:
            self.assertEqual((len(train), len(tst)), (90, 9))
            self.assertEqual({r["regime"] for r in tst}, {regime})
            self.assertNotIn(regime, {r["regime"] for r in train})
            for f in b.grouped_folds(train):
                self.assertEqual((len(f["train"]), len(f["validation"])), (45, 45))
                self.assertNotIn(regime, {train[i]["regime"] for i in f["train"] + f["validation"]})

    def test_candidate_fit_receives_only_current_training_rows_and_first_tie(self):
        dev, _ = documents()
        seen = []
        selected, reports = b.choose_candidate(dev["records"], "SVM", b.SVM_GRID[:2], lambda *_: ConstantFake(seen))
        self.assertEqual(selected, 0)
        self.assertEqual(len(seen), 16)
        for (x, y), f in zip(seen[:8], b.grouped_folds(dev["records"])):
            np.testing.assert_array_equal(x, b.array([dev["records"][i] for i in f["train"]]))
            self.assertEqual(len(y), 60)
        self.assertEqual(len(reports[0]["folds"][0]["validation_predictions"]), 60)

    def test_estimator_pipeline_scaler_is_fold_local_and_fixed_configuration(self):
        # A fresh scaler is created with each fold estimator; no fitting is done.
        a, other = b.estimator("SVM", b.SVM_GRID[0]), b.estimator("SVM", b.SVM_GRID[0])
        self.assertIsNot(a.named_steps["scale"], other.named_steps["scale"])
        self.assertFalse(hasattr(a.named_steps["scale"], "mean_"))
        self.assertEqual(a.named_steps["classifier"].kernel, "rbf")
        forest = b.estimator("RF", b.RF_GRID[0])
        self.assertEqual((forest.n_estimators, forest.n_jobs, forest.random_state), (200, 1, 20260928))

    def test_test_labels_cannot_change_fit_features(self):
        _, test = documents()
        modified = copy.deepcopy(test)
        for r in modified["records"]:
            r["label"] = b.LABELS[(b.LABELS.index(r["label"]) + 1) % 3]
        original_inputs, original_answers = b.split_test(test)
        changed_inputs, changed_answers = b.split_test(modified)
        self.assertEqual(original_inputs, changed_inputs)
        self.assertNotEqual(original_answers, changed_answers)
        self.assertTrue(all("label" not in r for r in original_inputs["records"]))
        np.testing.assert_array_equal(b.array(original_inputs["records"]), b.array(changed_inputs["records"]))

    def test_fit_loader_does_not_parse_answer_original_test_or_llm(self):
        with tempfile.TemporaryDirectory() as folder:
            data, out = prepared(folder)
            real_read = b.read
            allowed = {(out / "protocol.json").resolve(), (out / "test_inputs.json").resolve(), (data / "development.json").resolve()}
            def guarded_read(path):
                self.assertIn(Path(path).resolve(), allowed)
                return real_read(path)
            with patch.object(b, "read", side_effect=guarded_read):
                _, dr, tr = b.load_fit_inputs(out)
            self.assertEqual((len(dr), len(tr)), (120, 36))
            source = b.read(data / "test.json")
            source["records"][0]["label"] = "inner_race_fault"
            b.write(data / "test.json", source)
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                b.load_fit_inputs(out)

    def test_structural_row_and_background_invariants(self):
        dev, test = documents()
        self.assertTrue(all(b.structural_checks(ConstantFake(), b.array(test["records"]), b.array(dev["records"])).values()))
        class BatchDependent(ConstantFake):
            def predict(self, x):
                return np.repeat("healthy" if len(x) == 1 else "outer_race_fault", len(x))
        with self.assertRaisesRegex(ValueError, "row-independence"):
            b.structural_checks(BatchDependent(), b.array(test["records"]), b.array(dev["records"]))

    def test_fake_fit_freeze_score_and_no_rerun_or_postfreeze_change(self):
        with tempfile.TemporaryDirectory() as folder:
            _, out = prepared(folder)
            original_choose = b.choose_candidate
            def fake_choose(rows, model, grid):
                return original_choose(rows, model, grid, lambda *_: ConstantFake())
            with patch.object(b, "estimator", side_effect=lambda *_: ConstantFake()), patch.object(b, "choose_candidate", side_effect=fake_choose):
                status = b.fit(out)
            self.assertEqual(status["prediction_rows"], 216)
            b.audit_predictions(out)
            with self.assertRaisesRegex(ValueError, "no rerun"):
                b.fit(out)
            b.score(out)
            result = b.read(out / "analysis/scores.json")
            self.assertEqual(len(result["baseline"]), 18)
            self.assertEqual(len(result["llm_calls"]), 12)
            self.assertEqual(result["baseline"]["primary/SVM"]["correct"], 12)
            self.assertAlmostEqual(result["baseline"]["primary/SVM"]["macro_f1"], 1 / 6)
            self.assertEqual(result["baseline"]["primary/SVM"]["per_class_recall"], {"healthy": 1.0, "outer_race_fault": 0.0, "inner_race_fault": 0.0})
            with (out / "analysis/llm_call_comparisons.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 72)
            with (out / "fit/predictions.csv").open("a") as handle:
                handle.write("changed\n")
            with self.assertRaisesRegex(ValueError, "Frozen prediction artifact changed"):
                b.score(out)


if __name__ == "__main__":
    unittest.main()
