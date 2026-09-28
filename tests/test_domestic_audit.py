import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.analysis import audit_domestic_extension as audit
from src.experiments import domestic_api as api
from src.experiments import run_domestic_extension as runner
from test_domestic_api import FakeOpener, stream


class DomesticAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "experiment"
        env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "DASHSCOPE_API_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(self.base)

    def complete(self, invalid=False):
        original = api.run_batch
        def fake_run(config, batch, *args, **kwargs):
            values = [{"id": sid, "predicted": batch["allowed_labels"][0]} for sid in batch["expected_ids"]]
            body = "[]" if invalid else json.dumps(values)
            return original(config, batch, *args, **kwargs, opener=FakeOpener(stream(body)))
        with patch.object(api, "run_batch", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
            runner.run(self.base, "all")

    def test_incomplete_plan_cannot_be_frozen(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            audit.freeze(self.base)
        self.assertFalse((self.base / "frozen_manifest.json").exists())

    def test_scoring_without_freeze_cannot_open_labels(self):
        with self.assertRaises(FileNotFoundError):
            audit.score(self.base)
        self.assertFalse((self.base / "analysis").exists())

    def test_all_valid_freeze_counts_and_idempotence(self):
        self.complete()
        _, _, manifest = audit.freeze(self.base)
        self.assertEqual(manifest["valid_sessions"], 18)
        self.assertEqual(manifest["valid_predictions"], 1200)
        self.assertFalse(manifest["answer_keys_read"])
        before = (self.base / "frozen_manifest.json").read_bytes()
        audit.freeze(self.base)
        self.assertEqual(before, (self.base / "frozen_manifest.json").read_bytes())

    def test_invalid_batches_stay_unscored(self):
        self.complete(invalid=True)
        _, outcomes, manifest = audit.freeze(self.base)
        self.assertEqual(manifest["valid_sessions"], 0)
        self.assertEqual(manifest["valid_predictions"], 0)
        self.assertTrue(all(o["predictions"] is None for o in outcomes))
        self.assertTrue(all(o["usage"]["total_tokens"] == 42 for o in outcomes))

    def test_truncated_stream_still_records_usage(self):
        path = self.base / "truncated.sse"
        path.write_bytes(stream(finish="length"))
        result = audit.stream_metadata(path)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 42)

    def test_raw_response_tampering_is_detected(self):
        self.complete()
        raw = next(self.base.glob("sessions/*/*/*/response.sse"))
        raw.write_bytes(raw.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            audit.freeze(self.base)

    def test_source_snapshot_is_part_of_freeze(self):
        self.complete()
        audit.freeze(self.base)
        source = next(self.base.glob("source_snapshot/src/experiments/*.py"))
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit.verify(self.base)

    def test_scoring_known_synthetic_labels(self):
        from src.analysis import score_revision_v3 as original
        self.complete()
        audit.freeze(self.base)
        datasets, baselines = {}, {}
        for dataset in ("cmapss", "tep"):
            batch = audit.read_json(self.base / "batches" / (dataset + "_clean_r1.json"))
            labels = batch["allowed_labels"]
            datasets[dataset] = [{"id": sid, "label": labels[i % 2]}
                                 for i, sid in enumerate(batch["expected_ids"])]
            baselines[dataset] = {method: [labels[0]] * len(batch["expected_ids"])
                                  for method in original.BASELINES}
        with patch.object(original, "load_evaluation_data", return_value=(datasets, baselines, {})), \
                contextlib.redirect_stdout(io.StringIO()):
            report = audit.score(self.base)
        self.assertEqual(len(report["metrics"]), 18)
        for row in report["metrics"]:
            self.assertEqual(row["accuracy"], 0.5)
            self.assertAlmostEqual(row["macro_f1"], (2 / 3) / len(original.LABELS[row["dataset"]]))
            low, high = row["macro_f1_group_bootstrap95"]
            self.assertLess(low, row["macro_f1"])
            self.assertGreater(high, row["macro_f1"])
        self.assertTrue(all(p["accuracy_difference_pp"] == 0 for p in report["paired"]))
        self.assertTrue(all(p["accuracy_difference_pp_bootstrap95"] == [0, 0] for p in report["paired"]))
        self.assertTrue(all(v["accuracy"]["range"] == 0 for v in report["variability"]))
        self.assertEqual(len((self.base / "analysis/cases.csv").read_text().splitlines()), 1201)


if __name__ == "__main__":
    unittest.main()
