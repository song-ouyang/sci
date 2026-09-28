"""Synthetic statistics and key-access barrier checks for the revision scorer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "src/analysis/score_revision_v3.py"
spec = importlib.util.spec_from_file_location("revision_scorer", MODULE)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
freeze_spec = importlib.util.spec_from_file_location("revision_freezer", MODULE.with_name("freeze_revision_v3.py"))
freezer = importlib.util.module_from_spec(freeze_spec)
freeze_spec.loader.exec_module(freezer)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class StatisticsTests(unittest.TestCase):
    def test_wilson_perfect_score_retains_uncertainty(self):
        lo, hi = scorer.wilson(60, 60)
        self.assertLess(lo, 1)
        self.assertGreater(lo, 0.9)
        self.assertAlmostEqual(hi, 1)
        self.assertEqual(scorer.wilson(0, 0), [None, None])

    def test_fixed_label_macro_f1_with_fault_label_permutation(self):
        labels = ["Normal", "Fault_1", "Fault_13"]
        truth = labels * 2
        predicted = ["Normal", "Fault_13", "Fault_1"] * 2
        metric = scorer.metric_values(truth, predicted, labels, "tep")
        self.assertAlmostEqual(metric["accuracy"], 1 / 3)
        self.assertAlmostEqual(metric["macro_f1"], 1 / 3)
        self.assertEqual(metric["per_class"]["Fault_13"]["tp"], 0)
        self.assertEqual(metric["confusion_matrix"]["Fault_13"]["Fault_1"], 2)
        self.assertEqual(metric["miss_numerator"], 0)  # subtype errors are not fault-as-Normal errors
        self.assertEqual(metric["miss_denominator"], 4)

    def test_identical_paired_predictions_have_exact_zero_differences(self):
        labels = ["Normal", "Fault_1", "Fault_13"]
        truth = labels * 4
        predicted = ["Normal", "Normal", "Fault_13"] * 4
        draws = np.random.default_rng(20260926).integers(0, 12, size=(10000, 12))
        left = scorer.bootstrap_values(truth, predicted, labels, draws)
        right = scorer.bootstrap_values(truth, list(predicted), labels, draws)
        for metric in ["accuracy", "macro_f1"]:
            self.assertEqual(scorer.percentile95(left[metric] - right[metric]), [0.0, 0.0])

    def test_paired_difference_uses_same_resampled_groups(self):
        truth = ["Normal", "Fault_1", "Fault_13"]
        left = ["Normal", "Fault_1", "Normal"]
        right = ["Normal", "Normal", "Fault_13"]
        draws = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 2]])
        a = scorer.bootstrap_values(truth, left, truth, draws)
        b = scorer.bootstrap_values(truth, right, truth, draws)
        np.testing.assert_allclose(a["accuracy"] - b["accuracy"], [1 / 3, 1, -1])

    def test_critical_miss_definition_and_support(self):
        result = scorer.metric_values(["Healthy", "Critical", "Critical"],
                                      ["Critical", "Healthy", "Degrading"], scorer.LABELS["cmapss"], "cmapss")
        self.assertEqual(result["miss_definition"], "critical_as_noncritical")
        self.assertEqual(result["miss_numerator"], 2)
        self.assertEqual(result["miss_denominator"], 2)
        self.assertEqual(result["critical_as_healthy_count"], 1)

    def test_tep_false_alarm_is_normal_fn_over_normal_support(self):
        result = scorer.metric_values(["Normal", "Normal", "Normal", "Fault_1", "Fault_13"],
                                      ["Normal", "Fault_1", "Fault_13", "Normal", "Fault_1"],
                                      scorer.LABELS["tep"], "tep")
        normal = result["per_class"]["Normal"]
        self.assertEqual(result["false_alarm_numerator"], normal["fn"])
        self.assertEqual(result["false_alarm_denominator"], normal["support"])
        self.assertEqual(result["false_alarm_rate"], 2 / 3)
        self.assertEqual(result["false_alarm_rate_wilson95"], scorer.wilson(2, 3))
        self.assertEqual(result["miss_rate"], 1 / 2)


class FrozenAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.runner = self.base / "synthetic_runner.py"
        self.instructions = "Synthetic fixture: no tools."
        self.runner.write_text(f"INSTRUCTIONS = {self.instructions!r}\n")
        self.groups = {}
        self.payloads = {}
        for name in scorer.GROUP_NAMES:
            dataset, condition, repeat = name.split("_")
            samples = [{"id": f"{index:08x}", "prompt": f"Synthetic sensor value: {index}"}
                       for index in range(scorer.SIZES[dataset])]
            group = {"dataset": dataset, "condition": condition, "repeat": int(repeat[1:]),
                     "valid_labels": scorer.LABELS[dataset], "n_samples": len(samples), "samples": samples}
            self.groups[name] = group
            payload = "Synthetic classification request.\n\n" + json.dumps(
                {"valid_labels": group["valid_labels"], "samples": samples}) + "\n"
            self.payloads[name] = payload
            path = self.base / "inference/prompts" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        write_json(self.base / "inference/batches.json", {"groups": self.groups})
        self.input_hash = scorer.sha((self.base / "inference/batches.json").read_bytes())
        self.protocol = {"version": "revision-v3", "models": scorer.MODELS, "reasoning_effort": "medium",
                         "repeats": 3, "expected_groups": scorer.GROUP_NAMES, "expected_sessions": 27,
                         "primary_repeat": 1, "primary_metrics": ["macro_f1", "accuracy"],
                         "bootstrap": {"replicates": 10000, "seed": 20260926}, "sample_size": scorer.SIZES,
                         "input_sha256": self.input_hash, "independent_sample_calls": False,
                         "dataset_sha256": "no keys are present in this fixture",
                         "prompt_sha256": {name: scorer.sha(value.encode()) for name, value in self.payloads.items()}}
        write_json(self.base / "protocol.json", self.protocol)
        self.protocol_hash = scorer.sha((self.base / "protocol.json").read_bytes())
        self.runner_hash = scorer.sha(self.runner.read_bytes())
        self.failures = []
        self.reconnections = []
        for model in scorer.MODELS:
            for name, group in self.groups.items():
                folder = self.base / "predictions" / model
                result = [{"id": row["id"], "predicted": group["valid_labels"][-1]} for row in group["samples"]]
                write_json(folder / f"{name}_results.json", result)
                raw = json.dumps(result)
                (folder / f"{name}_attempt1_response.txt").write_text(raw)
                (folder / f"{name}_attempt1_stderr.txt").write_text("")
                usage = {"input_tokens": 12, "output_tokens": 34}
                events = [{"type": "thread.started", "thread_id": f"{model}-{name}"},
                          {"type": "turn.started"},
                          {"type": "item.completed", "item": {"type": "agent_message", "text": raw}},
                          {"type": "turn.completed", "usage": usage}]
                (folder / f"{name}_attempt1_events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
                request = {"model": model, "reasoning_effort": "medium",
                           "developer_instructions": self.instructions, "user_message": self.payloads[name]}
                write_json(folder / f"{name}_request.json", request)
                metadata = {"status": "complete", "protocol": scorer.PROTOCOL, "model_requested": model,
                            "reasoning_effort": "medium", "transport": "codex-cli-chatgpt-auth-https",
                            "runner_sha256": self.runner_hash, "protocol_sha256": self.protocol_hash,
                            "input_sha256": self.input_hash, "prompt_txt_sha256": scorer.sha(self.payloads[name].encode()),
                            "dataset": group["dataset"], "condition": group["condition"], "repeat": group["repeat"],
                            "batch": name, "n_samples": group["n_samples"], "tools_observed": 0,
                            "answer_keys_read": False, "independent_session_per_group": True,
                            "independent_session_per_sample": False,
                            "prediction_sha256": scorer.sha((folder / f"{name}_results.json").read_bytes()),
                            "cli_version": "synthetic-cli", "base_instructions_sha256": scorer.sha(self.instructions.encode()),
                            "fingerprint": scorer.sha(json.dumps({"model": model, "effort": "medium", "payload": self.payloads[name],
                                                                 "instructions": self.instructions, "protocol": scorer.PROTOCOL}, sort_keys=True).encode()),
                            "attempts": [{"attempt": 1, "events_file": f"{name}_attempt1_events.jsonl",
                                          "response_file": f"{name}_attempt1_response.txt", "stderr_file": f"{name}_attempt1_stderr.txt",
                                          "malformed_event_lines": [], "errors": [], "returncode": 0, "timed_out": False,
                                          "usage": usage}]}
                write_json(folder / f"{name}_metadata.json", metadata)
        self.refreeze_fixture()

    def refreeze_fixture(self):
        output = self.base / "predictions"
        amendment = self.base / "failure_reporting_amendment.json"
        transport_note = self.base / "transport_audit_note.json"
        valid_count = 1800 - sum(failure["expected_n"] for failure in self.failures)
        manifest = {"schema": "revision-v3-outcomes-frozen-v2", "protocol_sha256": self.protocol_hash,
                    "input_sha256": self.input_hash, "runner_sha256": self.runner_hash,
                    "failure_reporting_amendment_sha256": scorer.sha(amendment.read_bytes()) if amendment.exists() else None,
                    "transport_audit_note_sha256": scorer.sha(transport_note.read_bytes()) if transport_note.exists() else None,
                    "recovered_stream_reconnection_details": self.reconnections,
                    "sessions": 27, "planned_sessions": 27, "completed_sessions": 27 - len(self.failures),
                    "failed_sessions": len(self.failures), "predictions": valid_count, "total_valid_predictions": valid_count,
                    "expected_predictions": 1800, "session_failures": self.failures,
                    "answer_keys_read": False, "frozen_at": "2026-02-01T00:00:00+00:00",
                    "files": {path.relative_to(output).as_posix(): scorer.sha(path.read_bytes())
                              for path in output.rglob("*") if path.is_file() and path.name != "frozen_manifest.json"}}
        write_json(output / "frozen_manifest.json", manifest)

    def make_semantic_failure(self, name="cmapss_clean_r1", model="gpt-5.5"):
        folder = self.base / "predictions" / model
        result = folder / f"{name}_results.json"
        rows = json.loads(result.read_text())[:-1]
        result.unlink()
        raw = json.dumps(rows)
        (folder / f"{name}_attempt1_response.txt").write_text(raw)
        events_path = folder / f"{name}_attempt1_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        events[2]["item"]["text"] = raw
        events_path.write_text("\n".join(json.dumps(event) for event in events))
        metadata_path = folder / f"{name}_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["status"] = "invalid_response"
        metadata["validation_error"] = "Incomplete prediction count"
        del metadata["prediction_sha256"]
        write_json(metadata_path, metadata)
        group = self.groups[name]
        failure = {"model": model, "group": name, "dataset": group["dataset"],
                   "condition": group["condition"], "repeat": group["repeat"], "primary": group["repeat"] == 1,
                   "status": "invalid_response", "reason": "Incomplete prediction count",
                   "reproduced_validation_error": "Raw response: incomplete predictions",
                   "expected_n": group["n_samples"], "actual_raw_count": len(rows), "session_id": f"{model}-{name}"}
        self.failures.append(failure)
        write_json(self.base / "failure_reporting_amendment.json", {
            "amendment_id": "failure-reporting-1", "recorded_at_utc": "2026-01-01T00:00:00+00:00",
            "original_protocol_sha256": self.protocol_hash,
            "trigger": {key: failure[key] for key in ["model", "group", "status", "reason", "expected_n", "actual_raw_count"]},
        })
        self.refreeze_fixture()

    def add_reconnect(self, *, message=None, count=1, model="gpt-5.6-sol", name="cmapss_clean_r3"):
        message = scorer.RECOVERABLE_RECONNECT_MESSAGE if message is None else message
        folder = self.base / "predictions" / model
        events_path = folder / f"{name}_attempt1_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        errors = [{"type": "error", "message": message} for _ in range(count)]
        events[2:2] = errors
        events_path.write_text("\n".join(json.dumps(event) for event in events))
        metadata_path = folder / f"{name}_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["attempts"][0]["errors"] = errors
        write_json(metadata_path, metadata)
        self.reconnections = [{"model": model, "group": name, "attempt": 1, "session_id": f"{model}-{name}",
                               "events_file": events_path.name, "line": 3 + index, "message": message,
                               "subsequent_valid_output": True, "subsequent_turn_completed": True}
                              for index in range(count)]
        write_json(self.base / "transport_audit_note.json", {
            "note_id": "transport-audit-1", "recorded_at_utc": "2026-01-01T00:00:00+00:00",
            "original_protocol_sha256": self.protocol_hash,
            "trigger": {"model": model, "group": name, "attempt": 1, "message": message},
        })
        self.refreeze_fixture()

    def synthetic_scores(self):
        protocol, groups, predictions, audit = scorer.verify_frozen_runs(self.base, self.runner)
        datasets = {name: [{"id": f"{index:08x}", "label": scorer.LABELS[name][index % len(scorer.LABELS[name])],
                            "group": f"synthetic-{name}-{index}"} for index in range(size)]
                    for name, size in scorer.SIZES.items()}
        baselines = {name: {method: [scorer.LABELS[name][-1]] * len(rows) for method in scorer.BASELINES}
                     for name, rows in datasets.items()}
        return scorer.compute_scores(protocol, groups, predictions, audit, datasets, baselines, {"synthetic": True})

    def test_all_27_verified_with_leading_zero_ids_and_without_any_keys(self):
        protocol, groups, predictions, audit = scorer.verify_frozen_runs(self.base, self.runner)
        self.assertFalse((self.base / "data/datasets.json").exists())
        self.assertEqual(audit["completed_sessions"], 27)
        self.assertEqual(audit["unique_completed_sessions"], 27)
        self.assertEqual(audit["total_predictions"], 1800)
        self.assertEqual(predictions["gpt-5.5"]["tep_clean_r1"][0], {"id": "00000000", "predicted": "Fault_13"})

    def test_hash_mutation_is_rejected_before_keys(self):
        path = self.base / "predictions/gpt-5.5/tep_clean_r1_results.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Frozen predictions/logs changed"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_missing_file_is_rejected(self):
        (self.base / "predictions/gpt-6-astra/tep_clean_r3_results.json").unlink()
        with self.assertRaisesRegex(ValueError, "Frozen predictions/logs changed"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_tool_event_rejected_even_when_hash_is_frozen(self):
        path = self.base / "predictions/gpt-5.5/cmapss_clean_r1_attempt1_events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events.insert(2, {"type": "item.completed", "item": {"type": "command_execution", "command": "cat answers"}})
        path.write_text("\n".join(json.dumps(event) for event in events))
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "tool use"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_diagnostic_error_is_not_hidden_as_non_tool(self):
        path = self.base / "predictions/gpt-5.5/cmapss_clean_r1_attempt1_events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events.insert(2, {"type": "item.completed", "item": {"type": "error", "message": "diagnostic failure"}})
        path.write_text("\n".join(json.dumps(event) for event in events))
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "diagnostic error"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_reused_session_is_rejected(self):
        path = self.base / "predictions/gpt-5.5/cmapss_clean_r2_attempt1_events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[0]["thread_id"] = "gpt-5.5-cmapss_clean_r1"
        path.write_text("\n".join(json.dumps(event) for event in events))
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "reused session"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_main_never_opens_keys_if_audit_fails(self):
        with patch.object(scorer, "verify_frozen_runs", side_effect=ValueError("synthetic invalid freeze")), \
                patch.object(scorer, "load_evaluation_data", side_effect=AssertionError("key read")) as loader:
            self.assertEqual(scorer.main(["--root", str(self.base)]), 1)
            loader.assert_not_called()

    def test_report_keeps_27_sessions_separate_and_has_zero_paired_differences(self):
        protocol, groups, predictions, audit = scorer.verify_frozen_runs(self.base, self.runner)
        datasets = {name: [{"id": f"{index:08x}", "label": scorer.LABELS[name][index % len(scorer.LABELS[name])],
                            "group": f"synthetic-{name}-{index}"} for index in range(size)]
                    for name, size in scorer.SIZES.items()}
        baselines = {name: {method: [scorer.LABELS[name][-1]] * len(rows) for method in scorer.BASELINES}
                     for name, rows in datasets.items()}
        report = scorer.compute_scores(protocol, groups, predictions, audit, datasets, baselines, {"synthetic": True})
        self.assertEqual(len(report["metrics"]), 33)  # 27 sessions + six one-fit baseline evaluations
        self.assertEqual(len(report["repeat_variability"]), 9)
        self.assertEqual(len(report["paired_contrasts"]), 39)  # 30 clean method pairs + nine stress pairs
        self.assertEqual(sum(row["primary"] and row["model_type"] == "LLM" for row in report["metrics"]), 9)
        for row in report["paired_contrasts"]:
            self.assertEqual(row["accuracy_difference"], 0)
            self.assertEqual(row["accuracy_difference_group_bootstrap95"], [0, 0])
            self.assertEqual(row["macro_f1_difference_group_bootstrap95"], [0, 0])
            self.assertEqual(row["flip_rate"], 0)
        scorer.write_report(self.base, report)
        self.assertEqual(len(json.loads((self.base / "analysis/scores.json").read_text())["metrics"]), 33)
        self.assertEqual(len((self.base / "analysis/metrics.csv").read_text().splitlines()), 34)
        self.assertEqual(len((self.base / "analysis/paired.csv").read_text().splitlines()), 40)
        self.assertIn("no significance-based claim selection", (self.base / "analysis/summary.md").read_text())

    def test_session_prediction_disagreement_is_distinct_from_mapping_contrast(self):
        protocol, groups, predictions, audit = scorer.verify_frozen_runs(self.base, self.runner)
        # Two separate engines change across clean C-MAPSS sessions; TEP mappings remain identical.
        predictions["gpt-5.5"]["cmapss_clean_r2"][0]["predicted"] = "Healthy"
        predictions["gpt-5.5"]["cmapss_clean_r3"][1]["predicted"] = "Degrading"
        datasets = {name: [{"id": f"{index:08x}", "label": scorer.LABELS[name][index % len(scorer.LABELS[name])],
                            "group": f"synthetic-{name}-{index}"} for index in range(size)]
                    for name, size in scorer.SIZES.items()}
        baselines = {name: {method: [scorer.LABELS[name][-1]] * len(rows) for method in scorer.BASELINES}
                     for name, rows in datasets.items()}
        report = scorer.compute_scores(protocol, groups, predictions, audit, datasets, baselines, {"synthetic": True})
        variation = next(row for row in report["repeat_variability"]
                         if row["model"] == "gpt-5.5" and row["dataset"] == "cmapss")
        self.assertEqual(variation["prediction_disagreement"], [
            {"repeat_a": 1, "repeat_b": 2, "n": 60, "flip_count": 1, "flip_rate": 1 / 60},
            {"repeat_a": 1, "repeat_b": 3, "n": 60, "flip_count": 1, "flip_rate": 1 / 60},
            {"repeat_a": 2, "repeat_b": 3, "n": 60, "flip_count": 2, "flip_rate": 2 / 60},
        ])
        self.assertEqual(variation["any_repeat_flip_count"], 2)
        self.assertTrue(all(row["flip_count"] == 0 for row in report["paired_contrasts"]
                            if row["contrast"] == "clean_minus_descriptor_mapping_fault"))

    def test_semantic_failure_is_na_and_never_replaced_by_repeat2(self):
        self.make_semantic_failure()
        report = self.synthetic_scores()
        self.assertEqual(report["audit"]["planned_sessions"], 27)
        self.assertEqual(report["audit"]["completed_sessions"], 26)
        self.assertEqual(report["audit"]["failed_sessions"], 1)
        self.assertEqual(report["audit"]["total_valid_predictions"], 1740)
        self.assertEqual(report["audit"]["expected_predictions"], 1800)
        row = next(row for row in report["metrics"] if row["id"] == "gpt-5.5/cmapss_clean_r1")
        self.assertTrue(row["primary"])
        self.assertFalse(row["available"])
        self.assertEqual(row["n_scored"], 0)
        for field in ["accuracy", "macro_f1", "miss_rate", "correct", "per_class"]:
            self.assertIsNone(row[field])
        self.assertEqual(report["session_failures"][0]["actual_raw_count"], 59)
        variation = next(row for row in report["repeat_variability"]
                         if row["model"] == "gpt-5.5" and row["dataset"] == "cmapss")
        self.assertEqual(variation["missing_repeats"], [1])
        self.assertEqual(variation["available_repeats"], [2, 3])
        self.assertEqual(variation["n_successful_sessions"], 2)
        self.assertIsNone(variation["accuracy"]["repeat1"])
        self.assertEqual(len(variation["prediction_disagreement"]), 1)
        self.assertEqual(variation["prediction_disagreement"][0]["repeat_a"], 2)
        self.assertEqual(len(report["paired_contrasts"]), 34)
        self.assertEqual(len(report["unavailable_contrasts"]), 5)
        self.assertTrue(all("gpt-5.5/cmapss_clean_r1" in row["unavailable_sessions"]
                            for row in report["unavailable_contrasts"]))
        scorer.write_report(self.base, report)
        summary = (self.base / "analysis/summary.md").read_text()
        self.assertIn("| cmapss | gpt-5.5 | N/A | N/A | N/A |", summary)
        self.assertIn("pre-score reporting amendment", summary)

    def test_freezer_accepts_27_terminal_outcomes_without_opening_labels(self):
        self.make_semantic_failure()
        (self.base / "predictions/frozen_manifest.json").unlink()
        manifest = freezer.freeze(self.base, self.runner)
        self.assertEqual(manifest["completed_sessions"], 26)
        self.assertEqual(manifest["failed_sessions"], 1)
        self.assertEqual(manifest["total_valid_predictions"], 1740)
        self.assertFalse((self.base / "data/datasets.json").exists())
        self.assertEqual(freezer.freeze(self.base, self.runner), manifest)

    def test_semantic_failure_requires_pre_score_amendment(self):
        self.make_semantic_failure()
        (self.base / "failure_reporting_amendment.json").unlink()
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "requires the pre-score"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_amendment_hash_and_protocol_reference_are_checked(self):
        self.make_semantic_failure()
        path = self.base / "failure_reporting_amendment.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "failure_reporting_amendment_sha256"):
            scorer.verify_frozen_runs(self.base, self.runner)
        value = json.loads(path.read_text())
        value["original_protocol_sha256"] = "different"
        write_json(path, value)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "original_protocol_sha256"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_failed_partial_response_cannot_be_hand_repaired(self):
        self.make_semantic_failure()
        write_json(self.base / "predictions/gpt-5.5/cmapss_clean_r1_results.json",
                   [{"id": row["id"], "predicted": "Critical"} for row in self.groups["cmapss_clean_r1"]["samples"]])
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "repaired/scored results"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_amendment_cannot_be_future_dated_or_post_freeze(self):
        self.make_semantic_failure()
        path = self.base / "failure_reporting_amendment.json"
        amendment = json.loads(path.read_text())
        amendment["recorded_at_utc"] = "9999-01-01T00:00:00+00:00"
        write_json(path, amendment)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "timestamp before scoring"):
            scorer.verify_frozen_runs(self.base, self.runner)
        amendment["recorded_at_utc"] = "2026-03-01T00:00:00+00:00"
        write_json(path, amendment)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "after the outcome freeze"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_missing_or_pending_outcome_cannot_be_frozen(self):
        path = self.base / "predictions/gpt-5.5/cmapss_clean_r2_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["status"] = "pending"
        write_json(path, metadata)
        (self.base / "predictions/frozen_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "every planned outcome must be terminal"):
            freezer.freeze(self.base, self.runner)
        self.assertFalse((self.base / "predictions/frozen_manifest.json").exists())
        path.unlink()
        with self.assertRaisesRegex(ValueError, "27 planned outcome metadata"):
            freezer.freeze(self.base, self.runner)

    def test_semantic_retry_is_rejected_even_if_second_output_is_valid(self):
        name = "cmapss_clean_r1"
        folder = self.base / "predictions/gpt-5.5"
        metadata_path = folder / f"{name}_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        original = metadata["attempts"][0]
        second = dict(original, attempt=2)
        for field, suffix in [("events_file", "events.jsonl"), ("response_file", "response.txt"), ("stderr_file", "stderr.txt")]:
            second[field] = f"{name}_attempt2_{suffix}"
            raw = (folder / original[field]).read_text()
            if field == "events_file":
                events = [json.loads(line) for line in raw.splitlines()]
                events[0]["thread_id"] = "forbidden-semantic-retry"
                raw = "\n".join(json.dumps(event) for event in events)
            (folder / second[field]).write_text(raw)
        metadata["attempts"].append(second)
        write_json(metadata_path, metadata)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "rerun after completed semantic output"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_exact_reconnect_is_retained_and_counted_after_valid_completed_output(self):
        self.add_reconnect()
        audit = scorer.verify_frozen_runs(self.base, self.runner)[3]
        self.assertEqual(audit["diagnostic_errors_count"], 1)
        self.assertEqual(audit["recovered_stream_reconnections"], 1)
        self.assertEqual(audit["recovered_stream_reconnection_details"], self.reconnections)
        self.assertEqual(audit["total_cli_attempts"], 27)
        self.assertEqual(audit["total_transport_attempts_including_reconnections"], 28)
        self.assertEqual(audit["runs"]["gpt-5.6-sol"]["cmapss_clean_r3"]["transport_attempts_including_reconnections"], 2)
        self.assertFalse((self.base / "data/datasets.json").exists())

    def test_unknown_error_does_not_become_recoverable_by_completing(self):
        self.add_reconnect(message="Reconnecting... 1/5 (different unverified failure)")
        with self.assertRaisesRegex(ValueError, "fatal or unknown diagnostic error"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_turn_failed_is_fatal_even_if_followed_by_complete_output(self):
        self.add_reconnect()
        folder = self.base / "predictions/gpt-5.6-sol"
        path = folder / "cmapss_clean_r3_attempt1_events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[2]["type"] = "turn.failed"
        path.write_text("\n".join(json.dumps(event) for event in events))
        metadata_path = folder / "cmapss_clean_r3_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["attempts"][0]["errors"] = [events[2]]
        write_json(metadata_path, metadata)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "fatal or unknown diagnostic error"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_two_internal_reconnections_exceed_shared_transport_ceiling(self):
        self.add_reconnect(count=2)
        with self.assertRaisesRegex(ValueError, "plus logged stream reconnections exceed"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_cli_retry_plus_internal_reconnect_exceeds_shared_transport_ceiling(self):
        self.add_reconnect()
        folder = self.base / "predictions/gpt-5.6-sol"
        name = "cmapss_clean_r3"
        metadata_path = folder / f"{name}_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        first = metadata["attempts"][0]
        second = dict(first, attempt=2)
        for field in ["events_file", "response_file", "stderr_file"]:
            second[field] = first[field].replace("attempt1", "attempt2")
            (folder / second[field]).write_bytes((folder / first[field]).read_bytes())
        (folder / first["events_file"]).write_text("\n".join(json.dumps(event) for event in [
            {"type": "thread.started", "thread_id": "synthetic-timed-out-transport"}, {"type": "turn.started"}]))
        first.update(returncode=-1, timed_out=True, usage=None, errors=[])
        metadata["attempts"] = [first, second]
        write_json(metadata_path, metadata)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "plus logged stream reconnections exceed"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_reconnect_does_not_excuse_invalid_output(self):
        self.make_semantic_failure(name="cmapss_clean_r3", model="gpt-5.6-sol")
        self.add_reconnect()
        with self.assertRaisesRegex(ValueError, "subsequent complete valid output"):
            scorer.verify_frozen_runs(self.base, self.runner)

    def test_reconnect_requires_matching_metadata_error_and_audit_note(self):
        self.add_reconnect()
        path = self.base / "predictions/gpt-5.6-sol/cmapss_clean_r3_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["attempts"][0]["errors"] = []
        write_json(path, metadata)
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "errors metadata mismatch"):
            scorer.verify_frozen_runs(self.base, self.runner)
        metadata["attempts"][0]["errors"] = [{"type": "error", "message": scorer.RECOVERABLE_RECONNECT_MESSAGE}]
        write_json(path, metadata)
        (self.base / "transport_audit_note.json").unlink()
        self.refreeze_fixture()
        with self.assertRaisesRegex(ValueError, "requires pre-score transport audit note"):
            scorer.verify_frozen_runs(self.base, self.runner)


if __name__ == "__main__":
    unittest.main()
