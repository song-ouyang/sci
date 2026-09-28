"""Answer-key-free synthetic checks for the archival Claude audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "src/analysis/audit_revision_v3_claude_extension.py"
spec = importlib.util.spec_from_file_location("claude_extension_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.base, self.parent = root / "extension", root / "results/revision_v3"
        self.groups = {}
        prompts = {}
        for name in audit.GROUPS:
            dataset, rest = name.split("_", 1)
            condition, repeat = rest.rsplit("_r", 1)
            labels = ["Healthy", "Critical"] if dataset == "cmapss" else ["Normal", "Fault_1", "Fault_13"]
            group = {"dataset": dataset, "condition": condition, "repeat": int(repeat),
                     "n_samples": 2, "valid_labels": labels,
                     "samples": [{"id": "00000001", "prompt": "sensor = 1"},
                                 {"id": "00000002", "prompt": "sensor = 2"}]}
            self.groups[name] = group
            prompt_path = self.parent / f"inference/prompts/{name}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(f"{dataset}_{condition}\n00000001 sensor = 1\n00000002 sensor = 2\n")
            prompts[name] = prompt_path.read_bytes()
        write_json(self.parent / "inference/batches.json", {"groups": self.groups})
        write_json(self.parent / "protocol.json", {"version": "synthetic-parent"})
        self.input_hash = audit.sha256((self.parent / "inference/batches.json").read_bytes())
        self.instructions = "Classify using the supplied prompt."
        protocol = {"models_requested": list(audit.MODELS), "expected_groups": list(audit.GROUPS),
                    "expected_sessions": 18, "instructions": self.instructions,
                    "input_sha256": self.input_hash,
                    "parent_protocol_sha256": audit.sha256((self.parent / "protocol.json").read_bytes()),
                    "prompt_sha256": {name.rsplit("_r", 1)[0]: audit.sha256(raw) for name, raw in prompts.items()}}
        write_json(self.base / "protocol.json", protocol)
        write_json(self.base / "source_manifest.json", {
            "batches_json": {"path": "results/revision_v3/inference/batches.json", "sha256": self.input_hash},
            "prompt_files": {name: {"path": f"results/revision_v3/inference/prompts/{name}.txt",
                                     "sha256": audit.sha256(raw), "bytes": len(raw)} for name, raw in prompts.items()}})
        for model in audit.MODELS:
            for name in audit.GROUPS:
                group = self.groups[name]
                raw_prompt = prompts[name]
                predictions = [{"id": row["id"], "predicted": group["valid_labels"][0]} for row in group["samples"]]
                write_json(self.path(model, name, "results.json"), predictions)
                self.path(model, name, "response.txt").write_text(json.dumps(predictions))
                request = {"model": model, "group_name": name, "transport": "claude-code-agent-tool",
                           "instructions": self.instructions, "user_message": raw_prompt.decode(),
                           "prompt_file_sha256": audit.sha256(raw_prompt), "input_sha256": self.input_hash,
                           "full_prompt_sha256": audit.sha256((self.instructions + "\n\n" + raw_prompt.decode()).encode()),
                           "n_samples": 2, "valid_labels": group["valid_labels"]}
                write_json(self.path(model, name, "request.json"), request)
                metadata = {"model_requested": model, "group_name": name, "transport": "claude-code-agent-tool",
                            "status": "complete", "validation_error": None, "answer_keys_read": False,
                            "dataset": group["dataset"], "condition": group["condition"], "repeat": group["repeat"],
                            "n_samples": 2, "n_predictions": 2}
                for field, suffix in (("prediction_sha256", "results.json"), ("request_sha256", "request.json"), ("response_text_sha256", "response.txt")):
                    metadata[field] = audit.sha256(self.path(model, name, suffix).read_bytes())
                write_json(self.path(model, name, "metadata.json"), metadata)

    def path(self, model, name, suffix):
        return self.base / "predictions" / model / f"{name}_{suffix}"

    def report(self):
        return audit.audit_extension(self.base, self.parent)

    def session(self, report, model="claude-sonnet-4-6", name="tep_misnamed_r3"):
        return next(s for s in report["sessions"] if s["model"] == model and s["group"] == name)

    def codes(self, session):
        return {p["code"] for p in session["issues"]}

    def test_additional_model_is_audited_without_promoting_or_rewriting_protocol(self):
        import shutil
        model = "claude-fable-5-1"
        source = self.base / "predictions" / audit.MODELS[1]
        target = self.base / "predictions" / model
        shutil.copytree(source, target)
        protocol_before = (self.base / "protocol.json").read_bytes()
        for name in audit.GROUPS:
            request_path = self.path(model, name, "request.json")
            request = json.loads(request_path.read_text()); request["model"] = model
            write_json(request_path, request)
            metadata_path = self.path(model, name, "metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata.update(model_requested=model, model_alias="opus", request_sha256=audit.sha256(request_path.read_bytes()))
            write_json(metadata_path, metadata)
        report = self.report()
        self.assertEqual(report["counts"]["audited_records"], 27)
        self.assertEqual(report["counts"]["artifact_arrays_valid"], 27)
        self.assertEqual(report["counts"]["eligible_primary"], 0)
        self.assertEqual(report["protocol_declared_sessions"], 18)
        self.assertEqual((self.base / "protocol.json").read_bytes(), protocol_before)
        self.assertIn("INVOCATION_ALIAS_IDENTITY_UNVERIFIED", self.codes(self.session(report, model)))

    def test_partial_additional_directory_cannot_hide_missing_groups(self):
        model = "claude-extra-test"
        path = self.path(model, "tep_clean_r1", "results.json")
        write_json(path, [{"id": "00000001", "predicted": "Normal"}, {"id": "00000002", "predicted": "Normal"}])
        report = self.report()
        rows = [s for s in report["sessions"] if s["model"] == model]
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(s["artifact_array_valid"] for s in rows), 1)
        self.assertTrue(all(not s["eligible_for_primary_comparison"] for s in rows))
        self.assertTrue(all(not s["artifact_chain_consistent"] for s in rows))

    def test_consistent_arrays_never_imply_verified_inference(self):
        report = self.report()
        self.assertEqual(report["counts"]["planned"], 18)
        self.assertEqual(report["counts"]["artifact_arrays_valid"], 18)
        self.assertEqual(report["counts"]["artifact_chains_consistent"], 18)
        self.assertEqual(report["counts"]["eligible_primary"], 0)
        self.assertEqual(report["counts"]["reported_contaminated"], 0)
        self.assertTrue(all(not s["eligible_for_primary_comparison"] for s in report["sessions"]))
        self.assertTrue(all(s["status"] == "provenance_unverified" for s in report["sessions"]))
        # All synthetic condition repeats intentionally have identical predictions.
        self.assertNotIn("REPORTED_CONTAMINATED", self.codes(self.session(report)))

    def test_contaminated_and_suspect_are_not_eligible(self):
        path = self.path(audit.MODELS[0], "tep_misnamed_r3", "metadata.json")
        meta = json.loads(path.read_text())
        meta.update(integrity_flag="CONTAMINATED", data_integrity_flag="suspect")
        write_json(path, meta)
        report = self.report()
        session = self.session(report)
        self.assertTrue(session["artifact_array_valid"])
        self.assertEqual(session["status"], "reported_contaminated")
        self.assertFalse(session["eligible_for_primary_comparison"])
        self.assertIn("REPORTED_SUSPECT", self.codes(session))
        self.assertEqual(report["counts"]["reported_contaminated"], 1)

    def test_response_hash_change_detected_even_when_json_same(self):
        path = self.path(audit.MODELS[0], "tep_misnamed_r3", "response.txt")
        path.write_text(path.read_text() + "\n")
        session = self.session(self.report())
        self.assertIn("HASH_MISMATCH", self.codes(session))
        self.assertTrue(session["artifact_array_valid"])
        self.assertFalse(session["artifact_chain_consistent"])

    def test_invalid_case_sensitive_label_fails(self):
        path = self.path(audit.MODELS[0], "tep_misnamed_r3", "results.json")
        predictions = json.loads(path.read_text())
        predictions[0]["predicted"] = "fault_13"
        write_json(path, predictions)
        session = self.session(self.report())
        self.assertIn("PREDICTION_ARRAY_INVALID", self.codes(session))
        self.assertIn("RAW_RESULT_MISMATCH", self.codes(session))
        self.assertFalse(session["artifact_array_valid"])

    def test_missing_record_fails(self):
        path = self.path(audit.MODELS[0], "tep_misnamed_r3", "results.json")
        predictions = json.loads(path.read_text())
        write_json(path, predictions[:-1])
        report = self.report()
        self.assertEqual(report["counts"]["artifact_arrays_valid"], 17)
        self.assertIn("PREDICTION_ARRAY_INVALID", self.codes(self.session(report)))

    def test_missing_request_fails_even_with_valid_results(self):
        self.path(audit.MODELS[0], "tep_misnamed_r3", "request.json").unlink()
        session = self.session(self.report())
        self.assertTrue(session["artifact_array_valid"])
        self.assertFalse(session["artifact_chain_consistent"])
        self.assertIn("REQUEST_INVALID", self.codes(session))

    def test_wrong_id_order_and_extra_fields_rejected(self):
        group = self.groups["tep_clean_r1"]
        predictions = [{"id": "00000002", "predicted": "Fault_13"}, {"id": "00000001", "predicted": "Fault_1"}]
        with self.assertRaises(ValueError):
            audit.validate_array(predictions, group)
        predictions.reverse()
        audit.validate_array(predictions, group)
        predictions[0]["reason"] = "extra"
        with self.assertRaises(ValueError):
            audit.validate_array(predictions, group)

    def test_exact_v3_raw_parser_rejects_prose_and_invalid_fence(self):
        group = self.groups["tep_clean_r1"]
        predictions = [{"id": "00000001", "predicted": "Fault_1"}, {"id": "00000002", "predicted": "Fault_13"}]
        raw = json.dumps(predictions)
        self.assertEqual(audit.parse_raw_v3(raw, group), predictions)
        self.assertEqual(audit.parse_raw_v3("```json\n" + raw + "\n```", group), predictions)
        for bad in ("My answer: " + raw, "```python\n" + raw + "\n```", raw + "\nDone"):
            with self.assertRaises(ValueError):
                audit.parse_raw_v3(bad, group)
        with self.assertRaises(ValueError):
            audit.strict_json('{"id":"00000001","id":"00000002"}')

    def test_answers_scores_and_unlisted_files_never_read(self):
        write_json(self.parent / "data/datasets.json", {"SECRET_ANSWER_KEY": "not for audit"})
        write_json(self.base / "analysis/scores.json", {"SECRET_SCORE": "not for audit"})
        original_read_bytes, original_read_text = Path.read_bytes, Path.read_text
        observed = []

        def guarded_bytes(path, *args, **kwargs):
            observed.append(path)
            if path.name in {"datasets.json", "scores.json"} or "analysis" in path.parts:
                raise AssertionError("Audit opened answers or scores")
            return original_read_bytes(path, *args, **kwargs)

        def guarded_text(path, *args, **kwargs):
            if path.name in {"datasets.json", "scores.json"} or "analysis" in path.parts:
                raise AssertionError("Audit opened answers or scores")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", guarded_bytes), patch.object(Path, "read_text", guarded_text):
            report = self.report()
        self.assertFalse(report["answer_key_files_opened"])
        self.assertEqual(len(observed), 85)
        self.assertFalse(any("datasets.json" in key for key in report["source_snapshot_sha256"]))

    def test_placeholder_parent_hash_and_fake_later_freeze_do_not_restore_eligibility(self):
        path = self.base / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["parent_protocol_sha256"] = "see_revision_v3/protocol.json"
        write_json(path, protocol)
        write_json(self.base / "predictions/frozen_manifest.json", {"frozen_at": "yesterday", "files": {}})
        report = self.report()
        codes = {p["code"] for p in report["global_issues"]}
        self.assertIn("HASH_MISSING_OR_INVALID", codes)
        self.assertIn("PRE_SCORE_FREEZE_UNVERIFIABLE", codes)
        self.assertEqual(report["counts"]["eligible_primary"], 0)
        self.assertNotIn("extension:predictions/frozen_manifest.json", report["source_snapshot_sha256"])

    def test_manifest_paths_cannot_redirect_to_answers(self):
        path = self.base / "source_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["batches_json"]["path"] = "data/datasets.json"
        write_json(path, manifest)
        report = self.report()
        self.assertIn("FIELD_MISMATCH", {p["code"] for p in report["global_issues"]})
        self.assertNotIn("parent:data/datasets.json", report["source_snapshot_sha256"])

    def test_cli_writes_only_audit_and_returns_blocked(self):
        before = {p: audit.sha256(p.read_bytes()) for root in (self.base, self.parent) for p in root.rglob("*") if p.is_file()}
        self.assertEqual(audit.main(["--base", str(self.base), "--parent", str(self.parent)]), 2)
        after = {p: audit.sha256(p.read_bytes()) for p in before}
        self.assertEqual(before, after)
        self.assertTrue((self.base / "audit/report.json").is_file())
        self.assertTrue((self.base / "audit/report.md").is_file())
        snapshot = json.loads((self.base / "audit/report.json").read_text())
        self.assertEqual(snapshot["snapshot_kind"], "post_hoc_current_bytes_not_pre_score_freeze")


    def test_root_scorer_contaminated_session_has_null_metrics_without_keys(self):
        scorer_spec = importlib.util.spec_from_file_location("claude_blocked_scorer", MODULE.with_name("score_revision_v3_claude.py"))
        scorer = importlib.util.module_from_spec(scorer_spec)
        scorer_spec.loader.exec_module(scorer)
        path = self.path(audit.MODELS[0], "tep_misnamed_r3", "metadata.json")
        meta = json.loads(path.read_text())
        meta["integrity_flag"] = "CONTAMINATED"
        write_json(path, meta)
        original = Path.read_bytes

        def guard(path, *args, **kwargs):
            if path.name == "datasets.json":
                raise AssertionError("Scorer opened answer-key file")
            return original(path, *args, **kwargs)

        with patch.object(scorer, "EXT_DIR", self.base), patch.object(scorer, "ROOT", self.parent.parent.parent), patch.object(Path, "read_bytes", guard):
            self.assertEqual(scorer.main([]), 2)
        result = json.loads((self.base / "analysis/scores.json").read_text())
        session = result["scores"][audit.MODELS[0]]["tep_misnamed_r3"]
        self.assertFalse(result["answer_key_files_opened"])
        self.assertFalse(session["eligible_for_primary_comparison"])
        self.assertIsNone(session["accuracy"])
        self.assertIsNone(session["macro_f1"])
        self.assertEqual(session["accuracy_95ci"], [None, None])
        self.assertIn("REPORTED_CONTAMINATED", session["issue_codes"])

    def test_root_validate_only_does_not_create_or_change_any_file(self):
        scorer_spec = importlib.util.spec_from_file_location("claude_readonly_scorer", MODULE.with_name("score_revision_v3_claude.py"))
        scorer = importlib.util.module_from_spec(scorer_spec)
        scorer_spec.loader.exec_module(scorer)
        write_json(self.base / "analysis/scores.json", {"sentinel": "existing derived file"})
        fixture = self.parent.parent.parent
        before = {p: audit.sha256(p.read_bytes()) for p in fixture.rglob("*") if p.is_file()}
        # base is outside fixture's results/ but within fixture itself.
        with patch.object(scorer, "EXT_DIR", self.base), patch.object(scorer, "ROOT", fixture):
            self.assertEqual(scorer.main(["--validate-only"]), 2)
        after = {p: audit.sha256(p.read_bytes()) for p in fixture.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.base / "audit").exists())

    def test_markdown_retains_local_missing_hash_even_with_global_same_code(self):
        protocol_path = self.base / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["parent_protocol_sha256"] = "placeholder"
        write_json(protocol_path, protocol)
        metadata_path = self.path(audit.MODELS[0], "tep_clean_r3", "metadata.json")
        metadata = json.loads(metadata_path.read_text())
        del metadata["request_sha256"]
        write_json(metadata_path, metadata)
        report = self.report()
        audit.write_reports(report, self.base)
        markdown = (self.base / "audit/report.md").read_text()
        self.assertIn("protocol.parent_protocol_sha256", markdown)
        self.assertIn("metadata.request_sha256", markdown)


if __name__ == "__main__":
    unittest.main()
