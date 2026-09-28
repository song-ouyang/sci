"""Offline protocol invariants: no test executes Codex or reads answer files."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.experiments import run_hydraulic_v1 as h


class HydraulicRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.groups = {}
        # Deliberate prefix overlap tests independent ownership of artifacts.
        self.blocks = [["clean", "clean_repeat"], ["second"], ["third"], ["fourth"], ["last"]]
        for block in self.blocks:
            for name in block:
                self.groups[name] = {"dataset": "hydraulic", "condition": name, "repeat": 1,
                                     "samples": [{"id": "ab123456", "prompt": "pressure: 4 bar"}],
                                     "valid_labels": ["healthy", "degraded"]}
        h.write_json(self.base / "inference/batches.json", {"groups": self.groups})
        h.write_json(self.base / "schedule.json", {"blocks": self.blocks})
        self.protocol = {"model": "gpt-5.6-sol", "reasoning_effort": "medium", "cli_version": "codex-test",
                         "timeout_seconds": 1, "workers_max": 2, "max_elapsed_seconds": 14400,
                         "max_total_tokens": 600000, "code_sha256": {name: h.digest(h.ROOT / name) for name in [h.RUNNER, h.WRAPPER]},
                         "input_sha256": h.digest(self.base / "inference/batches.json"),
                         "schedule_sha256": h.digest(self.base / "schedule.json"), "prompts_sha256": {}}
        for name, group in self.groups.items():
            path = self.base / "inference/prompts" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Return the array.\n\n" + json.dumps({"valid_labels": group["valid_labels"], "samples": group["samples"]}))
            self.protocol["prompts_sha256"][name] = h.digest(path)
        self.seal()
        self.calls = []
        self.active = set()
        self.call_lock = threading.Lock()

    def seal(self):
        h.write_json(self.base / "protocol.json", self.protocol)
        (self.base / "protocol.sha256").write_text(h.digest(self.base / "protocol.json"))

    def fake_runner(self, args, cli, model, name, group, *, failed=False, usage=True):
        self.assertEqual(args.retries, 1)
        self.assertEqual(args.timeout, 1)
        with self.call_lock:
            index = next(i for i, b in enumerate(self.blocks) if name in b)
            self.assertTrue(all(next(i for i, b in enumerate(self.blocks) if n in b) == index for n in self.active))
            self.active.add(name)
            self.calls.append(name)
            self.assertLessEqual(len(self.active), 2)
        time.sleep(0.002)
        folder = args.output / model
        request = {"model": model, "reasoning_effort": args.effort, "developer_instructions": h.INSTRUCTIONS,
                   "user_message": args.prompt_payloads[name]}
        h.write_json(folder / f"{name}_request.json", request)
        raw = json.dumps([{"id": s["id"], "predicted": group["valid_labels"][0]} for s in group["samples"]])
        used = {"input_tokens": 80, "cached_input_tokens": 20, "output_tokens": 20} if usage else None
        events = [{"type": "thread.started", "thread_id": "session-" + name}, {"type": "turn.started"}]
        if failed:
            events.append({"type": "turn.failed", "error": {"message": "quota exceeded"}})
            used = None
        else:
            events.extend([{"type": "item.completed", "item": {"type": "agent_message", "text": raw}},
                           {"type": "turn.completed", "usage": used}])
            (folder / f"{name}_attempt1_response.txt").write_text(raw)
            h.write_json(folder / f"{name}_results.json", json.loads(raw))
        (folder / f"{name}_attempt1_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        (folder / f"{name}_attempt1_stderr.txt").write_text("quota exceeded" if failed else "")
        m = h.common_metadata(self.base, self.protocol, name, group)
        m.update(status="failed" if failed else "complete", previous_execution_metadata_file=None,
                 updated_at=h.now(), fingerprint=h.sha(json.dumps({"model": model, "effort": args.effort,
                    "payload": args.prompt_payloads[name], "instructions": h.INSTRUCTIONS,
                    "protocol": "codex-grouped-revision-v3"}, sort_keys=True).encode()),
                 attempts=[{"attempt": 1, "returncode": 1 if failed else 0, "timed_out": False,
                            "elapsed_seconds": 0.002, "events_file": f"{name}_attempt1_events.jsonl",
                            "response_file": f"{name}_attempt1_response.txt", "stderr_file": f"{name}_attempt1_stderr.txt",
                            "usage": used, "malformed_event_lines": []}])
        if not failed:
            m["prediction_sha256"] = h.digest(folder / f"{name}_results.json")
        h.write_json(folder / f"{name}_metadata.json", m)
        with self.call_lock:
            self.active.remove(name)
        if failed:
            raise RuntimeError("quota exceeded")

    def run_fake(self, runner=None, workers=2):
        with patch.object(h.shutil, "which", return_value="/fake/codex"), \
             patch.object(h.subprocess, "check_output", return_value="codex-test"), \
             patch.object(h, "run_group", side_effect=runner or self.fake_runner):
            return h.run(self.base, workers)

    def test_full_run_barriers_resume_and_freeze(self):
        result = self.run_fake()
        self.assertEqual(result["complete_sessions"], 6)
        self.assertEqual(result["observed_total_tokens"], 600)
        self.run_fake()
        self.assertEqual(len(self.calls), 6)
        h.freeze(self.base)
        h.audit_outputs(self.base, True)
        with self.assertRaisesRegex(ValueError, "Freeze already"):
            h.freeze(self.base)
        path = self.base / "predictions" / self.protocol["model"] / "clean_attempt1_stderr.txt"
        path.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Frozen inventory"):
            h.audit_outputs(self.base, True)

    def test_inherited_runner_artifacts_audit_without_inference(self):
        from src.experiments.run_revision_v3 import run_group as inherited

        def fake_subprocess(cmd, *, input, stdout, stderr, **kwargs):
            body = json.loads(input.split("\n\n", 1)[1])
            raw = json.dumps([{"id": sample["id"], "predicted": body["valid_labels"][0]} for sample in body["samples"]])
            response = Path(cmd[cmd.index("-o") + 1])
            response.write_text(raw)
            session = response.name.removesuffix("_attempt1_response.txt")
            events = [{"type": "thread.started", "thread_id": session}, {"type": "turn.started"},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": raw}},
                      {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}}]
            stdout.write("\n".join(json.dumps(e) for e in events))
            return SimpleNamespace(returncode=0)

        with patch.object(h.subprocess, "run", side_effect=fake_subprocess):
            result = self.run_fake(inherited, workers=1)
        self.assertEqual(result["complete_sessions"], 6)
        self.assertEqual(result["observed_total_tokens"], 30)
        h.freeze(self.base)

    def test_partial_execution_never_runs_again(self):
        folder = self.base / "predictions" / self.protocol["model"]
        folder.mkdir(parents=True)
        (folder / "clean_attempt1_events.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "Partial/orphaned"):
            self.run_fake()
        self.assertEqual(self.calls, [])

    def test_failure_stops_and_preserves_denominator(self):
        def runner(*args):
            return self.fake_runner(*args, failed=True)
        result = self.run_fake(runner, workers=1)
        self.assertEqual(result["attempted_sessions"], 1)
        self.assertEqual(result["not_attempted_sessions"], 5)
        self.assertEqual(result["planned_prediction_slots"], 6)
        self.run_fake()
        self.assertEqual(len(self.calls), 1)
        h.freeze(self.base)

    def test_unknown_usage_stops_dispatch(self):
        def runner(*args):
            return self.fake_runner(*args, usage=False)
        result = self.run_fake(runner, workers=1)
        self.assertEqual(result["unknown_usage_sessions"], 1)
        self.assertEqual(result["not_attempted_sessions"], 5)

    def test_token_threshold_and_elapsed_budget(self):
        self.protocol["max_total_tokens"] = 100
        self.seal()
        result = self.run_fake(workers=1)
        self.assertEqual(result["attempted_sessions"], 1)
        self.assertEqual(result["not_attempted_sessions"], 5)

    def test_insufficient_time_means_zero_calls(self):
        self.protocol["timeout_seconds"] = 900
        self.protocol["max_elapsed_seconds"] = 100
        self.seal()
        result = self.run_fake()
        self.assertEqual(result["attempted_sessions"], 0)
        self.assertEqual(result["not_attempted_sessions"], 6)

    def test_changed_prompt_and_private_field_rejected(self):
        prompt = self.base / "inference/prompts/clean.txt"
        prompt.write_text("modified")
        with self.assertRaisesRegex(ValueError, "Prompt hash"):
            h.validate(self.base)
        self.groups["clean"]["samples"][0]["true_label"] = "healthy"
        h.write_json(self.base / "inference/batches.json", {"groups": self.groups})
        self.protocol["input_sha256"] = h.digest(self.base / "inference/batches.json")
        self.seal()
        with self.assertRaisesRegex(ValueError, "Private/unknown"):
            h.validate(self.base)

    def test_unknown_event_and_tool_rejected(self):
        self.run_fake()
        path = self.base / "predictions" / self.protocol["model"] / "clean_attempt1_events.jsonl"
        original = path.read_text()
        for event in [{"type": "unexpected"}, {"type": "item.completed", "item": {"type": "command_execution"}}]:
            path.write_text(original + "\n" + json.dumps(event))
            with self.assertRaisesRegex(ValueError, "Unknown event|Tool/unknown"):
                h.audit_outputs(self.base)
        path.write_text(original)
        raw = path.with_name("clean_attempt1_response.txt")
        raw.write_text("[]")
        with self.assertRaisesRegex(ValueError, "Event/raw text mismatch"):
            h.audit_outputs(self.base)

    def test_duplicate_schedule_and_protocol_hash_rejected(self):
        (self.base / "protocol.sha256").write_text("0" * 64)
        with self.assertRaisesRegex(ValueError, "Protocol hash"):
            h.validate(self.base)
        self.seal()
        h.write_json(self.base / "schedule.json", {"blocks": [["clean", "clean"]]})
        self.protocol["schedule_sha256"] = h.digest(self.base / "schedule.json")
        self.seal()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            h.validate(self.base)


if __name__ == "__main__":
    unittest.main()
