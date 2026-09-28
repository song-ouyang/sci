"""Synthetic parser/provenance regression tests; no credentials or model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


MODULE = Path(__file__).resolve().parents[1] / "src/experiments/run_v2_experiments.py"
spec = importlib.util.spec_from_file_location("safe_v2_runner", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
LABELS = runner.VALID_LABELS["tep"]


def response(text="Fault_13", *, reason="end_turn", content=None):
    body = {"id": "synthetic", "model": "synthetic-model", "stop_reason": reason,
            "content": content if content is not None else [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 20}}
    obj = Mock()
    obj.model_dump.return_value = body
    return obj


class ParsingTests(unittest.TestCase):
    def test_exact_labels_do_not_overlap(self):
        for label in LABELS:
            with self.subTest(label=label):
                self.assertEqual(runner.parse_label(label, LABELS), label)

    def test_reasoning_requires_one_final_conclusion(self):
        self.assertEqual(runner.parse_label(
            "Normal was considered, but Fault_1 is not supported.\nFinal answer: Fault_13", LABELS),
            "Fault_13")
        self.assertEqual(runner.parse_label("Evidence favors a fault.\n<final>Fault_13</final>", LABELS),
                         "Fault_13")
        for text in ("Normal or Fault_13", "The diagnosis is Fault_13", "Fault_1 Fault_13",
                     "Answer: Fault_1\nFinal: Fault_13", "Answer: Fault_13\nAnswer: Fault_13",
                     "Final: Fault_13\nAdditional explanation", "Final: Fault_1 or Fault_13",
                     "Final: Fault_13.", "normal", "FAULT_13", "Final: fault_13", "", "   "):
            with self.subTest(text=text):
                self.assertEqual(runner.parse_label(text, LABELS), "UNPARSEABLE")

    def test_truncation_is_not_a_prediction(self):
        for reason in ("max_tokens", None, "tool_use", "refusal"):
            self.assertEqual(runner.parse_label("Fault_13", LABELS, stop_reason=reason), "UNPARSEABLE")
            client = Mock()
            client.messages.create.return_value = response(reason=reason)
            self.assertEqual(runner.classify_one(client, "model", "prompt"), "")
            self.assertIsNotNone(runner.classify_one(
                client, "model", "prompt", return_response=True)["response_error"])

    def test_missing_and_empty_text_blocks_fail_closed(self):
        for content, error in (([], "no_text_block"),
                               ([{"type": "thinking", "text": "Fault_13"}], "no_text_block"),
                               ([{"type": "text", "text": " "}], "empty_text")):
            details = runner.extract_response(response(content=content).model_dump())
            self.assertEqual(details["response_error"], error)

    def test_full_response_and_all_text_blocks_are_preserved(self):
        long_text = "Evidence " * 300
        contents = [{"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": long_text},
                    {"type": "text", "text": "Final: Fault_13"}]
        client = Mock()
        original = response(content=contents)
        client.messages.create.return_value = original
        details = runner.classify_one(client, "model", "prompt", return_response=True)
        self.assertEqual(details["response"], original.model_dump())
        self.assertEqual(details["raw"], long_text + "\nFinal: Fault_13")
        self.assertEqual(runner.parse_label(details["raw"], LABELS), "Fault_13")
        self.assertEqual(client.messages.create.call_args.kwargs["max_tokens"], 4096)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.batch = self.base / "batches"
        self.batch.mkdir()
        self.out = self.base / "new_results"
        self.input = self.batch / "tep_zero_shot_v2_1.json"
        self.input.write_text(json.dumps([{"id": "00000001", "prompt": "synthetic prompt"}]))
        self.result = self.out / "tep_zero_shot_v2_1_haiku.json"
        self.meta = self.result.with_suffix(".metadata.json")
        self.client = Mock()
        self.client.messages.create.return_value = response("Normal considered.\nFinal: Fault_13")
        self.batch_patch = patch.object(runner, "BATCH_DIR", self.batch)
        self.batch_patch.start()
        self.addCleanup(self.batch_patch.stop)
        self.api_patch = patch.object(runner.anthropic, "Anthropic", return_value=self.client)
        self.api = self.api_patch.start()
        self.addCleanup(self.api_patch.stop)

    def run_once(self):
        return runner.run_experiment("tep", "zero_shot", 1, "haiku", workers=1,
                                     output_dir=self.out)

    def test_saves_full_response_without_reading_answer_key_and_reuses_valid_cache(self):
        first = self.run_once()
        self.assertEqual(first[0]["predicted"], "Fault_13")
        self.assertEqual(first[0]["stop_reason"], "end_turn")
        self.assertNotIn("true_label", first[0])
        self.assertEqual(first[0]["response"]["usage"]["output_tokens"], 20)
        self.assertEqual(self.run_once(), first)
        self.api.assert_called_once()
        self.assertFalse(json.loads(self.meta.read_text())["provenance"]["answer_keys_read"])

    def test_old_cache_fails_without_call_or_overwrite(self):
        self.out.mkdir()
        original = '[{"id":"00000001","predicted":"Normal"}]'
        self.result.write_text(original)
        with self.assertRaisesRegex(ValueError, "no provenance"):
            self.run_once()
        self.api.assert_not_called()
        self.assertEqual(self.result.read_text(), original)

    def test_modified_input_fails_without_call_or_overwrite(self):
        self.run_once()
        before = self.result.read_bytes()
        self.input.write_text(self.input.read_text().replace("synthetic prompt", "different prompt"))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.run_once()
        self.api.assert_called_once()
        self.assertEqual(self.result.read_bytes(), before)

    def test_modified_model_parser_or_budget_fails(self):
        self.run_once()
        with patch.dict(runner.MODELS, {"haiku": {"model_id": "different-model", "suffix": "_haiku.json"}}):
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.run_once()
        with patch.object(runner, "PARSER_PROTOCOL", "new-parser"):
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.run_once()
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            runner.run_experiment("tep", "zero_shot", 1, "haiku", output_dir=self.out, max_tokens=100)
        self.api.assert_called_once()

    def test_modified_output_and_incomplete_cache_fail(self):
        self.run_once()
        original = self.result.read_bytes()
        self.result.write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_once()
        self.result.write_bytes(original)
        metadata = json.loads(self.meta.read_text())
        metadata["status"] = "running"
        self.meta.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.run_once()
        self.api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
