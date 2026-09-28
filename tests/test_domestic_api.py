import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.experiments import domestic_api as api
from src.experiments import run_domestic_extension as extension


def stream(content=None, finish="stop", done=True, tools=False, usage_only=True):
    if content is None:
        content = '[{"id":"smoke-1","predicted":"OK"}]'
    delta = {"content": content, "reasoning_content": "fixture reasoning"}
    if tools:
        delta["tool_calls"] = [{"id": "test"}]
    chunks = [{"model": "fixture-model", "id": "fixture-id", "choices": [
        {"index": 0, "delta": delta, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
    if usage_only:
        chunks.append({"choices": [], "usage": {"total_tokens": 42}})
    else:
        chunks[-1]["usage"] = {"total_tokens": 42}
    text = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


class Response(io.BytesIO):
    status = 200
    headers = {"x-request-id": "test-request"}


class FakeOpener:
    def __init__(self, data=None, failure=None):
        self.data = data if data is not None else stream()
        self.failure = failure
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request)
        if self.failure:
            raise self.failure
        return Response(self.data)


class DomesticAPITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-secret",
                                          "DASHSCOPE_API_KEY": "unit-test-secret"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = api.smoke_batch()

    def test_provider_parameters_preserve_prompt(self):
        for name in api.PROVIDERS:
            payload = api.build_payload(api.provider_config(name), self.batch)
            self.assertEqual(payload["messages"], self.batch["messages"])
            self.assertNotIn("tools", payload)
            self.assertNotIn("expected_ids", payload)
            self.assertNotIn("allowed_labels", payload)
            self.assertNotIn("temperature", payload)
        self.assertTrue(api.build_payload(api.provider_config("qwen"), self.batch)["enable_thinking"])
        self.assertEqual(api.build_payload(api.provider_config("deepseek"), self.batch)["thinking"],
                         {"type": "enabled"})

    def test_endpoint_rejects_credential_redirect_hosts(self):
        for value in ("http://api.deepseek.com", "https://api.deepseek.com.attacker.invalid",
                      "https://user:password@api.deepseek.com", "https://api.deepseek.com?secret=x"):
            with self.subTest(value=value), patch.dict(os.environ, {"DEEPSEEK_BASE_URL": value}):
                with self.assertRaises(ValueError):
                    api.provider_config("deepseek")

    def test_both_stream_usage_shapes(self):
        for usage_only in (True, False):
            parsed = api.parse_stream(stream(usage_only=usage_only))
            self.assertEqual(parsed["usage"]["total_tokens"], 42)
            self.assertEqual(api.validate_predictions(parsed["content"], self.batch)[0]["predicted"], "OK")

    def test_reject_truncated_and_tool_streams(self):
        for kwargs in ({"done": False}, {"finish": "length"}, {"tools": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.parse_stream(stream(**kwargs))

    def test_complete_batch_validation(self):
        for text in ('[]', '[{"id":"wrong","predicted":"OK"}]',
                     '[{"id":"smoke-1","predicted":"ok"}]',
                     '[{"id":"smoke-1","predicted":"OK","extra":1}]',
                     '[{"id":"wrong","id":"smoke-1","predicted":"OK"}]',
                     'Explanation: [{"id":"smoke-1","predicted":"OK"}]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                api.validate_predictions(text, self.batch)

    def test_single_json_fence_matches_original_protocol(self):
        self.assertEqual(api.validate_predictions(
            '```json\n[{"id":"smoke-1","predicted":"OK"}]\n```', self.batch)[0]["id"], "smoke-1")

    def test_answer_key_rejected(self):
        with self.assertRaises(ValueError):
            api.build_payload(api.provider_config("qwen"), {**self.batch, "answers": ["OK"]})

    def test_missing_key_makes_no_call(self):
        os.environ.pop("DASHSCOPE_API_KEY")
        opener = FakeOpener()
        with self.assertRaises(ValueError):
            api.run_batch(api.provider_config("qwen"), self.batch, self.root, opener=opener)
        self.assertFalse(opener.calls)

    def test_success_archives_hashes_and_omits_secret(self):
        opener = FakeOpener()
        folder, metadata = api.run_batch(api.provider_config("qwen"), self.batch, self.root, opener=opener)
        self.assertEqual(metadata["status"], "valid")
        self.assertEqual(len(opener.calls), 1)
        for name, digest in json.loads((folder / "sha256.json").read_text()).items():
            self.assertEqual(extension.sha(folder / name), digest)
            self.assertNotIn("unit-test-secret", (folder / name).read_text())

    def test_semantic_failure_never_retries_or_writes_predictions(self):
        opener = FakeOpener(stream(content="[]"))
        folder, metadata = api.run_batch(api.provider_config("deepseek"), self.batch, self.root, opener=opener)
        self.assertEqual(metadata["status"], "invalid_response")
        self.assertEqual(len(opener.calls), 1)
        self.assertFalse((folder / "predictions.json").exists())
        self.assertTrue((folder / "response.sse").exists())

    def test_http_failure_redacts_key_without_retry(self):
        failure = HTTPError("https://api.deepseek.com", 401, "Unauthorized", {},
                            io.BytesIO(b'bad credential unit-test-secret'))
        opener = FakeOpener(failure=failure)
        folder, metadata = api.run_batch(api.provider_config("deepseek"), self.batch, self.root, opener=opener)
        self.assertEqual(metadata["status"], "http_error")
        self.assertEqual(len(opener.calls), 1)
        self.assertNotIn("unit-test-secret", (folder / "http_error.txt").read_text())

    def test_network_failure_is_recorded(self):
        opener = FakeOpener(failure=URLError("offline"))
        folder, metadata = api.run_batch(api.provider_config("deepseek"), self.batch, self.root, opener=opener)
        self.assertEqual(metadata["status"], "transport_error")
        self.assertTrue((folder / "sha256.json").exists())

    def test_env_is_literal_and_preserves_existing_config(self):
        path = self.root / ".env"
        path.write_text('DEEPSEEK_API_KEY=other\nQWEN_MODEL="$(touch not-executed)"\n')
        api.load_env(path)
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "unit-test-secret")
        self.assertEqual(os.environ["QWEN_MODEL"], "$(touch not-executed)")

    def test_prepare_matches_original_prompts_and_detects_tampering(self):
        output = self.root / "extension"
        extension.prepare(output)
        plan = extension.verify(output)
        self.assertEqual(len(plan["jobs"]), 18)
        self.assertEqual(len({(j["provider"], j["group"]) for j in plan["jobs"]}), 18)
        for job in plan["jobs"]:
            original = (extension.BASE / "inference/prompts" / (job["group"] + ".txt")).read_text()
            self.assertEqual(job["request"]["messages"][1]["content"], original)
        with self.assertRaises(ValueError):
            extension.prepare(output)
        batch_path = next((output / "batches").glob("*.json"))
        batch_path.write_text('{}')
        with self.assertRaises(ValueError):
            extension.verify(output)

    def test_extension_resume_keeps_terminal_jobs_without_new_calls(self):
        output = self.root / "extension"
        extension.prepare(output)
        calls = []
        original_run = api.run_batch
        def fake_run(*args, **kwargs):
            calls.append(1)
            return original_run(*args, **kwargs, opener=FakeOpener(stream(content="[]")))
        with patch.object(api, "run_batch", side_effect=fake_run):
            self.assertEqual(extension.run(output, "deepseek"), 1)
            self.assertEqual(extension.run(output, "deepseek"), 1)
        self.assertEqual(len(calls), 9)

    def test_extension_refuses_interrupted_job(self):
        output = self.root / "extension"
        extension.prepare(output)
        first = extension.verify(output)["jobs"][0]
        (output / "sessions" / first["provider"] / first["group"]).mkdir(parents=True)
        with self.assertRaises(ValueError):
            extension.run(output, first["provider"])


if __name__ == "__main__":
    unittest.main()
