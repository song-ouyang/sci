import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.experiments import domestic_api_v2 as api
from src.experiments import run_domestic_extension_v2 as runner
from src.analysis import audit_domestic_extension as audit
from test_domestic_api import FakeOpener, stream


class BudgetV2Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test", "DASHSCOPE_API_KEY": "test"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "v2"
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(self.base)

    def test_explicit_budgets(self):
        qwen = api.build_payload(api.provider_config("qwen"), api.smoke_batch())
        deepseek = api.build_payload(api.provider_config("deepseek"), api.smoke_batch())
        self.assertEqual(qwen["max_completion_tokens"], 65536)
        self.assertEqual(qwen["thinking_budget"], 16384)
        self.assertNotIn("max_tokens", qwen)
        self.assertEqual(deepseek["max_tokens"], 65536)
        self.assertEqual(deepseek["reasoning_effort"], "high")

    def test_concurrency_is_bounded_and_results_freeze(self):
        original = api.run_batch
        lock = threading.Lock()
        counts = {"active": 0, "peak": 0, "calls": 0}
        def fake(config, batch, *args, **kwargs):
            with lock:
                counts["active"] += 1
                counts["calls"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
            try:
                time.sleep(0.02)
                content = json.dumps([{"id": sid, "predicted": batch["allowed_labels"][0]} for sid in batch["expected_ids"]])
                return original(config, batch, *args, **kwargs, opener=FakeOpener(stream(content)))
            finally:
                with lock:
                    counts["active"] -= 1
        with patch.object(api, "run_batch", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.base, "all"), 0)
            self.assertEqual(runner.run(self.base, "all"), 0)
        self.assertEqual(counts["calls"], 18)
        self.assertEqual(counts["peak"], 2)
        _, _, manifest = audit.freeze(self.base)
        self.assertEqual(manifest["valid_sessions"], 18)

    def test_http_failure_stops_new_submissions(self):
        def fake(config, batch, output, *args, **kwargs):
            folder = output / "fixture"
            folder.mkdir(parents=True)
            (folder / "sha256.json").write_text('{}')
            return folder, {"status": "http_error"}
        with patch.object(api, "run_batch", side_effect=fake) as mock, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.base, "deepseek"), 1)
        self.assertLessEqual(mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
