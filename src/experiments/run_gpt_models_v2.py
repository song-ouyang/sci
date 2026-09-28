"""Run all v2 GPT experiments with exact generated TXT payloads.

Each model/group uses a fresh ChatGPT-authenticated Codex HTTPS session.
Actual requested models are preserved; no aliases to other model families.
No answer keys are read. All predictions keep the v2 hexadecimal string IDs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from make_gpt_prompts import load_batches, render_group
DEFAULT_MODELS = ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra"]
INSTRUCTIONS = (
    "This is a text-only controlled classification experiment. Use only the "
    "supplied prompts and your existing knowledge. Do not use tools, inspect "
    "files, browse, delegate, or ask questions. Classify every sample from its "
    "own prompt. Do not infer labels from IDs, sample order, other samples, or "
    "an assumed class distribution. Return only the requested JSON array."
)
DISABLED_FEATURES = [
    "shell_tool", "unified_exec", "apps", "plugins", "multi_agent",
    "multi_agent_v2", "skill_search", "hooks", "goals", "sleep_tool",
    "browser_use", "browser_use_external", "computer_use", "in_app_browser",
    "view_image", "image_generation", "workspace_dependencies", "memories",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate(predictions, group):
    if not isinstance(predictions, list):
        raise ValueError("Response is not a JSON array")
    if len(predictions) != len(group["samples"]):
        raise ValueError("Incomplete prediction count")
    for result, sample in zip(predictions, group["samples"]):
        if not isinstance(result, dict) or set(result) != {"id", "predicted"}:
            raise ValueError("Each prediction must contain only id and predicted")
        if not isinstance(result["id"], str) or re.fullmatch(r"[0-9a-fA-F]{8}", result["id"]) is None or result["id"] != sample["id"]:
            raise ValueError("Prediction IDs do not exactly match input order")
        if not isinstance(result["predicted"], str) or result["predicted"] not in group["valid_labels"]:
            raise ValueError("Prediction label is outside valid_labels")


def parse_raw(raw, group):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON code fence")
        text = "\n".join(lines[1:-1]).strip()
    predictions = json.loads(text)
    validate(predictions, group)
    return predictions


def run_group(args, cli, model, name, group):
    folder = args.output / model
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{name}_results.json"
    metadata_path = folder / f"{name}_metadata.json"
    payload = args.prompt_payloads[name]
    fingerprint = sha(json.dumps({
        "model": model, "effort": args.effort, "payload": payload,
        "instructions": INSTRUCTIONS, "protocol": "codex-grouped-v2-txt",
    }, sort_keys=True).encode())
    if output.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"Existing experiment differs: {output}")
        if metadata.get("status") != "complete":
            raise ValueError(f"Existing prediction lacks complete provenance: {output}")
        if metadata.get("input_sha256") != args.input_sha256 or metadata.get("cli_version") != args.cli_version:
            raise ValueError(f"Existing input/CLI provenance differs: {output}")
        validate(json.loads(output.read_text(encoding="utf-8")), group)
        if metadata.get("prediction_sha256") != sha(output.read_bytes()):
            raise ValueError(f"Existing prediction hash differs: {output}")
        print(f"RESUME {model} {name}: {len(group['samples'])} predictions", flush=True)
        return

    # Preserve logs and metadata from unfinished earlier executions. Attempt
    # numbers are monotonically increasing even when this process is restarted.
    existing_attempts = []
    for path in folder.glob(f"{name}_attempt*_events.jsonl"):
        number = path.name.removeprefix(f"{name}_attempt").removesuffix("_events.jsonl")
        if number.isdigit():
            existing_attempts.append(int(number))
    attempt_offset = max(existing_attempts, default=0)
    previous_metadata = None
    if metadata_path.exists():
        previous_metadata = f"{name}_metadata_before_attempt{attempt_offset + 1}_{time.time_ns()}.json"
        shutil.copyfile(metadata_path, folder / previous_metadata)
    request_path = folder / f"{name}_request.json"
    if request_path.exists():
        shutil.copyfile(request_path, folder / f"{name}_request_before_attempt{attempt_offset + 1}_{time.time_ns()}.json")
    write_json(request_path, {
        "model": model, "reasoning_effort": args.effort,
        "developer_instructions": INSTRUCTIONS, "user_message": payload,
    })
    started = now()
    attempts = []
    for attempt_in_execution in range(1, args.retries + 1):
        attempt = attempt_offset + attempt_in_execution
        prefix = folder / f"{name}_attempt{attempt}"
        raw_path = Path(str(prefix) + "_response.txt")
        events_path = Path(str(prefix) + "_events.jsonl")
        stderr_path = Path(str(prefix) + "_stderr.txt")
        with tempfile.TemporaryDirectory(prefix="sci-gpt-classify-") as empty_cwd:
            cmd = [
                cli, "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "-C", empty_cwd,
                "-m", model, "-s", "read-only",
                "-c", 'model_provider="openai-https"',
                "-c", 'model_providers.openai-https.name="OpenAI HTTPS"',
                "-c", "model_providers.openai-https.supports_websockets=false",
                "-c", "model_providers.openai-https.requires_openai_auth=true",
                "-c", 'approval_policy="never"',
                "-c", f'model_reasoning_effort="{args.effort}"',
                "-c", 'web_search="disabled"',
                "-c", "features.skip_host_skill_discovery=true",
                "-c", "suppress_unstable_features_warning=true",
                "-c", "project_doc_max_bytes=0",
                "-c", "developer_instructions=" + json.dumps(INSTRUCTIONS),
            ]
            for feature in DISABLED_FEATURES:
                cmd += ["-c", f"features.{feature}=false"]
            cmd += ["--json", "-o", str(raw_path), "-"]
            print(f"START {model} {name} ({len(group['samples'])} samples), attempt {attempt}", flush=True)
            tick = time.monotonic()
            timed_out = False
            with events_path.open("w") as out, stderr_path.open("w") as err:
                try:
                    result = subprocess.run(
                        cmd, input=payload, text=True, stdout=out, stderr=err,
                        cwd=empty_cwd, timeout=args.timeout, check=False,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = -1
            duration = round(time.monotonic() - tick, 3)
        events = []
        malformed_event_lines = []
        for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Event is not an object")
                events.append(event)
            except json.JSONDecodeError:
                malformed_event_lines.append(line_number)
            except ValueError:
                malformed_event_lines.append(line_number)
        completed = [e for e in events if e.get("type") == "turn.completed"]
        # Fail closed on unknown item types so newly added tools cannot quietly
        # invalidate the text-only protocol.
        tool_items = [e for e in events if isinstance(e.get("item"), dict)
                      and e["item"].get("type") not in {"reasoning", "agent_message", "error"}]
        errors = [e for e in events if e.get("type") in {"error", "turn.failed"}]
        attempt_info = {
            "attempt": attempt, "returncode": returncode, "timed_out": timed_out,
            "elapsed_seconds": duration, "events_file": events_path.name,
            "response_file": raw_path.name, "stderr_file": stderr_path.name,
            "usage": completed[-1].get("usage") if completed else None,
            "errors": errors,
            "malformed_event_lines": malformed_event_lines,
        }
        attempts.append(attempt_info)
        metadata = {
            "protocol": "codex-grouped-v2-txt", "model_requested": model,
            "reasoning_effort": args.effort, "transport": "codex-cli-chatgpt-auth-https",
            "cli_version": args.cli_version,
            "runner_sha256": getattr(args, "runner_sha256", None),
            "input_sha256": args.input_sha256, "fingerprint": fingerprint,
            "prompt_file": str((args.prompts_dir / f"{name}.txt").resolve()),
            "prompt_txt_sha256": sha(payload.encode("utf-8")),
            "base_instructions_sha256": sha(INSTRUCTIONS.encode("utf-8")),
            "batch": name, "n_samples": len(group["samples"]),
            "started_at": started, "updated_at": now(), "attempts": attempts,
            "tools_observed": len(tool_items), "answer_keys_read": False,
            "independent_session_per_group": True, "independent_session_per_sample": False,
            "temperature": None, "temperature_note": "Not exposed/set by Codex CLI",
            "system_prompt_note": "Codex base instructions retained; user config, project instructions, skills and plugins disabled",
            "status": "pending",
            "previous_execution_metadata_file": previous_metadata,
            "server_resolved_model": None,
            "model_identity_note": "Model requested explicitly; server-resolved model identity is not provided by the CLI event stream",
        }
        write_json(metadata_path, metadata)
        if tool_items:
            metadata["status"] = "invalid_tool_use"
            write_json(metadata_path, metadata)
            raise RuntimeError(f"Tool use invalidated controlled run: {model}/{name}")
        if malformed_event_lines:
            metadata["status"] = "invalid_event_log"
            write_json(metadata_path, metadata)
            raise RuntimeError(f"Malformed events prevent protocol verification: {model}/{name}")
        if returncode == 0 and completed and raw_path.exists():
            try:
                predictions = parse_raw(raw_path.read_text(encoding="utf-8"), group)
            except (ValueError, TypeError) as exc:
                metadata["status"] = "invalid_response"
                metadata["validation_error"] = str(exc)
                write_json(metadata_path, metadata)
                raise RuntimeError(f"Invalid model output: {model}/{name}: {exc}") from exc
            write_json(output, predictions)
            metadata["status"] = "complete"
            metadata["prediction_sha256"] = sha(output.read_bytes())
            write_json(metadata_path, metadata)
            print(f"DONE {model} {name}: {len(predictions)} predictions in {duration}s", flush=True)
            return
        print(f"RETRY {model} {name}: exit={returncode}, completed={bool(completed)}", flush=True)
        if attempt_in_execution < args.retries:
            time.sleep(min(2 ** attempt_in_execution, 30))
    metadata["status"] = "failed"
    write_json(metadata_path, metadata)
    raise RuntimeError(f"Model call failed: {model}/{name}; see {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "gpt_experiment_batches_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gpt_models_v2")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "gpt_prompts_v2")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="medium")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.workers < 1 or args.timeout < 1 or args.retries < 1:
        parser.error("workers, timeout, and retries must be positive")
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate models would race on the same output files")
    if any(not model or model in {".", ".."} or "/" in model or "\\" in model for model in args.models):
        parser.error("Model IDs must be directory-safe names")
    args.runner_sha256 = sha(Path(__file__).read_bytes())
    groups, args.input_sha256 = load_batches(args.input)
    manifest = json.loads((args.prompts_dir / "manifest.json").read_text())
    if manifest["input_sha256"] != args.input_sha256:
        raise ValueError("TXT manifest does not match the current v2 input")
    args.prompt_payloads = {}
    for name, group in groups.items():
        raw_txt = (args.prompts_dir / f"{name}.txt").read_bytes()
        payload = raw_txt.decode("utf-8")
        if payload != render_group(name, group):
            raise ValueError(f"TXT differs from complete original prompts: {name}")
        if sha(raw_txt) != manifest["groups"][name]["txt_sha256"]:
            raise ValueError(f"TXT manifest hash mismatch: {name}")
        args.prompt_payloads[name] = payload
    if args.groups:
        groups = {name: groups[name] for name in args.groups}
    for name, group in groups.items():
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Invalid group name: {name!r}")
        ids = [s["id"] for s in group["samples"]]
        if type(group["n_samples"]) is not int or not ids or len(ids) != group["n_samples"] or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid sample metadata: {name}")
        if any(not isinstance(s["id"], str) or re.fullmatch(r"[0-9a-fA-F]{8}", s["id"]) is None or set(s) != {"id", "prompt"} for s in group["samples"]):
            raise ValueError(f"Unexpected fields in model input: {name}")
        if any(not isinstance(s["prompt"], str) or not s["prompt"] for s in group["samples"]):
            raise ValueError(f"Invalid sample prompt: {name}")
        labels = group["valid_labels"]
        if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError(f"Invalid valid_labels: {name}")
    cli = shutil.which("codex")
    if not cli:
        raise RuntimeError("codex CLI is not installed")
    args.cli_version = subprocess.check_output([cli, "--version"], text=True).strip()
    def run_model(model):
        for name, group in groups.items():
            run_group(args, cli, model, name, group)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_model, model): model for model in args.models}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures.append({"model": pending[future], "error": str(exc)})
                print(f"FAILED {pending[future]}: {exc}", flush=True)
    if failures:
        raise SystemExit(1)
    print("All requested model/group predictions saved and validated.", flush=True)


if __name__ == "__main__":
    main()
