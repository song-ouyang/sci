"""Label-free, single-attempt hydraulic inference with block barriers and audit.

Only ``run`` can invoke a model. ``validate``, ``audit``, and ``freeze`` are
offline and never open answers. Token limits stop dispatch; the inherited CLI
has no per-request output cap, so already running requests can exceed that
threshold. Missing usage also stops dispatch. Never retry a partial execution.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_revision_v3 import INSTRUCTIONS, parse_raw, run_group, sha, write_json

BASE = ROOT / "results/hydraulic_validation_v1"
RUNNER = "src/experiments/run_revision_v3.py"
WRAPPER = "src/experiments/run_hydraulic_v1.py"
TERMINAL = {"complete", "invalid_response", "invalid_tool_use", "invalid_event_log", "failed", "not_attempted"}
ALLOWED_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error",
                  "item.started", "item.updated", "item.completed"}
ALLOWED_ITEMS = {"reasoning", "agent_message", "error"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    need(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    need(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    return sha(path.read_bytes())


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate(base):
    base = Path(base)
    p = read(base / "protocol.json")
    need(digest(base / "protocol.json") == (base / "protocol.sha256").read_text().strip(), "Protocol hash changed")
    need(p["model"] == "gpt-5.6-sol" and p["reasoning_effort"] == "medium", "Unexpected model/effort")
    for key in ["timeout_seconds", "workers_max", "max_elapsed_seconds", "max_total_tokens"]:
        need(type(p[key]) is int and p[key] > 0, f"Invalid limit: {key}")
    need(p["workers_max"] <= 2, "At most two concurrent requests")
    need(isinstance(p["cli_version"], str) and p["cli_version"], "CLI version required")
    need({RUNNER, WRAPPER} <= set(p["code_sha256"]), "Both runner code hashes required")
    for name, expected in p["code_sha256"].items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts and path.parts[0] in {"src", "tests"}, "Unsafe frozen code path")
        need(digest(ROOT / path) == expected, f"Frozen code changed: {name}")
    need(digest(base / "inference/batches.json") == p["input_sha256"], "Input hash changed")
    need(digest(base / "schedule.json") == p["schedule_sha256"], "Schedule hash changed")
    data = read(base / "inference/batches.json")
    need(set(data) == {"groups"} and isinstance(data["groups"], dict) and data["groups"], "Invalid batches schema")
    groups = data["groups"]
    schedule = read(base / "schedule.json")
    need(set(schedule) == {"blocks"} and isinstance(schedule["blocks"], list) and schedule["blocks"], "Invalid schedule")
    blocks = schedule["blocks"]
    need(all(isinstance(b, list) and b and all(isinstance(n, str) for n in b) for b in blocks), "Invalid block")
    ordered = [n for block in blocks for n in block]
    need(len(ordered) == len(set(ordered)) and set(ordered) == set(groups), "Schedule must contain each group exactly once")
    need(set(p["prompts_sha256"]) == set(groups), "Prompt inventory mismatch")
    for name, group in groups.items():
        need(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name), "Unsafe group name")
        need(set(group) in ({"dataset", "condition", "repeat", "samples", "valid_labels"},
                            {"dataset", "condition", "repeat", "samples", "valid_labels", "n_samples"}), "Unexpected group fields")
        need(group["dataset"] == "hydraulic" and isinstance(group["condition"], str), "Invalid dataset/condition")
        need(type(group["repeat"]) is int and group["repeat"] > 0, "Invalid repeat")
        samples, labels = group["samples"], group["valid_labels"]
        need(isinstance(samples, list) and samples, "Empty samples")
        need(isinstance(labels, list) and labels and all(isinstance(x, str) and x for x in labels), "Invalid labels")
        need(len(labels) == len(set(labels)), "Duplicate labels")
        for sample in samples:
            need(isinstance(sample, dict) and set(sample) == {"id", "prompt"}, "Private/unknown sample fields")
            need(isinstance(sample["id"], str) and re.fullmatch(r"[0-9a-f]{8}", sample["id"]), "Invalid anonymous ID")
            need(isinstance(sample["prompt"], str) and sample["prompt"], "Empty prompt")
        need(len({s["id"] for s in samples}) == len(samples), "Duplicate input IDs")
        need(group.get("n_samples", len(samples)) == len(samples), "Sample count mismatch")
        prompt = base / "inference/prompts" / f"{name}.txt"
        need(digest(prompt) == p["prompts_sha256"][name], f"Prompt hash changed: {name}")
        payload = prompt.read_text(encoding="utf-8")
        need("\n\n" in payload, "Prompt must have instruction and JSON sections")
        need(json.loads(payload.split("\n\n", 1)[1]) == {"valid_labels": labels, "samples": samples}, "Prompt/input mismatch")
    return p, groups, blocks


def event_audit(path):
    """Whitelist event and item types, including logs from unsuccessful calls."""
    events, sessions, finals, completed = [], [], [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        need(isinstance(event, dict) and event.get("type") in ALLOWED_EVENTS, f"Unknown event at {path}:{number}")
        item = event.get("item")
        if event["type"].startswith("item.") or item is not None:
            need(isinstance(item, dict) and item.get("type") in ALLOWED_ITEMS, f"Tool/unknown item at {path}:{number}")
        if event["type"] == "thread.started":
            need(isinstance(event.get("thread_id"), str) and event["thread_id"], "Missing session ID")
            sessions.append(event["thread_id"])
        if event["type"] == "item.completed" and item["type"] == "agent_message":
            need(isinstance(item.get("text"), str), "Missing final text")
            finals.append(item["text"])
        if event["type"] == "turn.completed":
            completed.append(event)
        events.append(event)
    need(len(sessions) <= 1 and len(completed) <= 1, "Multiple sessions/completions")
    return {"events": events, "sessions": sessions, "finals": finals, "completed": completed}


def token_count(usage):
    if not isinstance(usage, dict):
        return None
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    if any(type(x) is not int or x < 0 for x in values):
        return None
    return sum(values)  # cached_input_tokens is already included in input_tokens.


def common_metadata(base, p, name, group):
    return {"batch": name, "dataset": group["dataset"], "condition": group["condition"],
            "repeat": group["repeat"], "n_samples": len(group["samples"]),
            "model_requested": p["model"], "reasoning_effort": p["reasoning_effort"],
            "cli_version": p["cli_version"], "protocol_sha256": digest(base / "protocol.json"),
            "input_sha256": p["input_sha256"], "runner_sha256": p["code_sha256"][RUNNER],
            "prompt_txt_sha256": p["prompts_sha256"][name],
            "base_instructions_sha256": sha(INSTRUCTIONS.encode()), "answer_keys_read": False,
            "tools_observed": 0}


def group_artifacts(folder, name):
    # Anchored suffixes prevent e.g. clean/clean_repeat from owning each
    # other's evidence while still finding attempts 2+, interrupted temp files,
    # and backups created by an accidentally reused inherited runner.
    pattern = re.compile(re.escape(name) + r"_(?:attempt\d+_(?:events\.jsonl|response\.txt|stderr\.txt)|"
                         r"metadata(?:_before[^/]*)?\.json|request(?:_before[^/]*)?\.json|dispatch\.json|results\.json)(?:\.tmp)?$")
    return [path for path in folder.iterdir() if pattern.fullmatch(path.name)] if folder.exists() else []


def group_audit(base, p, name, group):
    folder = base / "predictions" / p["model"]
    metadata = read(folder / f"{name}_metadata.json")
    need(metadata.get("status") in TERMINAL, f"Nonterminal group: {name}")
    for key, value in common_metadata(base, p, name, group).items():
        need(key in metadata and type(metadata[key]) is type(value) and metadata[key] == value, f"Metadata mismatch: {name}/{key}")
    attempts = metadata.get("attempts")
    if metadata["status"] == "not_attempted":
        need(attempts == [] and metadata.get("reason"), "Missing non-attempt reason")
        need({f.name for f in group_artifacts(folder, name)} == {f"{name}_metadata.json"}, "Not-attempted group has execution artifacts")
        return {"group": name, "status": "not_attempted", "n": len(group["samples"]), "usage": None,
                "condition": group["condition"], "repeat": group["repeat"],
                "reason": metadata["reason"], "sessions": [], "predictions": None, "completed_at": metadata["updated_at"]}
    need(isinstance(attempts, list) and len(attempts) == 1 and attempts[0].get("attempt") == 1, "Expected exactly one attempt")
    need(metadata.get("previous_execution_metadata_file") is None, "Previous attempt detected")
    dispatch = read(folder / f"{name}_dispatch.json")
    need(dispatch["group"] == name and dispatch["protocol_sha256"] == digest(base / "protocol.json"), "Dispatch provenance mismatch")
    payload = (base / "inference/prompts" / f"{name}.txt").read_text(encoding="utf-8")
    request = read(folder / f"{name}_request.json")
    need(request == {"model": p["model"], "reasoning_effort": p["reasoning_effort"],
                     "developer_instructions": INSTRUCTIONS, "user_message": payload}, "Request differs from frozen input")
    fingerprint = sha(json.dumps({"model": p["model"], "effort": p["reasoning_effort"], "payload": payload,
                                 "instructions": INSTRUCTIONS, "protocol": "codex-grouped-revision-v3"}, sort_keys=True).encode())
    need(metadata.get("fingerprint") == fingerprint, "Request fingerprint mismatch")
    attempt = attempts[0]
    need(type(attempt.get("returncode")) is int and type(attempt.get("timed_out")) is bool, "Invalid execution status")
    need(type(attempt.get("elapsed_seconds")) in {int, float} and attempt["elapsed_seconds"] >= 0, "Invalid execution duration")
    expected_names = {"events_file": f"{name}_attempt1_events.jsonl", "response_file": f"{name}_attempt1_response.txt",
                      "stderr_file": f"{name}_attempt1_stderr.txt"}
    for key, expected in expected_names.items():
        need(attempt.get(key) == expected, "Unsafe/unexpected artifact filename")
    allowed = {f"{name}_metadata.json", f"{name}_dispatch.json", f"{name}_request.json", f"{name}_results.json", *expected_names.values()}
    need({f.name for f in group_artifacts(folder, name)} <= allowed, "Extra group artifacts/attempts")
    logpath = folder / attempt["events_file"]
    digest(logpath)
    digest(folder / attempt["stderr_file"])
    events = event_audit(logpath)
    need(not attempt.get("malformed_event_lines"), "Malformed events recorded")
    observed_usage = events["completed"][0].get("usage") if events["completed"] else None
    need(attempt.get("usage") == observed_usage, "Usage differs from event log")
    result = folder / f"{name}_results.json"
    raw = folder / attempt["response_file"]
    predictions = None
    if metadata["status"] in {"complete", "invalid_response"}:
        need(attempt["returncode"] == 0 and attempt["timed_out"] is False, "Failed execution claimed valid output")
        need(len(events["sessions"]) == len(events["completed"]) == len(events["finals"]) == 1, "Incomplete final event")
        need(events["events"][-1]["type"] == "turn.completed", "Completion must be last event")
        need(sum(e["type"] == "turn.started" for e in events["events"]) == 1, "Unexpected turn count")
        kinds = [e["type"] for e in events["events"]]
        final_index = next(i for i, e in enumerate(events["events"]) if e["type"] == "item.completed" and e["item"]["type"] == "agent_message")
        need(kinds.index("thread.started") < kinds.index("turn.started") < final_index < kinds.index("turn.completed"), "Invalid session/turn/final ordering")
        need(not any(e["type"] == "turn.failed" for e in events["events"]), "Completed output also reports turn failure")
        digest(raw)
        need(events["finals"][0].strip() == raw.read_text(encoding="utf-8").strip(), "Event/raw text mismatch")
    if metadata["status"] == "complete":
        predictions = parse_raw(raw.read_text(encoding="utf-8"), group)
        need(read(result) == predictions and digest(result) == metadata.get("prediction_sha256"), "Prediction/raw/hash mismatch")
    else:
        need(not result.exists(), "Unsuccessful attempt has results")
        need(metadata["status"] in {"invalid_response", "failed"}, "Protocol violation cannot pass output audit")
        if metadata["status"] == "failed":
            need(attempt["returncode"] != 0 or attempt["timed_out"] or not events["completed"] or not raw.exists(), "Failed status has successful execution evidence")
        if metadata["status"] == "invalid_response":
            try:
                parse_raw(raw.read_text(encoding="utf-8"), group)
            except (ValueError, TypeError):
                pass
            else:
                raise ValueError("Invalid response is actually valid")
    return {"group": name, "status": metadata["status"], "condition": group["condition"], "repeat": group["repeat"],
            "n": len(group["samples"]), "usage": attempt.get("usage"), "sessions": events["sessions"],
            "predictions": predictions, "started_at": dispatch["started_at"], "completed_at": metadata["updated_at"],
            "block": dispatch["block"], "elapsed_seconds": attempt["elapsed_seconds"]}


def inventory(base):
    folder = base / "predictions"
    files = {}
    for path in sorted(folder.rglob("*")):
        need(not path.is_symlink(), "Symlinked prediction artifact")
        if path.is_file() and path != folder / "frozen_manifest.json":
            files[path.relative_to(folder).as_posix()] = digest(path)
    return files


def audit_outputs(base, require_frozen=False, *, _owns_lock=False):
    base = Path(base)
    p, groups, blocks = validate(base)
    need(_owns_lock or not (base / "runner.lock").exists(), "Runner is active/interrupted")
    records, sessions, earlier_end = [], set(), None
    for index, block in enumerate(blocks):
        ends = []
        for name in block:
            record = group_audit(base, p, name, groups[name])
            for sid in record["sessions"]:
                need(sid not in sessions, "Session reused between groups")
                sessions.add(sid)
            if record["status"] != "not_attempted":
                need(record["block"] == index, "Wrong schedule block")
                start, end = (datetime.fromisoformat(record[k]) for k in ["started_at", "completed_at"])
                need(start <= end and (earlier_end is None or start >= earlier_end), "Block barrier violated")
                ends.append(end)
            records.append(record)
        if ends:
            earlier_end = max(ends)
    files = inventory(base)
    allowed_files = {"execution_state.json"}
    for name in groups:
        allowed_files.update(f"{p['model']}/{name}_{suffix}" for suffix in [
            "metadata.json", "dispatch.json", "request.json", "results.json", "attempt1_events.jsonl",
            "attempt1_response.txt", "attempt1_stderr.txt"])
    need(set(files) <= allowed_files, "Unexpected files in predictions inventory")
    if require_frozen:
        manifest = read(base / "predictions/frozen_manifest.json")
        need(manifest["files"] == files and manifest["protocol_sha256"] == digest(base / "protocol.json"), "Frozen inventory/protocol mismatch")
    return p, groups, records, files


def summary(records):
    attempted = [r for r in records if r["status"] != "not_attempted"]
    return {"planned_sessions": len(records), "attempted_sessions": len(attempted),
            "complete_sessions": sum(r["status"] == "complete" for r in records),
            "not_attempted_sessions": len(records) - len(attempted),
            "planned_prediction_slots": sum(r["n"] for r in records),
            "valid_prediction_slots": sum(r["n"] for r in records if r["status"] == "complete"),
            "observed_total_tokens": sum(token_count(r["usage"]) or 0 for r in attempted),
            "unknown_usage_sessions": sum(token_count(r["usage"]) is None for r in attempted)}


def freeze(base):
    base = Path(base)
    path = base / "predictions/frozen_manifest.json"
    need(not path.exists(), "Freeze already exists; refusing overwrite")
    lock = base / "runner.lock"
    exclusive_json(lock, {"operation": "freeze", "started_at": now()})
    try:
        _, _, records, files = audit_outputs(base, False, _owns_lock=True)
        manifest = {"schema": "hydraulic-v1-freeze", "frozen_at": now(),
                    "protocol_sha256": digest(base / "protocol.json"), "files": files, **summary(records)}
        exclusive_json(path, manifest)
    finally:
        lock.unlink(missing_ok=True)
    return manifest


def run(base, workers=None):
    base = Path(base)
    p, groups, blocks = validate(base)
    workers = p["workers_max"] if workers is None else workers
    need(type(workers) is int and 1 <= workers <= p["workers_max"], "Invalid worker count")
    need(not (base / "predictions/frozen_manifest.json").exists(), "Already frozen")
    cli = shutil.which("codex")
    need(cli is not None, "Codex CLI not installed")
    need(subprocess.check_output([cli, "--version"], text=True).strip() == p["cli_version"], "CLI version changed")
    lock = base / "runner.lock"
    exclusive_json(lock, {"started_at": now()})
    folder = base / "predictions" / p["model"]
    statepath = base / "predictions/execution_state.json"
    state = None
    try:
        need(not (base / "predictions/frozen_manifest.json").exists(), "Already frozen")
        existing = {}
        for block in blocks:
            for name in block:
                if (folder / f"{name}_metadata.json").exists():
                    existing[name] = group_audit(base, p, name, groups[name])
                else:
                    need(not group_artifacts(folder, name), f"Partial/orphaned files; will not retry: {name}")
        for index, block in enumerate(blocks):
            if any(n in existing and existing[n]["status"] != "not_attempted" for later in blocks[index + 1:] for n in later):
                need(all(n in existing for n in block), "Later block attempted before earlier block finished")
        if statepath.exists():
            state = read(statepath)
            need(state["protocol_sha256"] == digest(base / "protocol.json"), "Execution-state protocol changed")
        else:
            need(not existing, "Existing results have no execution state")
            state = {"started_at": now(), "protocol_sha256": digest(base / "protocol.json"), "stop_reason": None,
                     "token_limit_semantics": "Dispatch threshold; in-flight calls can exceed it; unknown usage halts dispatch."}
            exclusive_json(statepath, state)
        elapsed_origin = datetime.fromisoformat(state["started_at"])
        records = list(existing.values())
        stop_reason = state.get("stop_reason")
        args = SimpleNamespace(output=base / "predictions", prompts_dir=base / "inference/prompts",
            runner_sha256=p["code_sha256"][RUNNER], protocol_sha256=digest(base / "protocol.json"),
            input_sha256=p["input_sha256"], effort=p["reasoning_effort"], timeout=p["timeout_seconds"],
            retries=1, cli_version=p["cli_version"],
            prompt_payloads={n: (base / "inference/prompts" / f"{n}.txt").read_text(encoding="utf-8") for n in groups})

        def gate():
            if stop_reason:
                return stop_reason
            if any(r["status"] != "not_attempted" and token_count(r["usage"]) is None for r in records):
                return "unknown_usage"
            if any(r["status"] == "failed" for r in records):
                return "prior_failed_execution"
            if sum(token_count(r["usage"]) or 0 for r in records) >= p["max_total_tokens"]:
                return "token_dispatch_threshold"
            if (datetime.now(timezone.utc) - elapsed_origin).total_seconds() >= p["max_elapsed_seconds"]:
                return "elapsed_time_budget"
            return None

        def execute(name, index):
            exclusive_json(folder / f"{name}_dispatch.json", {"group": name, "block": index,
                           "started_at": now(), "protocol_sha256": args.protocol_sha256})
            # The shared immutable runner writes request, events, raw response and metadata.
            try:
                run_group(args, cli, p["model"], name, groups[name])
            except Exception:
                metadata = folder / f"{name}_metadata.json"
                need(metadata.exists() and read(metadata).get("status") in TERMINAL,
                     f"Interrupted execution without terminal metadata: {name}; cannot retry")
            return group_audit(base, p, name, groups[name])

        pending_error = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, block in enumerate(blocks):
                queue = [n for n in block if n not in existing]
                running = {}
                while queue or running:
                    stop_reason = gate()
                    while queue and len(running) < workers and not stop_reason:
                        remaining = p["max_elapsed_seconds"] - (datetime.now(timezone.utc) - elapsed_origin).total_seconds()
                        if remaining < p["timeout_seconds"]:
                            stop_reason = "insufficient_time_for_request_timeout"
                            break
                        name = queue.pop(0)
                        running[pool.submit(execute, name, index)] = name
                    if not running:
                        break
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        name = running.pop(future)
                        try:
                            record = future.result()
                            records.append(record)
                            existing[name] = record
                            if record["status"] == "failed":
                                stop_reason = "authentication_quota_or_transport_failure"
                        except Exception as exc:
                            pending_error = exc
                            stop_reason = "audit_or_execution_violation"
                    state.update(stop_reason=stop_reason, updated_at=now(), **summary(records))
                    write_json(statepath, state)
                # This loop drains every in-flight request before entering the next block.
                if stop_reason:
                    break
        stop_reason = gate()
        if stop_reason:
            for block in blocks:
                for name in block:
                    if not (folder / f"{name}_metadata.json").exists() and not group_artifacts(folder, name):
                        metadata = common_metadata(base, p, name, groups[name])
                        metadata.update(status="not_attempted", attempts=[], reason=stop_reason, updated_at=now())
                        exclusive_json(folder / f"{name}_metadata.json", metadata)
        state.update(stop_reason=stop_reason, updated_at=now(), finished_at=now())
        write_json(statepath, state)
        if pending_error:
            raise RuntimeError("Audit/execution violation; evidence retained; never retry affected groups") from pending_error
    finally:
        lock.unlink(missing_ok=True)
    _, _, records, _ = audit_outputs(base)
    state.update(**summary(records))
    write_json(statepath, state)
    return summary(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run", "audit", "freeze"])
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    base = args.root.resolve()
    if args.command == "validate":
        _, groups, blocks = validate(base)
        result = {"groups": len(groups), "blocks": len(blocks), "label_access": False}
    elif args.command == "run":
        result = run(base, args.workers)
    elif args.command == "audit":
        _, _, records, _ = audit_outputs(base, (base / "predictions/frozen_manifest.json").exists())
        result = summary(records)
    else:
        result = freeze(base)
        result = {k: v for k, v in result.items() if k != "files"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
