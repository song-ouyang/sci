"""Audit all frozen revision-v3 outcomes before labels, then score valid outputs.

Run freeze_revision_v3.py after all 27 planned sessions have terminal outcomes.
Semantic failures remain N/A; no later repeat replaces the primary repeat.
This scorer never creates or refreshes a freeze. --validate-only opens no labels.
All bootstrap intervals condition on the observed grouped-session responses.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra"]
BASELINES = ["RF", "SVM", "Majority"]
LABELS = {"cmapss": ["Healthy", "Degrading", "Critical"],
          "tep": ["Normal", "Fault_1", "Fault_2", "Fault_4", "Fault_6", "Fault_7", "Fault_13"]}
SIZES = {"cmapss": 60, "tep": 70}
GROUP_NAMES = [f"{dataset}_{condition}_r{repeat}"
               for dataset, conditions in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]
               for condition in conditions for repeat in [1, 2, 3]]
PROTOCOL = "codex-grouped-revision-v3"
ALLOWED_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error",
                  "item.started", "item.updated", "item.completed"}
ALLOWED_ITEMS = {"reasoning", "agent_message", "error"}
RECOVERABLE_RECONNECT_MESSAGE = (
    "Reconnecting... 1/5 (stream disconnected before completion: Transport error: "
    "network error: error decoding response body)"
)
CI_NOTE = ("Descriptive 95% intervals conditional on the fixed selected devices/runs and observed "
           "grouped-session predictions. Resampling devices/runs does not rerun the model or "
           "capture dependence induced by a shared batch, model/session variability, selection "
           "bias, or benchmark contamination. They are not full independent-experiment intervals.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def exact_fields(value, expected, context):
    for key, wanted in expected.items():
        require(key in value and type(value[key]) is type(wanted) and value[key] == wanted,
                f"{context}: mismatch in {key}")


def local_file(folder, filename):
    require(isinstance(filename, str) and filename not in {"", ".", ".."}
            and Path(filename).name == filename and "\\" not in filename,
            f"Unsafe artifact filename: {filename!r}")
    path = folder / filename
    require(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    return path


def validate_predictions(rows, group, context):
    require(isinstance(rows, list) and len(rows) == group["n_samples"], f"{context}: incomplete predictions")
    for row, sample in zip(rows, group["samples"]):
        require(isinstance(row, dict) and set(row) == {"id", "predicted"}, f"{context}: invalid row schema")
        require(isinstance(row["id"], str) and row["id"] == sample["id"], f"{context}: ID/order mismatch")
        require(isinstance(row["predicted"], str) and row["predicted"] in group["valid_labels"],
                f"{context}: invalid exact label")


def decode_raw(raw):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        require(len(lines) >= 3 and lines[0] in {"```", "```json"} and lines[-1] == "```",
                "Malformed JSON fence")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def parse_raw(raw, group):
    predictions = decode_raw(raw)
    validate_predictions(predictions, group, "Raw response")
    return predictions


def verify_events(path):
    sessions, completed, diagnostics, final_messages, events = [], [], [], [], []
    completed_lines, final_message_lines = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        require(isinstance(event, dict) and event.get("type") in ALLOWED_EVENTS,
                f"{path}:{number}: unknown event")
        item = event.get("item")
        if event["type"].startswith("item.") or item is not None:
            require(isinstance(item, dict) and item.get("type") in ALLOWED_ITEMS,
                    f"{path}:{number}: tool use/unknown item")
        if event["type"] in {"error", "turn.failed"} or (isinstance(item, dict) and item["type"] == "error"):
            diagnostics.append({"line": number, "event": event})
        if event["type"] == "thread.started":
            require(isinstance(event.get("thread_id"), str) and event["thread_id"], f"{path}: no session ID")
            sessions.append(event["thread_id"])
        if event["type"] == "turn.completed":
            completed.append(event)
            completed_lines.append(number)
        if event["type"] == "item.completed" and isinstance(item, dict) and item["type"] == "agent_message":
            require(isinstance(item.get("text"), str), f"{path}: missing agent text")
            final_messages.append(item["text"])
            final_message_lines.append(number)
        events.append(event)
    require(events and len(sessions) <= 1, f"{path}: empty log or multiple sessions")
    return {"sessions": sessions, "completed": completed, "diagnostics": diagnostics,
            "final_messages": final_messages, "last_event_type": events[-1]["type"],
            "completed_lines": completed_lines, "final_message_lines": final_message_lines,
            "top_level_errors": [e for e in events if e["type"] in {"error", "turn.failed"}]}


def inventory_files(output):
    files = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), f"Symlink inside frozen directory: {path}")
        if path.is_file() and path != output / "frozen_manifest.json":
            files[path.relative_to(output).as_posix()] = sha(path.read_bytes())
    return files


def outcome_counts(audit):
    return {"sessions": audit["planned_sessions"], "planned_sessions": audit["planned_sessions"],
            "completed_sessions": audit["completed_sessions"], "failed_sessions": audit["failed_sessions"],
            "predictions": audit["total_valid_predictions"], "total_valid_predictions": audit["total_valid_predictions"],
            "expected_predictions": audit["expected_predictions"]}


def verify_frozen_runs(base, runner_path=None, *, require_frozen=True):
    """This entire function is label-free. No datasets/baseline files are read."""
    runner_path = runner_path or ROOT / "src/experiments/run_revision_v3.py"
    output = base / "predictions"
    protocol, protocol_hash = read_json(base / "protocol.json")
    inputs, input_hash = read_json(base / "inference/batches.json")
    manifest_path = output / "frozen_manifest.json"
    require(manifest_path.is_file() or not require_frozen, "Missing frozen outcome manifest; refusing key access")
    manifest, manifest_hash = read_json(manifest_path) if manifest_path.exists() else (None, None)
    amendment_path = base / "failure_reporting_amendment.json"
    amendment, amendment_hash = read_json(amendment_path) if amendment_path.exists() else (None, None)
    if amendment is not None:
        exact_fields(amendment, {"amendment_id": "failure-reporting-1", "original_protocol_sha256": protocol_hash},
                     "Pre-score failure-reporting amendment")
        amended_at = datetime.fromisoformat(amendment["recorded_at_utc"])
        require(amended_at.tzinfo is not None and amended_at < datetime.now(timezone.utc),
                "Failure-reporting amendment must have a timezone-aware timestamp before scoring")
    transport_note_path = base / "transport_audit_note.json"
    transport_note, transport_note_hash = read_json(transport_note_path) if transport_note_path.exists() else (None, None)
    if transport_note is not None:
        exact_fields(transport_note, {"note_id": "transport-audit-1", "original_protocol_sha256": protocol_hash},
                     "Pre-score transport audit note")
        transport_note_at = datetime.fromisoformat(transport_note["recorded_at_utc"])
        require(transport_note_at.tzinfo is not None and transport_note_at < datetime.now(timezone.utc),
                "Transport audit note must have a timezone-aware timestamp before scoring")
    runner_hash = sha(runner_path.read_bytes())
    exact_fields(protocol, {"version": "revision-v3", "models": MODELS, "reasoning_effort": "medium",
                           "repeats": 3, "expected_groups": GROUP_NAMES, "expected_sessions": 27,
                           "primary_repeat": 1, "primary_metrics": ["macro_f1", "accuracy"],
                           "bootstrap": {"replicates": 10000, "seed": 20260926},
                           "sample_size": SIZES, "input_sha256": input_hash,
                           "independent_sample_calls": False}, "Protocol")
    expected_total = 3 * (3 * SIZES["cmapss"] + 6 * SIZES["tep"])
    actual_files = inventory_files(output)
    if manifest is not None:
        exact_fields(manifest, {"schema": "revision-v3-outcomes-frozen-v2", "protocol_sha256": protocol_hash,
                               "input_sha256": input_hash, "runner_sha256": runner_hash,
                               "sessions": 27, "expected_predictions": expected_total, "answer_keys_read": False,
                               "failure_reporting_amendment_sha256": amendment_hash,
                               "transport_audit_note_sha256": transport_note_hash}, "Freeze")
        require(isinstance(manifest.get("frozen_at"), str) and manifest["frozen_at"], "Missing freeze timestamp")
        require(isinstance(manifest.get("files"), dict) and manifest["files"], "Missing frozen file map")
        require(actual_files == manifest["files"], "Frozen predictions/logs changed or incomplete; refusing key access")
        if amendment is not None:
            require(amended_at <= datetime.fromisoformat(manifest["frozen_at"]),
                    "Failure-reporting amendment was recorded after the outcome freeze")
        if transport_note is not None:
            require(transport_note_at <= datetime.fromisoformat(manifest["frozen_at"]),
                    "Transport audit note was recorded after the outcome freeze")
    groups = inputs.get("groups")
    require(isinstance(groups, dict) and list(groups) == GROUP_NAMES, "Unexpected input group set/order")
    payloads = {}
    for name, group in groups.items():
        dataset, condition, repeat_string = name.split("_")
        repeat = int(repeat_string[1:])
        require(isinstance(group, dict) and set(group) == {
            "dataset", "condition", "repeat", "valid_labels", "n_samples", "samples"}, f"{name}: group schema")
        exact_fields(group, {"dataset": dataset, "condition": condition, "repeat": repeat,
                             "valid_labels": LABELS[dataset], "n_samples": SIZES[dataset]}, name)
        samples = group["samples"]
        require(isinstance(samples, list) and len(samples) == group["n_samples"], f"{name}: sample count")
        require(all(isinstance(row, dict) and set(row) == {"id", "prompt"}
                    and isinstance(row["id"], str) and re.fullmatch(r"[0-9a-f]{8}", row["id"])
                    and isinstance(row["prompt"], str) and row["prompt"].strip() for row in samples),
                f"{name}: invalid IDs/prompts or answer-bearing input fields")
        ids = [row["id"] for row in samples]
        require(len(set(ids)) == len(ids), f"{name}: duplicate IDs")
        prompt_path = local_file(base / "inference/prompts", f"{name}.txt")
        raw = prompt_path.read_bytes()
        require(sha(raw) == protocol["prompt_sha256"].get(name), f"{name}: prompt hash mismatch")
        payload = raw.decode()
        require("\n\n" in payload and json.loads(payload.split("\n\n", 1)[1]) == {
            "valid_labels": group["valid_labels"], "samples": samples}, f"{name}: TXT/JSON mismatch")
        payloads[name] = payload
    for dataset, conditions in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]:
        for condition in conditions:
            first = groups[f"{dataset}_{condition}_r1"]["samples"]
            require(all(groups[f"{dataset}_{condition}_r{repeat}"]["samples"] == first for repeat in [2, 3]),
                    "Repeated sessions must have exactly identical prompts/order")
    require([s["id"] for s in groups["tep_clean_r1"]["samples"]] ==
            [s["id"] for s in groups["tep_misnamed_r1"]["samples"]], "Paired condition IDs/order mismatch")
    expected_results = set()
    actual_results = {path for path in actual_files if path.endswith("_results.json")}
    require({path for path in actual_files if path.endswith("_metadata.json")} ==
            {f"{model}/{name}_metadata.json" for model in MODELS for name in groups},
            "Expected exactly 27 planned outcome metadata files; no missing or extra sessions")
    predictions, verified, sessions, cli_versions, instructions = {}, {}, set(), set(), set()
    observed_logs, failures, recovered_reconnections = set(), [], []
    for model in MODELS:
        predictions[model], verified[model] = {}, {}
        folder = output / model
        for name, group in groups.items():
            context = f"{model}/{name}"
            metadata, _ = read_json(local_file(folder, f"{name}_metadata.json"))
            status = metadata.get("status")
            require(status in {"complete", "invalid_response"},
                    f"{context}: every planned outcome must be terminal (complete or invalid_response), got {status!r}")
            rows, result_hash = None, None
            if status == "complete":
                result_path = local_file(folder, f"{name}_results.json")
                rows, result_hash = read_json(result_path)
                validate_predictions(rows, group, context)
                expected_results.add(result_path.relative_to(output).as_posix())
                require(metadata.get("prediction_sha256") == result_hash, f"{context}: prediction hash mismatch")
            else:
                require(amendment is not None, f"{context}: semantic outcome requires the pre-score failure-reporting amendment")
                require(not (folder / f"{name}_results.json").exists(), f"{context}: failed session has repaired/scored results")
                require(metadata.get("prediction_sha256") is None, f"{context}: failed session has a prediction hash")
                require(isinstance(metadata.get("validation_error"), str) and metadata["validation_error"],
                        f"{context}: missing semantic failure reason")
            exact_fields(metadata, {"protocol": PROTOCOL, "model_requested": model,
                                    "reasoning_effort": "medium", "transport": "codex-cli-chatgpt-auth-https",
                                    "runner_sha256": runner_hash, "protocol_sha256": protocol_hash,
                                    "input_sha256": input_hash, "prompt_txt_sha256": sha(payloads[name].encode()),
                                    "dataset": group["dataset"], "condition": group["condition"],
                                    "repeat": group["repeat"], "batch": name, "n_samples": group["n_samples"],
                                    "tools_observed": 0, "answer_keys_read": False,
                                    "independent_session_per_group": True, "independent_session_per_sample": False}, context)
            require(isinstance(metadata.get("cli_version"), str) and metadata["cli_version"], f"{context}: CLI version")
            cli_versions.add(metadata["cli_version"])
            request, _ = read_json(local_file(folder, f"{name}_request.json"))
            exact_fields(request, {"model": model, "reasoning_effort": "medium", "user_message": payloads[name]}, context)
            instruction = request.get("developer_instructions")
            require(isinstance(instruction, str) and instruction, f"{context}: developer instruction missing")
            require(metadata.get("base_instructions_sha256") == sha(instruction.encode()), f"{context}: instruction hash")
            instructions.add(instruction)
            fingerprint = sha(json.dumps({"model": model, "effort": "medium", "payload": payloads[name],
                                          "instructions": instruction, "protocol": PROTOCOL}, sort_keys=True).encode())
            require(metadata.get("fingerprint") == fingerprint, f"{context}: request fingerprint")
            attempts = metadata.get("attempts")
            require(isinstance(attempts, list) and attempts, f"{context}: missing attempts")
            previous, archives = metadata.get("previous_execution_metadata_file"), set()
            while previous:
                require(previous not in archives, f"{context}: circular archived metadata")
                archives.add(previous)
                archived, _ = read_json(local_file(folder, previous))
                exact_fields(archived, {key: metadata[key] for key in (
                    "protocol", "model_requested", "reasoning_effort", "runner_sha256", "protocol_sha256",
                    "input_sha256", "fingerprint")}, context + " archived")
                require(isinstance(archived.get("attempts"), list), f"{context}: bad archived attempts")
                attempts = archived["attempts"] + attempts
                previous = archived.get("previous_execution_metadata_file")
            numbers, attempt_reports = [], []
            for attempt in attempts:
                require(isinstance(attempt, dict) and type(attempt.get("attempt")) is int and attempt["attempt"] > 0,
                        f"{context}: invalid attempt")
                number = attempt["attempt"]
                numbers.append(number)
                exact_fields(attempt, {"events_file": f"{name}_attempt{number}_events.jsonl",
                                       "response_file": f"{name}_attempt{number}_response.txt",
                                       "stderr_file": f"{name}_attempt{number}_stderr.txt",
                                       "malformed_event_lines": []}, context)
                event_path = local_file(folder, attempt["events_file"])
                local_file(folder, attempt["stderr_file"])
                observed_logs.add(event_path.relative_to(output).as_posix())
                info = verify_events(event_path)
                require(attempt.get("errors") == info["top_level_errors"], f"{context}: errors metadata mismatch")
                require(all(diagnostic["event"] == {"type": "error", "message": RECOVERABLE_RECONNECT_MESSAGE}
                            for diagnostic in info["diagnostics"]),
                        f"{context}: fatal or unknown diagnostic error; only the exact known transport reconnect is eligible")
                for session in info["sessions"]:
                    require(session not in sessions, f"{context}: reused session")
                    sessions.add(session)
                attempt_reports.append({"attempt": number, "returncode": attempt.get("returncode"),
                                        "timed_out": attempt.get("timed_out"), **info})
            require(numbers == sorted(set(numbers)), f"{context}: duplicate/unordered attempts")
            require(numbers == list(range(1, max(numbers) + 1)), f"{context}: missing earlier attempt provenance")
            require(len(attempts) <= 2 and max(numbers) <= 2,
                    f"{context}: more than two predeclared transport attempts")
            reconnect_count = sum(len(report["diagnostics"]) for report in attempt_reports)
            require(len(attempts) + reconnect_count <= 2,
                    f"{context}: CLI attempts plus logged stream reconnections exceed transport ceiling of two")
            require(not any(report["diagnostics"] for report in attempt_reports[:-1]),
                    f"{context}: diagnostic error in an unrecovered earlier attempt")
            # Additional attempts may only follow a transport failure, never an already completed semantic output.
            require(all(not info["completed"] for info in attempt_reports[:-1]),
                    f"{context}: rerun after completed semantic output")
            final, info = attempts[-1], attempt_reports[-1]
            exact_fields(final, {"returncode": 0, "timed_out": False}, context)
            require(len(info["completed"]) == 1 and len(info["sessions"]) == 1
                    and info["last_event_type"] == "turn.completed", f"{context}: missing complete final turn")
            if info["diagnostics"]:
                require(transport_note is not None, f"{context}: recovered reconnect requires pre-score transport audit note")
                require(status == "complete" and info["final_message_lines"]
                        and all(diagnostic["line"] < info["final_message_lines"][-1]
                                and diagnostic["line"] < info["completed_lines"][0] for diagnostic in info["diagnostics"]),
                        f"{context}: reconnect lacks a subsequent complete valid output and turn")
            usage = info["completed"][0].get("usage")
            require(isinstance(usage, dict) and usage and final.get("usage") == usage, f"{context}: usage mismatch")
            raw = local_file(folder, final["response_file"]).read_text()
            try:
                decoded = decode_raw(raw)
                actual_raw_count = len(decoded) if isinstance(decoded, list) else None
            except (ValueError, TypeError):
                actual_raw_count = None
            if status == "complete":
                require(parse_raw(raw, group) == rows, f"{context}: raw response/prediction mismatch")
            else:
                try:
                    parse_raw(raw, group)
                except (ValueError, TypeError, KeyError) as exc:
                    reproduced_error = str(exc)
                else:
                    raise ValueError(f"{context}: claimed semantic failure is actually a valid response")
                failures.append({"model": model, "group": name, "dataset": group["dataset"],
                                 "condition": group["condition"], "repeat": group["repeat"],
                                 "primary": group["repeat"] == 1, "status": status,
                                 "reason": metadata["validation_error"], "reproduced_validation_error": reproduced_error,
                                 "expected_n": group["n_samples"], "actual_raw_count": actual_raw_count,
                                 "session_id": info["sessions"][0]})
            require(info["final_messages"] and info["final_messages"][-1].strip() == raw.strip(),
                    f"{context}: final event/raw response mismatch")
            for diagnostic in info["diagnostics"]:
                recovered_reconnections.append({"model": model, "group": name, "attempt": final["attempt"],
                                                 "session_id": info["sessions"][0], "events_file": final["events_file"],
                                                 "line": diagnostic["line"], "message": diagnostic["event"]["message"],
                                                 "subsequent_valid_output": True, "subsequent_turn_completed": True})
            predictions[model][name] = rows
            verified[model][name] = {"status": status, "session_id": info["sessions"][0],
                                      "n_predictions": len(rows) if rows is not None else 0,
                                      "expected_n": group["n_samples"], "actual_raw_count": actual_raw_count,
                                      "recovered_stream_reconnections": reconnect_count,
                                      "transport_attempts_including_reconnections": len(attempts) + reconnect_count,
                                      "prediction_sha256": result_hash,
                                      "attempts": [{k: v for k, v in report.items() if k != "final_messages"}
                                                   for report in attempt_reports]}
    require({path for path in actual_files if path.endswith("_events.jsonl")} == observed_logs,
            "Unaccounted inference event logs")
    require(actual_results == expected_results, "Prediction files do not exactly match successful outcomes")
    require(len(cli_versions) == 1 and len(instructions) == 1, "Inconsistent CLI/instructions")
    # The requested instructions must agree with the exact frozen runner source.
    import ast
    tree = ast.parse(runner_path.read_text())
    assignment = next((node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "INSTRUCTIONS" for target in node.targets)), None)
    require(assignment is not None and ast.literal_eval(assignment.value) == next(iter(instructions)),
            "Developer instructions do not match frozen runner")
    valid_predictions = sum(run["n_predictions"] for model in verified.values() for run in model.values())
    if failures:
        trigger = amendment.get("trigger")
        require(isinstance(trigger, dict) and any(all(failure.get(key) == value for key, value in trigger.items())
                                                 for failure in failures), "Amendment trigger does not match an observed failure")
    if recovered_reconnections:
        trigger = transport_note.get("trigger")
        require(isinstance(trigger, dict) and any(all(item.get(key) == value for key, value in trigger.items())
                                                 for item in recovered_reconnections),
                "Transport audit note trigger does not match a recovered reconnect")
    audit = {"status": "verified", "protocol_sha256": protocol_hash, "input_sha256": input_hash,
             "frozen_manifest_sha256": manifest_hash, "runner_sha256": runner_hash,
             "frozen_at": manifest["frozen_at"] if manifest else None, "planned_sessions": 27,
             "completed_sessions": 27 - len(failures), "failed_sessions": len(failures),
             "unique_completed_sessions": len({r["session_id"] for model in verified.values()
                                                for r in model.values() if r["status"] == "complete"}),
             "unique_terminal_sessions": len({r["session_id"] for model in verified.values() for r in model.values()}),
             "total_attempt_sessions": len(sessions), "tools_observed": 0,
             "diagnostic_errors_count": sum(len(a["diagnostics"]) for model in verified.values()
                                              for run in model.values() for a in run["attempts"]),
             "recovered_stream_reconnections": len(recovered_reconnections),
             "recovered_stream_reconnection_details": recovered_reconnections,
             "total_cli_attempts": sum(len(run["attempts"]) for model in verified.values() for run in model.values()),
             "total_transport_attempts_including_reconnections": sum(run["transport_attempts_including_reconnections"]
                                                                      for model in verified.values() for run in model.values()),
             "total_predictions": valid_predictions, "total_valid_predictions": valid_predictions,
             "expected_predictions": expected_total, "session_failures": failures,
             "failure_reporting_amendment_sha256": amendment_hash,
             "failure_reporting_amendment_recorded_at_utc": amendment["recorded_at_utc"] if amendment else None,
             "transport_audit_note_sha256": transport_note_hash,
             "transport_audit_note_recorded_at_utc": transport_note["recorded_at_utc"] if transport_note else None,
             "cli_version": next(iter(cli_versions)), "runs": verified, "files_sha256": actual_files}
    if manifest is not None:
        exact_fields(manifest, outcome_counts(audit), "Freeze outcome counts")
        require(manifest.get("session_failures") == failures, "Frozen semantic failure descriptions changed")
        require(manifest.get("recovered_stream_reconnection_details") == recovered_reconnections,
                "Frozen recovered reconnect descriptions changed")
    return protocol, groups, predictions, audit


def load_evaluation_data(base, protocol, groups):
    """Only call after verify_frozen_runs has returned successfully for all runs."""
    data, data_hash = read_json(base / "data/datasets.json")
    require(protocol.get("dataset_sha256") == data_hash, "Dataset differs from pre-inference protocol")
    data_manifest, data_manifest_hash = read_json(base / "data/manifest.json")
    require(data_manifest.get("dataset_sha256") == data_hash, "Dataset manifest hash mismatch")
    datasets = data.get("datasets")
    require(isinstance(datasets, dict) and set(datasets) == set(SIZES), "Dataset set mismatch")
    baseline_manifest, baseline_manifest_hash = read_json(base / "baselines/manifest.json")
    require(baseline_manifest.get("dataset_sha256") == data_hash, "Baseline/dataset manifest mismatch")
    require(isinstance(baseline_manifest.get("files"), dict)
            and set(baseline_manifest["files"]) == {"predictions.json", "scores.json"}, "Baseline manifest file set")
    for filename, expected_hash in baseline_manifest["files"].items():
        require(sha(local_file(base / "baselines", filename).read_bytes()) == expected_hash,
                f"Baseline manifest mismatch: {filename}")
    baseline_predictions, baseline_hash = read_json(base / "baselines/predictions.json")
    baseline_scores, baseline_scores_hash = read_json(base / "baselines/scores.json")
    require(baseline_scores.get("dataset_sha256") == data_hash and baseline_scores.get("seed") == 20260926,
            "Baseline score dataset/seed mismatch")
    require(set(baseline_predictions) == set(SIZES), "Baseline dataset set mismatch")
    ordered, baselines = {}, {}
    for name, ds in datasets.items():
        require(ds.get("valid_labels") == LABELS[name], f"{name}: dataset labels mismatch")
        rows = ds.get("test")
        require(isinstance(rows, list) and len(rows) == SIZES[name], f"{name}: test sample count")
        require(all(isinstance(row, dict) and row.get("label") in LABELS[name]
                    and isinstance(row.get("group"), (str, int)) and not isinstance(row.get("group"), bool)
                    for row in rows), f"{name}: invalid evaluation labels/groups")
        ids = [row.get("id") for row in rows]
        group_ids = [str(row["group"]) for row in rows]
        require(len(set(ids)) == len(ids) and len(set(group_ids)) == len(rows),
                f"{name}: one independent device/run group per test observation required")
        require(all(sum(row["label"] == label for row in rows) == SIZES[name] // len(LABELS[name])
                    for label in LABELS[name]), f"{name}: preregistered test balance mismatch")
        require([{"id": row["id"], "prompt": row["prompt"]} for row in rows] ==
                groups[f"{name}_clean_r1"]["samples"], f"{name}: test/input order or prompt mismatch")
        train = ds.get("train")
        require(isinstance(train, list) and train, f"{name}: missing training provenance")
        require(not set(group_ids).intersection(str(row["group"]) for row in train), f"{name}: train/test group leakage")
        features = ds.get("feature_names")
        require(isinstance(features, list) and features and len(features) == len(set(features)), f"{name}: feature schema")
        require(all(isinstance(row.get("features"), list) and len(row["features"]) == len(features)
                    and all(isinstance(value, (float, int)) and not isinstance(value, bool)
                            and math.isfinite(value) for value in row["features"]) for row in train + rows),
                f"{name}: invalid common feature vectors")
        baseline_info = baseline_scores.get("datasets", {}).get(name)
        require(isinstance(baseline_info, dict) and baseline_info.get("test_tuning") is False
                and baseline_info.get("train_records") == len(train)
                and baseline_info.get("test_records") == len(rows), f"{name}: baseline split/tuning metadata")
        ordered[name] = rows
        baseline = baseline_predictions[name]
        require(isinstance(baseline, dict) and set(baseline) == set(BASELINES), f"{name}: baseline methods mismatch")
        baselines[name] = {}
        for method, predictions in baseline.items():
            require(isinstance(predictions, list) and len(predictions) == len(rows), f"{name}/{method}: incomplete baseline")
            require(all(isinstance(row, dict) and set(row) == {"id", "predicted"}
                        and isinstance(row["id"], str) and row["predicted"] in LABELS[name] for row in predictions),
                    f"{name}/{method}: invalid baseline schema or label")
            by_id = {row["id"]: row["predicted"] for row in predictions}
            require(len(by_id) == len(rows) and set(by_id) == set(ids), f"{name}/{method}: baseline common-sample mismatch")
            baselines[name][method] = [by_id[sid] for sid in ids]
            actual = metric_values([row["label"] for row in rows], baselines[name][method], LABELS[name], name)
            previous = baseline_info.get("metrics", {}).get(method, {})
            require(previous.get("n_samples") == len(rows) and previous.get("correct") == actual["correct"]
                    and math.isclose(previous.get("accuracy_pct", -1), 100 * actual["accuracy"], abs_tol=1e-10)
                    and math.isclose(previous.get("macro_f1_pct", -1), 100 * actual["macro_f1"], abs_tol=1e-10),
                    f"{name}/{method}: baseline metrics do not reproduce from frozen common-sample predictions")
    displays = datasets["tep"].get("feature_display_names")
    require(isinstance(displays, list) and len(displays) == 52 and len(set(displays)) == 52,
            "TEP: expected 52 unique sensor descriptors")
    rotation = {name: displays[(index + 1) % 52] for index, name in enumerate(displays)}
    for clean, altered in zip(groups["tep_clean_r1"]["samples"], groups["tep_misnamed_r1"]["samples"]):
        lines, count = clean["prompt"].splitlines(keepends=True), 0
        for index, line in enumerate(lines):
            field, separator, numeric = line.partition(":")
            if field in rotation:
                lines[index] = rotation[field] + separator + numeric
                count += 1
        require(count == 52 and "".join(lines) == altered["prompt"],
                "TEP paired intervention changed values/instructions or did not rotate exactly 52 descriptors")
    return ordered, baselines, {"datasets_sha256": data_hash, "data_manifest_sha256": data_manifest_hash,
                                "baseline_manifest_sha256": baseline_manifest_hash,
                                "baseline_predictions_sha256": baseline_hash, "baseline_scores_sha256": baseline_scores_hash,
                                "paired_descriptor_rotation_verified": True}


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def metric_values(truth, predicted, labels, dataset):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(len(truth) == len(predicted) and len(truth) > 0, "Metric arrays must have identical nonzero length")
    confusion = {label: {other: int(np.sum((truth == label) & (predicted == other)))
                         for other in labels} for label in labels}
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        per_class[label] = {"support": int(np.sum(truth == label)), "tp": tp, "fp": fp, "fn": fn,
                            "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "recall_wilson95": wilson(tp, tp + fn),
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    correct = int(np.sum(truth == predicted))
    if dataset == "tep":
        affected = truth != "Normal"
        missed = affected & (predicted == "Normal")
        name = "fault_as_normal"
    else:
        affected = truth == "Critical"
        missed = affected & (predicted != "Critical")
        name = "critical_as_noncritical"
    numerator, denominator = int(missed.sum()), int(affected.sum())
    normal_count = int(np.sum(truth == "Normal")) if dataset == "tep" else 0
    false_alarms = int(np.sum((truth == "Normal") & (predicted != "Normal"))) if dataset == "tep" else 0
    return {"n": len(truth), "correct": correct, "accuracy": correct / len(truth),
            "accuracy_wilson95": wilson(correct, len(truth)),
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
            "miss_definition": name, "miss_numerator": numerator, "miss_denominator": denominator,
            "miss_rate": numerator / denominator if denominator else None,
            "miss_rate_wilson95": wilson(numerator, denominator),
            "false_alarm_numerator": false_alarms if dataset == "tep" else None,
            "false_alarm_denominator": normal_count if dataset == "tep" else None,
            "false_alarm_rate": false_alarms / normal_count if normal_count else None,
            "false_alarm_rate_wilson95": wilson(false_alarms, normal_count),
            "critical_as_healthy_count": int(np.sum((truth == "Critical") & (predicted == "Healthy"))) if dataset == "cmapss" else None,
            "per_class": per_class, "confusion_matrix": confusion}


def bootstrap_values(truth, predicted, labels, draws):
    """Common group draws support paired contrasts; each group has one row."""
    truth, predicted = np.asarray(truth)[draws], np.asarray(predicted)[draws]
    macro = np.zeros(len(draws))
    for label in labels:
        actual, guessed = truth == label, predicted == label
        tp = (actual & guessed).sum(axis=1)
        denominator = actual.sum(axis=1) + guessed.sum(axis=1)
        macro += np.divide(2 * tp, denominator, out=np.zeros(len(draws)), where=denominator != 0) / len(labels)
    return {"accuracy": (truth == predicted).mean(axis=1), "macro_f1": macro}


def percentile95(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975], method="linear")]


def unavailable_metrics(expected_n):
    return {"n": expected_n, "correct": None, "accuracy": None, "accuracy_wilson95": [None, None],
            "macro_f1": None, "macro_f1_group_bootstrap95": [None, None],
            "miss_definition": None, "miss_numerator": None, "miss_denominator": None,
            "miss_rate": None, "miss_rate_wilson95": [None, None],
            "false_alarm_numerator": None, "false_alarm_denominator": None, "false_alarm_rate": None,
            "false_alarm_rate_wilson95": [None, None], "critical_as_healthy_count": None,
            "per_class": None, "confusion_matrix": None}


def compute_scores(protocol, groups, predictions, audit, datasets, baselines, provenance):
    metrics, bootstrap, prediction_vectors, draws_by_dataset = [], {}, {}, {}
    failure_map = {f"{item['model']}/{item['group']}": item for item in audit["session_failures"]}
    truth_by_dataset = {dataset: [row["label"] for row in rows] for dataset, rows in datasets.items()}
    for dataset, rows in datasets.items():
        draws_by_dataset[dataset] = np.random.default_rng(protocol["bootstrap"]["seed"]).integers(
            0, len(rows), size=(protocol["bootstrap"]["replicates"], len(rows)))
    for model in MODELS:
        for name, group in groups.items():
            dataset = group["dataset"]
            key = f"{model}/{name}"
            if predictions[model][name] is None:
                failure = failure_map[key]
                metrics.append({"id": key, "model": model, "model_type": "LLM", "dataset": dataset,
                                "condition": group["condition"], "repeat": group["repeat"], "primary": group["repeat"] == 1,
                                "status": "invalid_response", "available": False, "failure_reason": failure["reason"],
                                "n_groups": len(datasets[dataset]), "n_expected": group["n_samples"], "n_scored": 0,
                                **unavailable_metrics(group["n_samples"])})
                continue
            vector = [row["predicted"] for row in predictions[model][name]]
            resampled = bootstrap_values(truth_by_dataset[dataset], vector, LABELS[dataset], draws_by_dataset[dataset])
            values = metric_values(truth_by_dataset[dataset], vector, LABELS[dataset], dataset)
            metrics.append({"id": key, "model": model, "model_type": "LLM", "dataset": dataset,
                            "condition": group["condition"], "repeat": group["repeat"], "primary": group["repeat"] == 1,
                            "status": "complete", "available": True, "failure_reason": None,
                            "n_groups": len(datasets[dataset]), "n_expected": group["n_samples"], "n_scored": len(vector), **values,
                            "macro_f1_group_bootstrap95": percentile95(resampled["macro_f1"])})
            bootstrap[key], prediction_vectors[key] = resampled, vector
    for dataset in SIZES:
        for method in BASELINES:
            key = f"{method}/{dataset}_clean"
            vector = baselines[dataset][method]
            resampled = bootstrap_values(truth_by_dataset[dataset], vector, LABELS[dataset], draws_by_dataset[dataset])
            metrics.append({"id": key, "model": method, "model_type": "supervised_baseline", "dataset": dataset,
                            "condition": "clean", "repeat": None, "primary": True, "n_groups": len(datasets[dataset]),
                            "status": "complete", "available": True, "failure_reason": None,
                            "n_expected": len(vector), "n_scored": len(vector),
                            **metric_values(truth_by_dataset[dataset], vector, LABELS[dataset], dataset),
                            "macro_f1_group_bootstrap95": percentile95(resampled["macro_f1"])})
            bootstrap[key], prediction_vectors[key] = resampled, vector
    metric_map = {row["id"]: row for row in metrics}
    unavailable_contrasts = []

    def paired(a, b, dataset, contrast, repeat):
        left, right = metric_map[a], metric_map[b]
        unavailable = [row for row in [left, right] if not row["available"]]
        if unavailable:
            unavailable_contrasts.append({"contrast": contrast, "dataset": dataset, "a": a, "b": b,
                                          "repeat": repeat, "primary": repeat == 1, "n_expected": len(datasets[dataset]),
                                          "unavailable_sessions": [row["id"] for row in unavailable],
                                          "reason": "; ".join(f"{row['id']}: {row['failure_reason']}" for row in unavailable)})
            return None
        flips = np.asarray(prediction_vectors[a]) != np.asarray(prediction_vectors[b])
        result = {"contrast": contrast, "dataset": dataset, "a": a, "b": b, "repeat": repeat,
                  "primary": repeat == 1, "n": len(flips), "n_groups": len(flips),
                  "difference_direction": "a minus b", "flip_count": int(flips.sum()),
                  "flip_rate": float(flips.mean()),
                  "flip_rate_group_bootstrap95": percentile95(flips[draws_by_dataset[dataset]].mean(axis=1)),
                  "flip_rate_wilson95": wilson(int(flips.sum()), len(flips))}
        for metric in ["accuracy", "macro_f1"]:
            result[f"{metric}_difference"] = left[metric] - right[metric]
            result[f"{metric}_difference_group_bootstrap95"] = percentile95(bootstrap[a][metric] - bootstrap[b][metric])
        return result

    contrasts = []
    for dataset in SIZES:
        methods = [f"{model}/{dataset}_clean_r1" for model in MODELS] + [f"{method}/{dataset}_clean" for method in BASELINES]
        for a, b in itertools.combinations(methods, 2):
            result = paired(a, b, dataset, "clean_method_comparison", 1)
            if result is not None:
                contrasts.append(result)
    for model in MODELS:
        for repeat in [1, 2, 3]:
            result = paired(f"{model}/tep_clean_r{repeat}", f"{model}/tep_misnamed_r{repeat}",
                            "tep", "clean_minus_descriptor_mapping_fault", repeat)
            if result is not None:
                contrasts.append(result)
    variability = []
    for model in MODELS:
        for dataset, conditions in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]:
            for condition in conditions:
                rows = [metric_map[f"{model}/{dataset}_{condition}_r{repeat}"] for repeat in [1, 2, 3]]
                available_repeats = [row["repeat"] for row in rows if row["available"]]
                missing_repeats = [row["repeat"] for row in rows if not row["available"]]
                item = {"model": model, "dataset": dataset, "condition": condition,
                        "n_unique_observations": len(datasets[dataset]), "n_sessions": 3,
                        "n_successful_sessions": len(available_repeats), "available_repeats": available_repeats,
                        "missing_repeats": missing_repeats, "primary_repeat": 1}
                for metric in ["accuracy", "macro_f1", "miss_rate"]:
                    values = [row[metric] for row in rows]
                    available_values = [value for value in values if value is not None]
                    item[metric] = {"repeat1": values[0], "repeat2": values[1], "repeat3": values[2],
                                    "min": min(available_values) if available_values else None,
                                    "max": max(available_values) if available_values else None,
                                    "range": max(available_values) - min(available_values) if len(available_values) >= 2 else None}
                vectors = {repeat: np.asarray(prediction_vectors[f"{model}/{dataset}_{condition}_r{repeat}"])
                           for repeat in available_repeats}
                item["prediction_disagreement"] = []
                item["unavailable_prediction_disagreement"] = []
                for a, b in itertools.combinations([1, 2, 3], 2):
                    if a not in vectors or b not in vectors:
                        item["unavailable_prediction_disagreement"].append({"repeat_a": a, "repeat_b": b,
                                                                           "reason": "One or both sessions failed semantic validation."})
                        continue
                    flips = vectors[a] != vectors[b]
                    item["prediction_disagreement"].append({"repeat_a": a, "repeat_b": b,
                                                            "n": len(flips), "flip_count": int(flips.sum()),
                                                            "flip_rate": float(flips.mean())})
                if len(vectors) >= 2:
                    first = vectors[available_repeats[0]]
                    item["any_repeat_flip_count"] = int(np.logical_or.reduce(
                        [first != vectors[repeat] for repeat in available_repeats[1:]]).sum())
                else:
                    item["any_repeat_flip_count"] = None
                variability.append(item)
    reliability = []
    for model in MODELS:
        for dataset, conditions in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]:
            for condition in conditions:
                selected = [row for row in metrics if row["model"] == model and row["dataset"] == dataset
                            and row["condition"] == condition]
                valid = sum(row["available"] for row in selected)
                reliability.append({"model": model, "dataset": dataset, "condition": condition,
                                    "planned_sessions": 3, "valid_sessions": valid, "failed_sessions": 3 - valid,
                                    "session_success_rate": valid / 3,
                                    "expected_predictions": 3 * len(datasets[dataset]),
                                    "validated_predictions": sum(row["n_scored"] for row in selected)})
    return {"schema_version": 1, "experiment_version": "revision-v3",
            "scored_at_utc": datetime.now(timezone.utc).isoformat(), "audit": audit,
            "evaluation_provenance": provenance, "primary_repeat": 1,
            "session_failures": audit["session_failures"],
            "structured_output_reliability": reliability,
            "units": "All rates and F1 values are fractions; differences are fraction points (multiply by 100 for pp).",
            "statistics": {"bootstrap": protocol["bootstrap"], "alpha": 0.05,
                           "sampling_unit": "device/run group; exactly one test observation per group",
                           "accuracy_interval": "two-sided 95% Wilson score interval, z=1.959963984540054",
                           "macro_f1": "unweighted mean over all fixed valid labels; zero_division=0",
                           "bootstrap_interval": "unstratified paired group percentile interval; numpy linear quantiles",
                           "pairing": "The same group-index draws are reused for all methods, conditions and repeats of a dataset.",
                           "repeat_policy": "Repeat 1 is primary. Repeats 2/3 and min/max report session variation; no pooling as extra samples.",
                           "semantic_failure_policy": "Pre-score failure-reporting amendment: keep every planned terminal outcome; semantic failures have N/A classification metrics and no partial labels are scored. Never substitute a later repeat for primary repeat 1.",
                           "multiple_comparisons": "All prespecified pairs reported without significance selection; intervals are unadjusted descriptive intervals, not familywise tests.",
                           "ci_scope": CI_NOTE,
                           "study_scope": "Exploratory convenience sample; no confirmatory power calculation or population representativeness claim.",
                           "precision_at_accuracy_half": {name: {"n": n, "wilson_half_width": (wilson(n / 2, n)[1] - wilson(n / 2, n)[0]) / 2}
                                                          for name, n in SIZES.items()}},
            "metrics": metrics, "paired_contrasts": contrasts, "unavailable_contrasts": unavailable_contrasts,
            "planned_contrasts": 39, "available_contrasts": len(contrasts), "repeat_variability": variability}


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def flatten_csv(rows):
    flattened = []
    for row in rows:
        values = {}
        for key, value in row.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, list):
                if key.endswith("95"):
                    require(len(value) == 2, f"Unexpected CSV interval: {key}")
                    values[key + "_low"], values[key + "_high"] = value
                else:
                    values[key] = json.dumps(value)
            else:
                values[key] = value
        flattened.append(values)
    return flattened


def write_report(base, report):
    output = base / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "scores.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in [("metrics", report["metrics"]), ("paired", report["paired_contrasts"]),
                       ("unavailable_contrasts", report["unavailable_contrasts"]), ("session_failures", report["session_failures"])]:
        rows = flatten_csv(rows)
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
        temporary = output / f"{name}.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / f"{name}.csv")
    lines = ["# Revision v3: frozen exploratory comparison", "",
             f"Verified all {report['audit']['planned_sessions']} planned terminal outcomes: "
             f"{report['audit']['completed_sessions']} valid sessions and {report['audit']['failed_sessions']} semantic failures, "
             f"with {report['audit']['total_valid_predictions']}/{report['audit']['expected_predictions']} valid predictions before opening answer labels. "
             "Repeat 1 was designated primary before inference. Rates below are percentages.", "", CI_NOTE, "",
             "## Primary clean comparison", "",
             "| Dataset | Method | Accuracy, Wilson 95% | Macro F1, conditional group bootstrap 95% | Misses |",
             "|---|---|---:|---:|---:|"]
    for row in report["metrics"]:
        if row["primary"] and row["condition"] == "clean":
            if not row["available"]:
                lines.append(f"| {row['dataset']} | {row['model']} | N/A | N/A | N/A |")
                continue
            a, f = row["accuracy_wilson95"], row["macro_f1_group_bootstrap95"]
            lines.append(f"| {row['dataset']} | {row['model']} | {100*row['accuracy']:.1f} [{100*a[0]:.1f}, {100*a[1]:.1f}] "
                         f"| {100*row['macro_f1']:.1f} [{100*f[0]:.1f}, {100*f[1]:.1f}] "
                         f"| {row['miss_numerator']}/{row['miss_denominator']} |")
    lines += ["", "C-MAPSS misses are Critical classified as Healthy/Degrading. TEP misses are a fault classified as Normal; "
              "confusion between fault types is an accuracy/F1 error but not a fault-as-normal miss.", "",
              "## Primary descriptor mapping stress test", "",
              "The same TEP observations and numeric strings receive a cyclic +1 rotation of all sensor descriptors, including units. "
              "A positive difference means the clean condition scores higher. This is a synthetic gross mapping defect.", "",
              "| Model | Accuracy difference, pp [95%] | Macro F1 difference, pp [95%] | Prediction flip rate [95%] |",
              "|---|---:|---:|---:|"]
    for row in report["paired_contrasts"]:
        if row["primary"] and row["contrast"] == "clean_minus_descriptor_mapping_fault":
            a, f, flip = (row[name] for name in ["accuracy_difference_group_bootstrap95",
                                                "macro_f1_difference_group_bootstrap95", "flip_rate_group_bootstrap95"])
            lines.append(f"| {row['a'].split('/')[0]} | {100*row['accuracy_difference']:.1f} [{100*a[0]:.1f}, {100*a[1]:.1f}] "
                         f"| {100*row['macro_f1_difference']:.1f} [{100*f[0]:.1f}, {100*f[1]:.1f}] "
                         f"| {100*row['flip_rate']:.1f} [{100*flip[0]:.1f}, {100*flip[1]:.1f}] |")
    for row in report["unavailable_contrasts"]:
        if row["primary"] and row["contrast"] == "clean_minus_descriptor_mapping_fault":
            lines.append(f"| {row['a'].split('/')[0]} | N/A | N/A | N/A |")
    lines += ["", "## Repeated-session sensitivity", "",
              "Three planned sessions reuse the same observations and fixed order. Ranges use the available valid sessions and are not an "
              "independent-observation confidence interval. No repeated predictions are pooled into a larger test set.", "",
              "| Dataset / condition | Model | Accuracy r1 / r2 / r3 | Macro F1 r1 / r2 / r3 |",
              "|---|---|---:|---:|"]
    for row in report["repeat_variability"]:
        display = lambda value: "N/A" if value is None else f"{100*value:.1f}"
        a = " / ".join(display(row['accuracy'][f'repeat{repeat}']) for repeat in [1, 2, 3])
        f = " / ".join(display(row['macro_f1'][f'repeat{repeat}']) for repeat in [1, 2, 3])
        lines.append(f"| {row['dataset']} / {row['condition']} | {row['model']} | {a} | {f} |")
    lines += ["", "## Semantic failures and unavailable comparisons", "",
              "A pre-score reporting amendment retains failed sessions as N/A; it was not an originally prespecified missing-data rule. "
              "No partial response is scored, no missing prediction is manually filled, and no later repeat replaces primary repeat 1.", ""]
    for failure in report["session_failures"]:
        lines.append(f"- {failure['model']}/{failure['group']}: {failure['reason']}; raw array count "
                     f"{failure['actual_raw_count']}, expected {failure['expected_n']}; zero observations scored from this session.")
    lines += ["", f"{report['available_contrasts']}/{report['planned_contrasts']} planned contrasts are available. "
              "Every unavailable contrast and its reason is in unavailable_contrasts.csv; valid contrasts are in paired.csv. "
              "Intervals are unadjusted and descriptive; no significance-based claim selection was performed. "
              "The bootstrap uses 10,000 group resamples with seed 20260926. At n=60/70, worst-case Wilson half-widths "
              "are approximately 12.3/11.4 percentage points, limiting small-effect and ranking conclusions.", ""]
    atomic_text(output / "summary.md", "\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/revision_v3")
    parser.add_argument("--validate-only", action="store_true", help="Audit frozen inference without opening labels or writing files")
    args = parser.parse_args(argv)
    try:
        base = args.root.resolve()
        protocol, groups, predictions, audit = verify_frozen_runs(base)
        if args.validate_only:
            print(f"Verified {audit['planned_sessions']} outcomes: {audit['completed_sessions']} valid, "
                  f"{audit['failed_sessions']} failed; answer keys not read.")
            return 0
        datasets, baselines, provenance = load_evaluation_data(base, protocol, groups)
        report = compute_scores(protocol, groups, predictions, audit, datasets, baselines, provenance)
        # Check the frozen artifacts again before writing derived output.
        verify_frozen_runs(base)
        write_report(base, report)
        print(f"Reported all 27 outcomes, scoring {audit['completed_sessions']} valid sessions with "
              f"{audit['failed_sessions']} N/A failures; report: {base / 'analysis/summary.md'}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
