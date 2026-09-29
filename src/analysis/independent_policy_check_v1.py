"""Independent, standard-library-only check of a completed policy study.

This program cannot run inference or create/repair a freeze. It refuses to parse
prediction contents or labels until an existing response freeze and every
referenced evidence/code hash have passed. It imports no project code.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]
POLICIES = {
    "Single_F": ("single", ("F1",)),
    "Single_R": ("single", ("R1",)),
    "Same": ("gate", ("F1", "F2")),
    "View": ("gate", ("F1", "R2")),
    "Majority_F": ("vote", ("F1", "F2", "F3")),
    "Majority_R": ("vote", ("R1", "R2", "R3")),
    "Unanimity_F": ("gate", ("F1", "F2", "F3")),
    "Unanimity_R": ("gate", ("R1", "R2", "R3")),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    require(path.is_file() and not path.is_symlink(), "Missing/symlinked evidence: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_child(parent, relative):
    require(isinstance(relative, str) and relative, "Invalid evidence path")
    part = Path(relative)
    require(not part.is_absolute() and ".." not in part.parts, "Unsafe evidence path: " + relative)
    path = parent / part
    require(not any((parent.joinpath(*part.parts[:n])).is_symlink()
                    for n in range(1, len(part.parts) + 1)), "Symlink in evidence path")
    require(path.resolve().is_relative_to(parent.resolve()), "Evidence escapes root")
    return path


def verify_frozen_evidence(base):
    """No raw output, metadata or truth is parsed in this initial hash pass."""
    manifest_path = base / "predictions/frozen_manifest.json"
    require(manifest_path.is_file(), "No frozen_manifest: refusing running/unfrozen study")
    require(not (base / "runner.lock").exists(), "Runner lock exists: refusing active/interrupted study")
    protocol = read_json(base / "protocol.json")
    protocol_hash = sha(base / "protocol.json")
    require((base / "protocol.sha256").read_text().strip() == protocol_hash, "Protocol digest changed")
    manifest = read_json(manifest_path)
    require(manifest.get("protocol_sha256") == protocol_hash, "Freeze/protocol mismatch")
    require(protocol.get("schema") == "policy-validation-v1", "Wrong study schema")
    require(manifest.get("schema") == "policy-validation-v1-freeze", "Wrong freeze schema")
    require(protocol.get("dataset_id") in {"hydraulic", "paderborn"}, "Unsupported dataset")
    checked = {}

    def check(path, expected):
        actual = sha(path)
        require(actual == expected, "Evidence hash changed: " + str(path))
        checked[str(path.resolve())] = actual

    for filename, expected in protocol["code_sha256"].items():
        check(safe_child(REPO, filename), expected)
    for filename, expected in protocol["source_files_sha256"].items():
        check(safe_child(REPO, filename), expected)
    for filename, key in {
        "inference/batches.json": "input_sha256",
        "inference/dataset.json": "dataset_sha256",
        "schedule.json": "schedule_sha256",
        "source_audit.json": "source_audit_sha256",
        "answers/truth.json": "answers_sha256",
    }.items():
        check(base / filename, protocol[key])
    for name, expected in protocol["prompts_sha256"].items():
        require(re.fullmatch(r"[a-z0-9_]+", name), "Unsafe prompt group")
        check(base / "inference/prompts" / (name + ".txt"), expected)
    prediction_root = base / "predictions"
    require(not prediction_root.is_symlink(), "Symlinked predictions directory")
    actual_files = {}
    for path in sorted(prediction_root.rglob("*")):
        require(not path.is_symlink(), "Symlinked prediction artifact")
        if path.is_file() and path != manifest_path:
            actual_files[path.relative_to(prediction_root).as_posix()] = sha(path)
    require(isinstance(manifest.get("files"), dict) and actual_files == manifest["files"],
            "Prediction inventory or hash mismatch")
    for name, expected in manifest["files"].items():
        check(safe_child(prediction_root, name), expected)
    checked[str((base / "protocol.json").resolve())] = protocol_hash
    checked[str(manifest_path.resolve())] = sha(manifest_path)
    return protocol, manifest, checked


def usage_total(usage):
    if not isinstance(usage, dict):
        return None
    values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
    return sum(values) if all(type(v) is int and v >= 0 for v in values) else None


def decode_predictions(raw, ids, labels):
    """Reparse the raw text, independently of saved parsed predictions."""
    clean = raw.strip()
    if clean.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", clean, re.DOTALL)
        require(match is not None, "Invalid JSON fence")
        clean = match.group(1).strip()
    values = json.loads(clean)
    require(isinstance(values, list) and len(values) == len(ids), "Wrong raw prediction count")
    for pos, entry in enumerate(values):
        require(isinstance(entry, dict) and set(entry) == {"id", "predicted"}, "Wrong prediction fields")
        require(entry["id"] == ids[pos] and isinstance(entry["id"], str), "Raw ID/order mismatch")
        require(isinstance(entry["predicted"], str) and entry["predicted"] in labels, "Invalid raw label")
    return values


def load_records(base, protocol, groups, pairs):
    """All files were hash-checked before entry; inspect terminal evidence now."""
    folder = base / "predictions" / protocol["model"]
    records, sessions = {}, set()
    assignment = {name: (pair, slot) for pair in pairs for slot, name in enumerate(pair["names"])}
    allowed_events = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error",
                      "item.started", "item.updated", "item.completed"}
    for name, group in groups.items():
        meta = read_json(folder / (name + "_metadata.json"))
        status = meta.get("status")
        require(status in {"complete", "failed", "invalid_response", "not_attempted"},
                "Absent or unauditable terminal status: " + name)
        pair, slot = assignment[name]
        ids = [row["id"] for row in group["samples"]]
        expected_meta = {
            "batch": name, "n_samples": len(ids), "condition": group["condition"],
            "repeat": group["repeat"], "dataset": protocol["dataset_id"],
            "model_requested": protocol["model"], "reasoning_effort": protocol["reasoning_effort"],
            "cli_version": protocol["cli_version"], "input_sha256": protocol["input_sha256"],
            "protocol_sha256": sha(base / "protocol.json"), "prompt_txt_sha256": protocol["prompts_sha256"][name],
            "runner_sha256": protocol["code_sha256"]["src/experiments/run_revision_v3.py"],
            "tools_observed": 0, "answer_keys_read": False,
        }
        for key, value in expected_meta.items():
            require(type(meta.get(key)) is type(value) and meta[key] == value, "Metadata mismatch: " + name + "/" + key)
        row = {"name": name, "status": status, "ids": ids, "predictions": None,
               "tokens": None, "usage": None, "elapsed": None, "start": None,
               "end": meta["updated_at"], "lateness": None}
        if status == "not_attempted":
            require(meta.get("attempts") == [] and meta.get("reason"), "Unexplained non-attempt")
            require(not (folder / (name + "_dispatch.json")).exists(), "Not-attempted request was dispatched")
            records[name] = row
            continue
        attempts = meta.get("attempts")
        require(isinstance(attempts, list) and len(attempts) == 1 and attempts[0].get("attempt") == 1,
                "Attempt count changed")
        attempt = attempts[0]
        require(not meta.get("previous_execution_metadata_file") and not attempt.get("malformed_event_lines"),
                "Retry or malformed event evidence")
        dispatch = read_json(folder / (name + "_dispatch.json"))
        require(dispatch["group"] == name and dispatch["protocol_sha256"] == expected_meta["protocol_sha256"],
                "Dispatch identity mismatch")
        require(dispatch["slot"] == slot and dispatch["block"] == pair["index"], "Dispatch assignment changed")
        require(dispatch["target_offset"] == pair["index"] * protocol["pair_period_seconds"], "Wrong dispatch grid")
        offset = dispatch["monotonic_offset"]
        require(type(offset) in {int, float} and math.isfinite(offset) and offset >= dispatch["target_offset"],
                "Invalid dispatch clock")
        request = read_json(folder / (name + "_request.json"))
        require(request["model"] == protocol["model"] and request["reasoning_effort"] == protocol["reasoning_effort"],
                "Wrong requested model/settings")
        require(request["user_message"] == (base / "inference/prompts" / (name + ".txt")).read_text(),
                "Request differs from prompt")
        require(hashlib.sha256(request["developer_instructions"].encode()).hexdigest() == meta["base_instructions_sha256"],
                "Instruction digest mismatch")
        for key, suffix in (("events_file", "events.jsonl"), ("response_file", "response.txt"), ("stderr_file", "stderr.txt")):
            require(attempt[key] == name + "_attempt1_" + suffix, "Unsafe attempt artifact")
        events = []
        for line in (folder / attempt["events_file"]).read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                require(isinstance(event, dict) and event.get("type") in allowed_events, "Unknown event")
                if event["type"].startswith("item.") or "item" in event:
                    require(isinstance(event.get("item"), dict) and event["item"].get("type") in {"reasoning", "agent_message", "error"},
                            "Tool/unknown item")
                events.append(event)
        started = [e["thread_id"] for e in events if e["type"] == "thread.started"]
        completed = [e for e in events if e["type"] == "turn.completed"]
        finals = [e["item"]["text"] for e in events if e["type"] == "item.completed" and e["item"]["type"] == "agent_message"]
        require(len(started) <= 1 and len(completed) <= 1 and not (set(started) & sessions), "Reused/duplicate session")
        sessions.update(started)
        usage = completed[0].get("usage") if completed else None
        require(attempt.get("usage") == usage, "Usage/event mismatch")
        duration = attempt["elapsed_seconds"]
        require(type(duration) in {int, float} and math.isfinite(duration) and duration >= 0, "Invalid client duration")
        require(type(attempt["returncode"]) is int and type(attempt["timed_out"]) is bool, "Invalid execution state")
        raw_path = folder / attempt["response_file"]
        parsed_path = folder / (name + "_results.json")
        if status in {"complete", "invalid_response"}:
            require(attempt["returncode"] == 0 and not attempt["timed_out"], "Valid output claimed for failed call")
            require(len(started) == len(completed) == len(finals) == 1 and events[-1]["type"] == "turn.completed", "Incomplete trace")
            require(raw_path.read_text().strip() == finals[0].strip(), "Raw/final-event mismatch")
        if status == "complete":
            parsed = decode_predictions(raw_path.read_text(), ids, group["valid_labels"])
            require(parsed == read_json(parsed_path) and sha(parsed_path) == meta["prediction_sha256"], "Saved/raw predictions mismatch")
            row["predictions"] = {entry["id"]: entry["predicted"] for entry in parsed}
        else:
            require(not parsed_path.exists(), "Failed output has saved valid predictions")
            if status == "invalid_response":
                try:
                    decode_predictions(raw_path.read_text(), ids, group["valid_labels"])
                except (ValueError, TypeError):
                    pass
                else:
                    raise ValueError("Invalid-response status is actually valid")
            else:
                require(attempt["returncode"] != 0 or attempt["timed_out"] or not completed or not raw_path.exists(),
                        "Failed status contradicts successful execution")
        row.update(tokens=usage_total(usage), usage=usage, elapsed=duration, start=dispatch["started_at"],
                   lateness=offset - dispatch["target_offset"])
        require(datetime.fromisoformat(row["start"]) <= datetime.fromisoformat(row["end"]), "Negative request envelope")
        records[name] = row
    return records


def choose(labels, kind):
    """Independent fixed-rule implementation; missing calls always abstain."""
    if None in labels:
        return None, "required_call_missing"
    if kind == "single":
        return labels[0], "accepted"
    if kind == "gate":
        return (labels[0], "accepted") if all(x == labels[0] for x in labels[1:]) else (None, "disagreement")
    require(kind == "vote" and len(labels) == 3, "Unsupported fixed policy")
    for label in sorted(set(labels)):
        if labels.count(label) >= 2:
            return label, "accepted"
    return None, "tie"


def explicit_randomization(differences):
    threshold = abs(sum(differences))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        extreme += abs(sum(s * d for s, d in zip(signs, differences))) >= threshold
    return {"extreme_assignments": extreme, "total_assignments": 2 ** len(differences),
            "p": extreme / (2 ** len(differences))}


def policy_summary(rows, name, costs):
    """Count mutually exclusive actions directly, including unavailable bases."""
    counters = {key: 0 for key in (
        "accepted_correct", "accepted_error", "required_call_missing", "disagreement_or_tie", "base_correct",
        "base_errors", "base_missing", "rejected_base_correct", "rejected_base_error",
        "disagreement_rejected_base_correct", "disagreement_rejected_base_error", "corrected_base_error", "harmed_base_correct")}
    for row in rows:
        base = row["base_correct"]
        counters["base_missing" if base is None else ("base_correct" if base else "base_errors")] += 1
        if row["output"] is not None:
            counters[row["action"]] += 1
            counters["corrected_base_error"] += base is False and row["action"] == "accepted_correct"
            counters["harmed_base_correct"] += base is True and row["action"] == "accepted_error"
        else:
            missing = row["reason"] == "required_call_missing"
            counters["required_call_missing" if missing else "disagreement_or_tie"] += 1
            if base is not None:
                suffix = "base_correct" if base else "base_error"
                counters["rejected_" + suffix] += 1
                if not missing:
                    counters["disagreement_rejected_" + suffix] += 1
    n = len(rows)
    accepted = counters["accepted_correct"] + counters["accepted_error"]
    rejected_valid_base = counters["rejected_base_correct"] + counters["rejected_base_error"]
    divide = lambda a, b: a / b if b else None
    result = {"planned": n, "accepted": accepted, **counters,
              "coverage": divide(accepted, n), "selective_error": divide(counters["accepted_error"], accepted),
              "usable_correct_fraction": divide(counters["accepted_correct"], n),
              "error_capture": divide(counters["rejected_base_error"], counters["base_errors"]),
              "false_rejection": divide(counters["rejected_base_correct"], counters["base_correct"]),
              "rejection_break_even_cost_ignoring_extra_calls": None if name.startswith("Majority") else divide(counters["rejected_base_error"], rejected_valid_base)}
    for key in ("planned_requests", "attempted_requests", "known_tokens", "unknown_token_calls", "client_service_seconds_sum"):
        result[key] = sum(row[key] for row in costs if row["policy"] == name)
    require(n == accepted + counters["required_call_missing"] + counters["disagreement_or_tie"], "Action denominator lost")
    return result


def recompute(protocol, groups, pairs, panels, truth, records):
    tables = {name: [] for name in ("requests", "randomized_pairs", "predictions", "policy_cases", "policy_costs", "flips", "incremental_gate_cases")}
    for pair in pairs:
        focal = panels[pair["panel_id"]]["focal_ids"]
        by_condition = {}
        for slot, name in enumerate(pair["names"]):
            record = records[name]
            condition = pair["conditions"][slot]
            pred = record["predictions"]
            available = pred is not None
            row = {"group": name, "panel": pair["panel_id"], "round": pair["round"], "pair": pair["index"],
                   "repeat": pair["repeat"], "slot": slot, "condition": condition, "status": record["status"],
                   "planned_slots": len(record["ids"]), "valid_slots": len(record["ids"]) if available else 0,
                   "usable_correct": sum(pred[i] == truth[i] for i in record["ids"]) if available else 0,
                   "focal_planned": len(focal), "focal_valid": len(focal) if available else 0,
                   "focal_usable_correct": sum(pred[i] == truth[i] for i in focal) if available else 0,
                   "tokens": record["tokens"], "elapsed_seconds": record["elapsed"], "started_at": record["start"],
                   "completed_at": record["end"], "grid_lateness_seconds": record["lateness"]}
            tables["requests"].append(row)
            by_condition[condition] = row
            for identity in record["ids"]:
                label = pred[identity] if available else None
                tables["predictions"].append({"group": name, "panel": pair["panel_id"], "round": pair["round"],
                    "condition": condition, "repeat": pair["repeat"], "id": identity, "focal": identity in focal,
                    "truth": truth[identity], "predicted": label, "correct": label == truth[identity] if available else None})
        f, r = by_condition["F"], by_condition["R"]
        skew = abs((datetime.fromisoformat(f["started_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds()) if f["started_at"] and r["started_at"] else None
        tables["randomized_pairs"].append({"pair": pair["index"], "panel": pair["panel_id"], "round": pair["round"],
            "repeat": pair["repeat"], "both_dispatched": f["status"] != "not_attempted" and r["status"] != "not_attempted",
            "both_valid": f["status"] == r["status"] == "complete", "launch_skew_seconds": skew,
            "f_correct": f["focal_usable_correct"], "r_correct": r["focal_usable_correct"], "focal_n": len(focal),
            "f_minus_r_correct": f["focal_usable_correct"] - r["focal_usable_correct"]})
    pair_rows, request_rows = tables["randomized_pairs"], tables["requests"]
    eligible = all(x["both_dispatched"] and x["launch_skew_seconds"] <= 2 for x in pair_rows)
    eligible = eligible and all(x["grid_lateness_seconds"] is not None and x["grid_lateness_seconds"] <= 3 for x in request_rows)
    enumeration = explicit_randomization([x["f_minus_r_correct"] for x in pair_rows]) if eligible else None
    context = {"pairs_scheduled": len(pair_rows), "pairs_dispatched": sum(x["both_dispatched"] for x in pair_rows),
        "pairs_valid": sum(x["both_valid"] for x in pair_rows),
        "mean_f_minus_r_usable_correct_fraction": sum(x["f_minus_r_correct"] / x["focal_n"] for x in pair_rows) / len(pair_rows),
        "focal_slots_each_condition": sum(x["focal_n"] for x in pair_rows),
        "f_usable_correct": sum(x["f_correct"] for x in pair_rows), "r_usable_correct": sum(x["r_correct"] for x in pair_rows),
        "randomization_schedule_eligible": eligible, "two_sided_sharp_null_p": enumeration["p"] if enumeration else None}
    for block in sorted({x["block"] for x in pairs}):
        selected = [x for x in pairs if x["block"] == block]
        panel, rnd = selected[0]["panel_id"], selected[0]["round"]
        focal = panels[panel]["focal_ids"]
        lookup = {condition + str(pair["repeat"]): name for pair in selected for condition, name in zip(pair["conditions"], pair["names"])}
        per_policy = {}
        for name, (kind, keys) in POLICIES.items():
            calls = [records[lookup[key]] for key in keys]
            base = records[lookup[keys[0][0] + "1"]]["predictions"]
            per_policy[name] = {}
            for identity in focal:
                votes = [None if call["predictions"] is None else call["predictions"][identity] for call in calls]
                output, why = choose(votes, kind)
                base_prediction = base[identity] if base is not None else None
                row = {"block": block, "panel": panel, "round": rnd, "policy": name, "id": identity, "truth": truth[identity],
                    "base_prediction": base_prediction, "base_correct": base_prediction == truth[identity] if base_prediction is not None else None,
                    "output": output, "reason": why, "action": "rejected" if output is None else ("accepted_correct" if output == truth[identity] else "accepted_error")}
                per_policy[name][identity] = row
                tables["policy_cases"].append(row)
            gap = abs((datetime.fromisoformat(calls[-1]["start"]) - datetime.fromisoformat(calls[0]["start"])).total_seconds()) if all(x["start"] for x in calls) else None
            tables["policy_costs"].append({"block": block, "panel": panel, "round": rnd, "policy": name,
                "planned_requests": len(calls), "attempted_requests": sum(x["status"] != "not_attempted" for x in calls),
                "known_tokens": sum(x["tokens"] or 0 for x in calls),
                "unknown_token_calls": sum(x["status"] != "not_attempted" and x["tokens"] is None for x in calls),
                "client_service_seconds_sum": sum(0 if x["elapsed"] is None else x["elapsed"] for x in calls), "check_start_gap_seconds": gap})
        for identity in focal:
            same, view = per_policy["Same"][identity], per_policy["View"][identity]
            s, v = same["output"] is not None, view["output"] is not None
            tables["incremental_gate_cases"].append({"block": block, "panel": panel, "round": rnd, "id": identity,
                "base_correct": same["base_correct"], "same_accept": s, "view_accept": v,
                "view_only_reject": s and not v, "same_only_reject": v and not s,
                "both_checks_valid": same["reason"] != "required_call_missing" and view["reason"] != "required_call_missing"})
        for category in ("within_F", "within_R", "cross_FR"):
            left = [lookup["F" + str(n)] for n in (1, 2, 3)]
            right = [lookup["R" + str(n)] for n in (1, 2, 3)]
            comparisons = itertools.product(left, right) if category == "cross_FR" else itertools.combinations(left if category == "within_F" else right, 2)
            for a, b in comparisons:
                p1, p2 = records[a]["predictions"], records[b]["predictions"]
                valid = p1 is not None and p2 is not None
                tables["flips"].append({"block": block, "panel": panel, "round": rnd, "kind": category,
                    "left": a, "right": b, "planned": len(focal), "paired": len(focal) if valid else 0,
                    "flips": sum(p1[i] != p2[i] for i in focal) if valid else None})
    policies = {name: policy_summary([x for x in tables["policy_cases"] if x["policy"] == name], name, tables["policy_costs"]) for name in POLICIES}
    incremental = {}
    for side in ("view_only_reject", "same_only_reject"):
        for label, correct in (("base_error", False), ("base_correct", True)):
            incremental[side + "_" + label] = sum(x[side] and x["base_correct"] is correct and x["both_checks_valid"] for x in tables["incremental_gate_cases"])
    attempted = [x for x in records.values() if x["status"] != "not_attempted"]
    totals = {"planned_sessions": len(records), "attempted_sessions": len(attempted),
        "complete_sessions": sum(x["status"] == "complete" for x in records.values()),
        "not_attempted_sessions": len(records) - len(attempted),
        "planned_prediction_slots": sum(len(x["ids"]) for x in records.values()),
        "valid_prediction_slots": sum(len(x["ids"]) for x in records.values() if x["status"] == "complete"),
        "observed_total_tokens": sum(x["tokens"] or 0 for x in attempted),
        "unknown_usage_sessions": sum(x["tokens"] is None for x in attempted),
        "usable_correct_slots": sum(x["usable_correct"] for x in request_rows),
        "focal_valid_slots": sum(x["focal_valid"] for x in request_rows),
        "focal_planned_slots": sum(x["focal_planned"] for x in request_rows)}
    return {"totals": totals, "context": context, "policies": policies,
            "incremental_gate_rejections": incremental}, tables, enumeration


def compare_value(actual, expected, location, mismatches):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            mismatches.append({"location": location, "error": "dictionary keys differ"})
            return
        for key in expected:
            compare_value(actual[key], expected[key], location + "/" + key, mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append({"location": location, "error": "list length/type differs"})
            return
        for index, value in enumerate(expected):
            compare_value(actual[index], value, location + "/" + str(index), mismatches)
    elif type(expected) in {int, float}:
        if type(actual) not in {int, float} or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"location": location, "actual": actual, "expected": expected})
    elif type(actual) is not type(expected) or actual != expected:
        mismatches.append({"location": location, "actual": actual, "expected": expected})


def compare_csv(path, rows, mismatches):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed = list(reader)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate/empty CSV header")
    if len(observed) != len(rows):
        mismatches.append({"location": path.name, "actual_rows": len(observed), "expected_rows": len(rows)})
        return
    for index, (actual, expected) in enumerate(zip(observed, rows)):
        if set(actual) != set(expected):
            mismatches.append({"location": path.name + "/" + str(index), "error": "CSV columns differ"})
            continue
        decoded = {}
        for key, value in expected.items():
            text = actual[key]
            if value is None:
                decoded[key] = None if text == "" else text
            elif type(value) is bool:
                decoded[key] = text == "True" if text in {"True", "False"} else text
            elif type(value) is int:
                decoded[key] = int(text) if re.fullmatch(r"-?\d+", text or "") else text
            elif type(value) is float:
                try:
                    decoded[key] = float(text)
                except (ValueError, TypeError):
                    decoded[key] = text
            else:
                decoded[key] = text
        compare_value(decoded, expected, path.name + "/" + str(index), mismatches)


def check(base):
    protocol, manifest, evidence = verify_frozen_evidence(base)
    groups = read_json(base / "inference/batches.json")["groups"]
    dataset = read_json(base / "inference/dataset.json")
    panels = {panel["panel_id"]: panel for panel in dataset["panels"]}
    pairs = read_json(base / "schedule.json")["pairs"]
    truth = read_json(base / "answers/truth.json")["labels"]
    count = 18 if protocol["dataset_id"] == "hydraulic" else 6
    require(len(pairs) == count and len(groups) == 2 * count == protocol["calls_scheduled"], "Planned schedule count changed")
    require([x["index"] for x in pairs] == list(range(count)), "Schedule indices changed")
    names = [name for pair in pairs for name in pair["names"]]
    require(len(names) == len(set(names)) and set(names) == set(groups) == set(protocol["prompts_sha256"]), "Planned group inventory differs")
    for pair in pairs:
        require(len(pair["names"]) == 2 and set(pair["conditions"]) == {"F", "R"}, "Unpaired assignment")
        panel = panels[pair["panel_id"]]
        for condition, name in zip(pair["conditions"], pair["names"]):
            expected_ids = panel["focal_ids"] + panel["background_ids"] if condition == "F" else panel["background_ids"] + panel["focal_ids"]
            require([x["id"] for x in groups[name]["samples"]] == expected_ids, "F/R record order changed")
            require(groups[name]["condition"] == condition and groups[name]["repeat"] == pair["repeat"], "Group/assignment mismatch")
    require(set(truth) == {row["id"] for panel in panels.values() for row in panel["records"]}, "Truth/source ID mismatch")
    require(set(truth.values()) <= set(dataset["allowed_labels"]), "Invalid truth labels")
    records = load_records(base, protocol, groups, pairs)
    recomputed, tables, enumeration = recompute(protocol, groups, pairs, panels, truth, records)
    analysis = base / "analysis"
    score_path = analysis / "scores.json"
    evidence[str(score_path.resolve())] = sha(score_path)
    score = read_json(score_path)
    mismatches = []
    for key in ("totals", "policies", "incremental_gate_rejections"):
        compare_value(score.get(key), recomputed[key], "scores/" + key, mismatches)
    context = {key: value for key, value in score.get("context", {}).items() if key != "assumptions"}
    compare_value(context, recomputed["context"], "scores/context", mismatches)
    for key, expected in {"dataset": protocol["dataset_id"], "protocol_sha256": sha(base / "protocol.json"),
                          "frozen_manifest_sha256": sha(base / "predictions/frozen_manifest.json")}.items():
        compare_value(score.get(key), expected, "scores/" + key, mismatches)
    for key in ("planned_sessions", "attempted_sessions", "complete_sessions", "not_attempted_sessions",
                "planned_prediction_slots", "valid_prediction_slots", "observed_total_tokens", "unknown_usage_sessions"):
        compare_value(manifest.get(key), recomputed["totals"][key], "freeze/" + key, mismatches)
    for name, rows in tables.items():
        path = analysis / (name + ".csv")
        evidence[str(path.resolve())] = sha(path)
        compare_csv(path, rows, mismatches)
    # Check the same bytes remain in place at report creation, including analysis outputs.
    for filename, expected in evidence.items():
        require(sha(Path(filename)) == expected, "Evidence changed during independent check")
    return {"schema": "independent-policy-check-v1", "status": "PASS" if not mismatches else "FAIL",
        "checked_at": datetime.now(timezone.utc).isoformat(), "dataset": protocol["dataset_id"],
        "root": str(base), "frozen_at": manifest["frozen_at"], "checker_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(base / "protocol.json"), "checked_files": evidence,
        "checked_file_count": len(evidence), "csv_row_counts": {k: len(v) for k, v in tables.items()},
        "mismatch_count": len(mismatches), "mismatches": mismatches, "recomputed": recomputed,
        "explicit_sign_enumeration": enumeration, "recomputed_tables": tables,
        "method": "Standard library only; independently reparse raw responses, enumerate all signed assignments, count policy actions, and compare every cell of seven CSVs. No scorer/runner/audit imports.",
        "limits": ["Evidence integrity and numerical agreement do not establish scientific validity or untestable randomization assumptions.",
                   "Only completed valid arrays supply labels; failed, invalid and unattempted calls retain planned denominators and missing-call abstention.",
                   "Randomization p is absent when the frozen schedule is ineligible; no replacement experiment or extrapolation is performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.root.resolve())
    destination = REPO / "results" / ("policy_" + report["dataset"] + "_independent_check.json")
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "mismatches": report["mismatch_count"], "checked_files": report["checked_file_count"]}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
